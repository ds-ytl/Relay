//! The Function Calling tool layer shared by all three agents.
//!
//! The understanding turn, the execution turn and the verification turn all follow the same
//! protocol: the model only calls tools and finishes with `submit_*`; tool results are fed back
//! verbatim through `function_call_output`. Tool definitions and read-only tool
//! implementations live here so that each agent does not carry its own copy.

use std::path::{Path, PathBuf};

use relay_core::Workspace;
use relay_openai::{
    InputItem, LanguageModel, SearchRequest, SearchResult, ToolCall, ToolDef, WebTools,
};
use serde_json::{json, Value};

use crate::vision;

/// Cap on the output of a single tool call, so the context never blows up.
pub(crate) const TOOL_OUTPUT_LIMIT: usize = 4000;

/// Result of one tool call: a text receipt plus an optional image (for screenshot tools).
pub(crate) struct ToolOutcome {
    pub text: String,
    pub image: Option<PathBuf>,
}

pub(crate) fn bash_tool_def() -> ToolDef {
    ToolDef::new(
        "bash",
        "Run a Bash script inside the workspace directory and return exit_code, stdout and stderr. Do one thing at a time.",
        json!({
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Bash script snippet to run"},
                "description": {"type": "string", "description": "One sentence describing what this command does"}
            },
            "required": ["command"]
        }),
    )
}

pub(crate) fn read_file_def() -> ToolDef {
    ToolDef::new(
        "read_file",
        "Read the first lines of any text file, to check its content.",
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path"},
                "max_lines": {"type": "integer", "description": "Maximum number of lines to read, default 60"}
            },
            "required": ["path"]
        }),
    )
}

pub(crate) fn list_files_def() -> ToolDef {
    ToolDef::new(
        "list_files",
        "List files under the workspace directory (two levels deep, skipping hidden directories). \
Omit workspace to list the current workspace; when there is no workspace yet, list all of them. \
subdir may be a relative directory or an absolute path.",
        json!({
            "type": "object",
            "properties": {
                "workspace": {"type": "string", "description": "Workspace name or id; omit for the current workspace"},
                "subdir": {"type": "string", "description": "Relative subdirectory, e.g. outputs; an absolute path is also accepted"},
                "limit": {"type": "integer", "description": "Maximum entries returned per directory, default 40"}
            },
            "required": []
        }),
    )
}

pub(crate) fn read_image_def() -> ToolDef {
    ToolDef::new(
        "read_image",
        "Recognize the content of local images (PNG/JPEG/GIF/WebP/BMP): extract text, table fields, \
numbers and UI elements. Images are downscaled before upload to keep cost down; 2-9 images of the \
same kind (e.g. several screenshots of one page) should be combined into one image before calling.",
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "description": "Image path, or an array of paths (at most 3 images)",
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}, "minItems": 1}
                    ]
                },
                "prompt": {"type": "string", "description": "What to extract from the image, e.g. \"read out every row of this table\""},
                "max_edge": {
                    "type": "integer",
                    "enum": [768, 960, 1024, 1280, 1568],
                    "default": 1024,
                    "description": "Long-edge pixels to downscale to before upload. Increase it to read small text, at the cost of more tokens."
                }
            },
            "required": ["path", "prompt"],
            "additionalProperties": false
        }),
    )
}

pub(crate) fn search_def() -> ToolDef {
    ToolDef::new(
        "search",
        "Search the web and get ranked results (title, URL, snippet, summary). Use it for current \
facts, documentation, prices or anything that is not on the local machine; pass site:, -site: or \
quoted phrases to narrow it. Set need_content only for the one or two results worth reading in full.",
        json!({
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search terms, 1-128 characters"},
                "count": {"type": "integer", "description": "Number of results, 1-50, default 10"},
                "need_content": {"type": "boolean", "description": "Also return each page body, default false"},
                "ori_question": {"type": "string", "description": "The user's original question (1-128 characters); helps understand intent"}
            },
            "required": ["query"],
            "additionalProperties": false
        }),
    )
}

pub(crate) fn fetch_def() -> ToolDef {
    ToolDef::new(
        "fetch",
        "Fetch one web page and return its title and Markdown body. Use it to read a URL the user \
pasted or a search result in full; it cannot run JavaScript, so interactive pages come back partial.",
        json!({
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Absolute http(s) URL, at most 8192 characters"}
            },
            "required": ["url"],
            "additionalProperties": false
        }),
    )
}

pub(crate) fn submit_result_def() -> ToolDef {
    ToolDef::new(
        "submit_result",
        "Close out this execution turn: write back the phase, the conclusion and the artifacts. \
Call it whenever the task is completed, failed, needs confirmation or is paused; \
not calling it is treated as continue and moves on to the next turn. Before phase=done, first \
output your final reply; \
summary holds only the single hardest conclusion from it.",
        json!({
            "type": "object",
            "properties": {
                "phase": {
                    "type": "string",
                    "enum": ["continue", "done", "await_approval", "paused", "failed"],
                    "default": "continue",
                    "description": "continue=continue to the next turn; done=task completed; await_approval=needs user confirmation; paused=waiting for a supplement; failed=cannot continue"
                },
                "summary": {"type": "string", "description": "What this turn did and how it turned out, in one sentence"},
                "artifacts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Artifact paths, relative to the workspace or absolute; may be repeated"
                },
                "delivery": {
                    "type": "string",
                    "enum": ["open_file", "open_folder", "notify_only"],
                    "description": "How to deliver when an artifact must be opened"
                }
            },
            "required": ["phase", "summary"]
        }),
    )
}

pub(crate) fn submit_verification_def() -> ToolDef {
    ToolDef::new(
        "submit_verification",
        "Submit the verification conclusion: call this exactly once after checking the artifacts, and do not output explanatory text.",
        json!({
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["pass", "fail"]},
                "summary": {"type": "string", "description": "One-sentence conclusion"},
                "checks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Check items and results, 2-5 entries"
                },
                "deliver": {"type": "string", "description": "Artifact path to open (relative to the workspace); leave empty if none"}
            },
            "required": ["verdict", "summary"]
        }),
    )
}

/// Execution turn: does the work, plus read-only checks and a structured finish.
pub(crate) fn executor_tools() -> Vec<ToolDef> {
    vec![
        bash_tool_def(),
        read_file_def(),
        list_files_def(),
        read_image_def(),
        search_def(),
        fetch_def(),
        crate::computer::accessibility_def(),
        crate::computer::input_def(),
        submit_result_def(),
    ]
}

/// Verification turn: read-only checks plus a structured submission; artifacts must not change.
pub(crate) fn verifier_tools() -> Vec<ToolDef> {
    // Verification is read-only: no bash here, so a "check" cannot turn into an "edit".
    vec![
        read_file_def(),
        list_files_def(),
        read_image_def(),
        submit_verification_def(),
    ]
}

pub(crate) fn function_output(call: &ToolCall, text: String) -> InputItem {
    InputItem::Raw(json!({
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": text,
    }))
}

pub(crate) fn describe_call(call: &ToolCall) -> String {
    let args = call.arguments.to_string();
    if args == "{}" || args.is_empty() {
        return format!("call {}", call.name);
    }
    format!("call {} {}", call.name, truncate(&args, 80))
}

pub(crate) fn first_line(text: &str) -> String {
    let line = text.lines().next().unwrap_or("").trim();
    truncate(line, 80)
}

pub(crate) fn truncate(value: &str, max: usize) -> String {
    if value.chars().count() <= max {
        return value.to_string();
    }
    let mut out: String = value.chars().take(max).collect();
    out.push('…');
    out
}

/// Read the first lines of a file (no authorized-folder check); errors go back as text too.
pub(crate) fn read_file_tool(_allowed: &[PathBuf], args: &Value) -> String {
    let raw = args
        .get("path")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim();
    if raw.is_empty() {
        return "read_file is missing path".into();
    }
    let max_lines = args
        .get("max_lines")
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .unwrap_or(60)
        .clamp(1, 400);
    let path = Path::new(raw);
    if let Ok(meta) = std::fs::metadata(path) {
        if meta.len() > 2 * 1024 * 1024 {
            return format!("file too large ({} bytes), skipping read", meta.len());
        }
    }
    let content = match std::fs::read_to_string(path) {
        Ok(content) => content,
        Err(err) => return format!("read failed: {raw} ({err})"),
    };
    let head: Vec<&str> = content.lines().take(max_lines).collect();
    let mut out = format!(
        "file: {} ({} lines, first {} read)\n",
        path.display(),
        content.lines().count(),
        head.len()
    );
    out.push_str(&head.join("\n"));
    truncate(&out, TOOL_OUTPUT_LIMIT)
}

/// List files in workspaces: one implementation covers three callers.
///
/// - `workspace` given as a name/id: list only the matching workspaces;
/// - not given but `current` is set: list the current workspace;
/// - neither given (the understanding turn has not settled on a target yet): list all.
pub(crate) fn list_files_tool(
    workspaces: &[Workspace],
    current: Option<&Path>,
    args: &Value,
) -> String {
    let needle = args
        .get("workspace")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let subdir = args
        .get("subdir")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    let limit = args
        .get("limit")
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .unwrap_or(40)
        .min(120);

    let selected: Vec<(String, PathBuf)> = match needle {
        Some(needle) => workspaces
            .iter()
            .filter(|workspace| {
                workspace.name.contains(needle) || workspace.id.to_string() == needle
            })
            .map(|workspace| (workspace.name.clone(), workspace.root_dir.clone()))
            .collect(),
        None => match current {
            Some(root) => vec![(
                root.file_name()
                    .map(|name| name.to_string_lossy().to_string())
                    .unwrap_or_else(|| "current workspace".into()),
                root.to_path_buf(),
            )],
            None => workspaces
                .iter()
                .map(|workspace| (workspace.name.clone(), workspace.root_dir.clone()))
                .collect(),
        },
    };

    if selected.is_empty() {
        return match needle {
            Some(needle) => format!("workspace not found: {needle}"),
            None => "No workspaces yet, and no materials to list.".into(),
        };
    }

    let mut out = String::new();
    for (name, root) in selected {
        let dir = if subdir.is_empty() {
            root
        } else if Path::new(subdir).is_absolute() {
            PathBuf::from(subdir)
        } else {
            root.join(subdir)
        };
        out.push_str(&format!("## {name} ({})\n", dir.display()));
        if !dir.exists() {
            out.push_str("(directory does not exist)\n");
            continue;
        }
        let mut entries = Vec::new();
        walk_dir(&dir, 2, limit, &mut entries);
        if entries.is_empty() {
            out.push_str("(empty directory)\n");
        } else {
            for entry in entries {
                out.push_str(&format!("- {entry}\n"));
            }
        }
    }
    truncate(&out, TOOL_OUTPUT_LIMIT)
}

fn walk_dir(dir: &Path, depth: usize, limit: usize, out: &mut Vec<String>) {
    if depth == 0 || out.len() >= limit {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<_> = entries.flatten().collect();
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        if out.len() >= limit {
            return;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') {
            continue;
        }
        let path = entry.path();
        let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
        out.push(format!(
            "{}{}",
            path.display(),
            if is_dir { "/" } else { "" }
        ));
        if is_dir {
            walk_dir(&path, depth - 1, limit, out);
        }
    }
}

/// Image recognition: downscale and upload, then let the vision model read the content (paths
/// get no authorized-folder check).
pub(crate) async fn read_image_tool(
    model: &dyn LanguageModel,
    vision_model: Option<&str>,
    _allowed: &[PathBuf],
    args: &Value,
) -> String {
    let raw_paths: Vec<String> = match args.get("path") {
        Some(Value::String(path)) => vec![path.clone()],
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect(),
        _ => Vec::new(),
    };
    let prompt = args
        .get("prompt")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    if prompt.is_empty() {
        return "read_image requires prompt: state what to extract from the image".into();
    }
    let max_edge = args
        .get("max_edge")
        .and_then(Value::as_u64)
        .map(|value| value as u32)
        .unwrap_or(vision::DEFAULT_MAX_EDGE);
    if let Err(err) = vision::validate_max_edge(max_edge) {
        return err;
    }

    let resolved: Vec<PathBuf> = raw_paths.iter().map(PathBuf::from).collect();
    if let Err(err) = vision::validate_paths(&resolved) {
        return err;
    }

    // Decoding and downscaling are CPU work; do not hold up the async thread.
    let prepared = tokio::task::spawn_blocking(move || {
        resolved
            .iter()
            .map(|path| vision::prepare(path, max_edge))
            .collect::<std::result::Result<Vec<_>, _>>()
    })
    .await;
    let prepared = match prepared {
        Ok(Ok(images)) => images,
        Ok(Err(err)) => return format!("image preparation failed: {err}"),
        Err(err) => return format!("image preparation failed: {err}"),
    };

    match vision::describe(model, &prepared, &prompt, max_edge, vision_model).await {
        Ok(text) => text,
        Err(err) => format!("image recognition failed: {err}"),
    }
}

/// Web search: validate the arguments, call the tools API and render a compact result list.
pub(crate) async fn search_tool(web: &WebTools, args: &Value) -> String {
    let query = args
        .get("query")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string();
    if query.is_empty() {
        return "search requires query: the terms to look up".into();
    }
    if query.chars().count() > 128 {
        return "search query must be at most 128 characters".into();
    }
    let count = args
        .get("count")
        .and_then(Value::as_u64)
        .map(|value| value.clamp(1, 50) as u32)
        .unwrap_or(10);
    let need_content = args
        .get("need_content")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let ori_question = args
        .get("ori_question")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| truncate(value, 128));

    let request = SearchRequest {
        query: query.clone(),
        count: Some(count),
        ori_question,
        need_content,
    };
    match web.search(&request).await {
        Ok(results) => format_search_results(&query, &results, need_content),
        Err(err) => format!("search failed: {err}"),
    }
}

fn format_search_results(query: &str, results: &[SearchResult], need_content: bool) -> String {
    if results.is_empty() {
        return format!("No results for \"{query}\".");
    }
    let mut out = format!("{} result(s) for \"{query}\":", results.len());
    for (index, result) in results.iter().enumerate() {
        let title = if result.title.trim().is_empty() {
            "(untitled)"
        } else {
            result.title.trim()
        };
        out.push_str(&format!("\n{}. {}", index + 1, title));
        out.push_str(&format!("\n   URL: {}", result.url));
        let mut meta = Vec::new();
        if let Some(site) = result
            .site_name
            .as_deref()
            .filter(|site| !site.trim().is_empty())
        {
            meta.push(site.trim().to_string());
        }
        meta.push(format!("authority {}", result.authority));
        meta.push(format!("score {:.1}", result.rerank_score));
        if let Some(date) = result
            .date
            .as_deref()
            .filter(|date| !date.trim().is_empty())
        {
            meta.push(truncate(date.trim(), 10));
        }
        out.push_str(&format!("\n   {}", meta.join(" · ")));
        if !result.snippet.trim().is_empty() {
            out.push_str(&format!(
                "\n   Snippet: {}",
                truncate(result.snippet.trim(), 200)
            ));
        }
        if !result.summary.trim().is_empty() {
            out.push_str(&format!(
                "\n   Summary: {}",
                truncate(result.summary.trim(), 300)
            ));
        }
        if need_content && !result.content.trim().is_empty() {
            out.push_str(&format!(
                "\n   Content: {}",
                truncate(result.content.trim(), 500)
            ));
        }
    }
    truncate(&out, TOOL_OUTPUT_LIMIT)
}

/// Page fetch: returns the title plus Markdown body, capped so one page cannot flood the context.
pub(crate) async fn fetch_tool(web: &WebTools, args: &Value) -> String {
    let url = args
        .get("url")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string();
    if url.is_empty() {
        return "fetch requires url: the page to read".into();
    }
    if url.chars().count() > 8192 {
        return "fetch url must be at most 8192 characters".into();
    }
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return "fetch url must be an absolute http(s) URL".into();
    }
    match web.fetch(&url).await {
        Ok(page) => {
            let title = if page.title.trim().is_empty() {
                "(untitled)"
            } else {
                page.title.trim()
            };
            if page.content.trim().is_empty() {
                return format!("Fetched: {title}\nURL: {url}\n(no readable content)");
            }
            truncate(
                &format!(
                    "Fetched: {title}\nURL: {url}\n--- content ---\n{}",
                    page.content.trim()
                ),
                TOOL_OUTPUT_LIMIT,
            )
        }
        Err(err) => format!("fetch failed: {err}"),
    }
}
