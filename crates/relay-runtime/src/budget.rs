//! Context budget: estimate request size and, once it exceeds the window threshold (90% by
//! default), have the model compress the history into a summary.
//!
//! Two goals must hold at the same time:
//! - **Compaction**: no deterministic trimming; the model summarizes the history into notes the
//!   task can continue from;
//! - **Caching**: model calls are stateless replays and the server hits its token cache by
//!   prefix. So compaction keeps the first messages verbatim and appends the summary after
//!   them: the more stable the prefix, the higher the hit rate.

use relay_openai::{CompletionRequest, InputItem, LanguageModel, ModelError, ToolDef};

/// Summary prompt, kept under `prompts/` like the other prompts.
const SUMMARY_PROMPT: &str = include_str!("../../../prompts/compact.md");
/// Heading of the summary block: becomes the second history item after compaction, and also
/// marks that compaction already happened.
pub const SUMMARY_HEADING: &str = "## History summary (context compaction)";
/// Cap per rendered item when building the summarizer input: affects only the summary request
/// size, never the main context.
const RENDER_OUTPUT_LIMIT: usize = 4_000;
/// Total length cap of the summary request (characters): for extremely long histories, keep
/// head and tail.
const RENDER_TOTAL_LIMIT: usize = 200_000;
/// Rough token estimate for one 1024px-class screenshot.
const IMAGE_TOKENS: u32 = 1_200;
/// Structural overhead per message (role, separators, and so on).
const ITEM_OVERHEAD: u32 = 4;

/// Window budget: the window size plus the ratio that triggers compaction.
#[derive(Debug, Clone, Copy)]
pub struct Budget {
    window: u32,
    compact_ratio: f32,
}

impl Budget {
    pub fn new(window: u32, compact_ratio: f32) -> Self {
        Self {
            window,
            compact_ratio: compact_ratio.clamp(0.1, 1.0),
        }
    }

    /// With an unknown window (0) no budget decisions are made: behavior matches the code from
    /// before this feature existed.
    pub fn enabled(&self) -> bool {
        self.window > 0
    }

    /// Whether to compact: fires only when **strictly above** the threshold (default `> 90%`).
    pub fn should_compact(&self, tokens: u32) -> bool {
        if !self.enabled() {
            return false;
        }
        // Round to an integer threshold before comparing, avoiding float jitter at edges like
        // 0.9 * 256_000.
        let threshold = (self.window as f32 * self.compact_ratio).round() as u32;
        tokens > threshold
    }

    /// Reserve output space proportionally for small windows; tighten again to whatever space is
    /// left at request time.
    pub fn output_limit(&self, requested: u32, input: u32) -> u32 {
        if !self.enabled() {
            return requested;
        }
        requested
            .min((self.window / 4).max(1))
            .min(self.window.saturating_sub(input).max(1))
    }

    pub(crate) fn checked_output_limit(
        &self,
        requested: u32,
        input: u32,
    ) -> Result<u32, ModelError> {
        if self.enabled() && input >= self.window {
            return Err(ModelError::Other(format!("After reserving the task instructions and tool definitions, the input is estimated at {input} tokens, which already reaches the configured window {}. Shorten the task input or check the configured model window.", self.window)));
        }
        Ok(self.output_limit(requested, input))
    }

    pub fn needs_summary(&self, input: u32, requested: u32) -> bool {
        self.should_compact(input.saturating_add(self.output_limit(requested, 0)))
    }

    /// Understanding and verification share this budget. Only rejected model requests are retried;
    /// tool results stay replayable.
    pub(crate) async fn complete(
        &self,
        model: &dyn LanguageModel,
        mut request: CompletionRequest,
        items: &mut Vec<InputItem>,
        usage: &mut relay_core::ContextUsage,
    ) -> Result<relay_openai::ModelOutput, ModelError> {
        let overhead = request_overhead(&request.instructions, &request.tools);
        let mut retry = false;
        loop {
            let input = overhead + estimate(items);
            if (retry || self.needs_summary(input, request.max_output_tokens))
                && items.iter().any(|item| matches!(item, InputItem::Raw(_)))
            {
                match summarize_for_window(model, &items[1..], *self).await {
                    Ok(summary) => *items = compacted_items(items, summary.record(usage), 1),
                    Err(err) if retry => return Err(err),
                    Err(err) => tracing::warn!(%err, "history compaction failed"),
                }
            }
            request.max_output_tokens =
                self.checked_output_limit(request.max_output_tokens, overhead + estimate(items))?;
            let mut attempt = request.clone();
            attempt.input.clone_from(items);
            match model.complete(attempt).await {
                Err(err) if !retry && is_context_overflow(&err.to_string()) => {
                    request.max_output_tokens = (request.max_output_tokens / 2).clamp(1, 1024);
                    retry = true;
                }
                result => return result,
            }
        }
    }

    pub fn window(&self) -> u32 {
        self.window
    }
}

/// Tell apart overflow errors that a smaller request can recover from.
pub(crate) fn is_context_overflow(error: &str) -> bool {
    let lower = error.to_lowercase();
    [
        "context length",
        "context_length",
        "maximum context",
        "too many tokens",
        "context window",
        "prompt is too long",
        "reduce the length",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

/// Estimate the token count of a set of inputs.
pub fn estimate(items: &[InputItem]) -> u32 {
    items.iter().map(item_tokens).sum()
}

/// Instructions and tool definitions are resent on every request; compute them once so the loop
/// can reuse them.
pub fn request_overhead(instructions: &str, tools: &[ToolDef]) -> u32 {
    let definitions: Vec<_> = tools.iter().map(|t| serde_json::json!({
        "type": "function", "name": t.name, "description": t.description, "parameters": t.parameters,
    })).collect();
    text_tokens(instructions)
        + text_tokens(&serde_json::Value::Array(definitions).to_string())
        + ITEM_OVERHEAD
}

pub fn text_tokens(text: &str) -> u32 {
    let mut cjk = 0u32;
    let mut other = 0u32;
    for ch in text.chars() {
        if is_cjk(ch) {
            cjk += 1;
        } else {
            other += 1;
        }
    }
    // Chinese runs near 1 token per character; English and code run about 4 characters per
    // token.
    cjk + other.saturating_add(3) / 4
}

fn is_cjk(ch: char) -> bool {
    matches!(
        ch as u32,
        0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF | 0x3040..=0x30FF
    )
}

fn item_tokens(item: &InputItem) -> u32 {
    match item {
        InputItem::Text(text) => text_tokens(text).saturating_add(ITEM_OVERHEAD),
        // Replayed reasoning / function_call items are tiny, so the serialized length is a good
        // enough estimate.
        InputItem::Raw(value) => text_tokens(&value.to_string()).saturating_add(ITEM_OVERHEAD),
        InputItem::Image { .. } | InputItem::ImageBytes { .. } => IMAGE_TOKENS,
    }
}

/// Compaction result: the summary text plus the usage of this compaction call itself (which
/// counts toward task stats).
#[derive(Debug, Clone)]
pub struct Compacted {
    pub summary: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cached_tokens: u32,
}

impl Compacted {
    pub(crate) fn record(self, usage: &mut relay_core::ContextUsage) -> String {
        usage.record(self.input_tokens, self.output_tokens, self.cached_tokens);
        usage.compactions += 1;
        self.summary
    }
}

/// Have the model compress the history into a summary: sends only the summary prompt plus the
/// rendered history, with no tools.
pub async fn summarize(
    model: &dyn LanguageModel,
    items: &[InputItem],
) -> Result<Compacted, ModelError> {
    summarize_for_window(model, items, Budget::new(0, 0.9)).await
}

pub(crate) async fn summarize_for_window(
    model: &dyn LanguageModel,
    items: &[InputItem],
    budget: Budget,
) -> Result<Compacted, ModelError> {
    let mut history = render_history(items);
    let overhead = request_overhead(SUMMARY_PROMPT, &[]) + ITEM_OVERHEAD;
    let available = budget
        .window()
        .saturating_sub(overhead + budget.output_limit(2048, 0));
    if budget.enabled() && text_tokens(&history) > available {
        // Assume at most one token per character. Keep head and tail, and leave room for the
        // omission marker.
        history = clip_total(history, available.saturating_sub(40) as usize);
    }
    let input_tokens = overhead + text_tokens(&history);
    let mut request = CompletionRequest::text(SUMMARY_PROMPT, history);
    request.max_output_tokens = budget.checked_output_limit(2048, input_tokens)?;
    let output = model.complete(request).await?;
    let summary = output.text.trim().to_string();
    if summary.is_empty() {
        return Err(ModelError::Other("the summary returned no content".into()));
    }
    let (input_tokens, output_tokens, cached_tokens) = output
        .usage
        .as_ref()
        .map(|cost| (cost.input_tokens, cost.output_tokens, cost.cached_tokens))
        .unwrap_or((0, 0, 0));
    Ok(Compacted {
        summary,
        input_tokens,
        output_tokens,
        cached_tokens,
    })
}

/// Compacted input: the first `keep_prefix` items are **kept verbatim** (conversation history
/// plus task instructions, preserving the prefix cache), and the summary follows right after
/// them in place of the tool transcript.
///
/// The caller supplies `keep_prefix` in assembly order: only prefix segments that really stay
/// byte-identical are worth keeping, because retaining changed segments just burns budget
/// without improving cache hits.
pub fn compacted_items(items: &[InputItem], summary: String, keep_prefix: usize) -> Vec<InputItem> {
    let keep = keep_prefix.min(items.len());
    let mut out: Vec<InputItem> = items[..keep].to_vec();
    // Original images may be the only evidence of what the user meant, and the summarizer
    // only accepts text.
    out.extend(
        items[keep..]
            .iter()
            .filter(|item| matches!(item, InputItem::Image { .. } | InputItem::ImageBytes { .. }))
            .cloned(),
    );
    out.push(InputItem::Text(format!("{SUMMARY_HEADING}\n{summary}")));
    out
}

/// Render history for the summarizer: only what was said, which tool was called, and what came
/// back.
///
/// reasoning is left out (it restates current reasoning, adds little once compacted, and would
/// eat into the summary budget); a screenshot leaves only a one-line marker, and the caller
/// keeps the original.
pub fn render_history(items: &[InputItem]) -> String {
    let mut out = String::new();
    for item in items {
        match item {
            InputItem::Text(text) => {
                out.push_str("[text]\n");
                push_clipped(&mut out, text.trim(), RENDER_OUTPUT_LIMIT);
                out.push('\n');
            }
            InputItem::Raw(value) => match value.get("type").and_then(|v| v.as_str()) {
                Some("function_call") => {
                    let name = value.get("name").and_then(|v| v.as_str()).unwrap_or("tool");
                    let args = value
                        .get("arguments")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    out.push_str(&format!("[call {}] {}\n", name, clip(args, 800)));
                }
                Some("function_call_output") => {
                    let text = value.get("output").and_then(|v| v.as_str()).unwrap_or("");
                    out.push_str("[result] ");
                    push_clipped(&mut out, text.trim(), RENDER_OUTPUT_LIMIT);
                    out.push('\n');
                }
                _ if value.get("role").is_some() => {
                    out.push_str(value["role"].as_str().unwrap_or("message"));
                    out.push_str(": ");
                    if let Some(content) = value["content"].as_str() {
                        push_clipped(&mut out, content, RENDER_OUTPUT_LIMIT);
                    } else if let Some(parts) = value["content"].as_array() {
                        for part in parts {
                            if let Some(text) = part["text"].as_str() {
                                push_clipped(&mut out, text, RENDER_OUTPUT_LIMIT);
                                out.push('\n');
                            }
                        }
                    }
                    out.push('\n');
                }
                _ => {}
            },
            InputItem::Image { .. } | InputItem::ImageBytes { .. } => {
                out.push_str("[one screenshot, original kept separately]\n");
            }
        }
    }
    clip_total(out, RENDER_TOTAL_LIMIT)
}

fn push_clipped(out: &mut String, text: &str, limit: usize) {
    out.push_str(&clip(text, limit));
}

fn clip(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let mut clipped: String = text.chars().take(limit).collect();
    clipped.push('…');
    clipped
}

/// Over the total limit, keep head and tail: the task statement sits at the start and the
/// latest results at the end.
fn clip_total(text: String, limit: usize) -> String {
    let count = text.chars().count();
    if count <= limit {
        return text;
    }
    let head: String = text.chars().take(limit * 2 / 3).collect();
    let tail: String = text.chars().skip(count - limit / 3).collect();
    format!("{head}\n…(middle omitted, {count} characters total)…\n{tail}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use relay_openai::{MockModel, MockStep};
    use serde_json::json;

    fn output(call_id: &str, text: &str) -> InputItem {
        InputItem::Raw(json!({
            "type": "function_call_output",
            "call_id": call_id,
            "output": text,
        }))
    }

    fn call(call_id: &str, command: &str) -> InputItem {
        InputItem::Raw(json!({
            "type": "function_call",
            "call_id": call_id,
            "name": "bash",
            "arguments": json!({ "command": command }).to_string(),
        }))
    }

    #[test]
    fn compaction_keeps_untranscribed_screenshots_and_reserves_output() {
        let items = vec![
            InputItem::Text("act on the current screen".into()),
            InputItem::ImageBytes {
                mime: "image/png".into(),
                bytes: vec![17, 23],
            },
            output("c1", "tool result"),
        ];
        let compacted = compacted_items(&items, "materials read".into(), 1);
        assert!(matches!(&compacted[1], InputItem::ImageBytes { bytes, .. } if bytes == &[17, 23]));
        let budget = Budget::new(4000, 0.9);
        assert_eq!(budget.output_limit(8192, 3100), 900);
        assert!(budget.needs_summary(3100, 8192));
        let tool = ToolDef::new("read", "read the materials", json!({"type":"object"}));
        assert!(request_overhead("execute", &[tool]) > request_overhead("execute", &[]));
    }

    #[test]
    fn estimates_cjk_denser_than_ascii() {
        // U+5B57 is a representative CJK ideograph: one token per character.
        let cjk = text_tokens(&"\u{5b57}".repeat(400));
        let ascii = text_tokens(&"a".repeat(400));
        assert_eq!(cjk, 400);
        assert_eq!(ascii, 100);
    }

    #[test]
    fn triggers_strictly_above_ninety_percent() {
        let budget = Budget::new(100_000, 0.9);
        assert!(!budget.should_compact(90_000));
        assert!(budget.should_compact(90_001));
        assert!(!Budget::new(0, 0.9).should_compact(999_999));
    }

    #[test]
    fn renders_calls_and_results_but_not_reasoning() {
        let items = vec![
            InputItem::Text("## Task\ncheck the data".into()),
            InputItem::Raw(json!({ "type": "reasoning", "summary": [] })),
            call("c1", "ls data"),
            output("c1", "a.csv\nb.csv"),
        ];
        let rendered = render_history(&items);
        assert!(rendered.contains("[text]"));
        assert!(rendered.contains("check the data"));
        assert!(rendered.contains("[call bash]"));
        assert!(rendered.contains("ls data"));
        assert!(rendered.contains("a.csv"));
        assert!(!rendered.contains("reasoning"));
    }

    #[test]
    fn compaction_keeps_head_verbatim_for_prefix_cache() {
        let head = "## Task\ncheck the data\n## Confirmed plan\n- read the file";
        let items = vec![
            InputItem::Text(head.into()),
            call("c1", "ls"),
            output("c1", "a.csv"),
        ];
        let compacted = compacted_items(&items, "## Goal\ncheck the data".into(), 1);
        assert_eq!(compacted.len(), 2);
        let InputItem::Text(first) = &compacted[0] else {
            panic!("the first item should be the original text item");
        };
        // Not one byte of the prefix may change, or the whole cached segment is lost.
        assert_eq!(first, head);
        let InputItem::Text(second) = &compacted[1] else {
            panic!("the second item should be the summary");
        };
        assert!(second.starts_with(SUMMARY_HEADING));
        assert!(second.contains("check the data"));
    }

    #[tokio::test]
    async fn summarize_returns_text_and_usage() {
        let model = MockModel::scripted([MockStep::Text("## Goal\ncheck the data".into())]);
        let items = vec![
            InputItem::Text("## Task\ncheck the data".into()),
            call("c1", "ls"),
            output("c1", "a.csv"),
        ];
        let compacted = summarize(&model, &items).await.unwrap();
        // MockModel returns a fixed structure for summary requests (aligned with
        // prompts/compact.md).
        assert!(compacted.summary.starts_with("## Goal"));
        assert!(compacted.input_tokens > 0);
        assert!(compacted.output_tokens > 0);
        assert!(compacted.cached_tokens > 0);
    }
}
