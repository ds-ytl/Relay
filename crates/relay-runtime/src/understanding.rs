//! Understanding agent: first get a clear picture of the environment and history across several
//! rounds of tool probing, then submit the combined "workspace / session / plan" decision in a
//! single `submit_plan` call.
//!
//! There is deliberately **no** standalone string-matching router here: which workspace to use,
//! whether to continue an existing session and whether to create one are all judgments the model
//! makes after seeing real information. The routing result is a by-product of understanding.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::Utc;
use relay_core::{ContextUsage, DeliveryMode, EnvironmentSnapshot, Plan, Workspace, WorkspaceId};
use relay_openai::{CompletionRequest, InputItem, LanguageModel, ToolCall, ToolDef, WebTools};
use relay_storage::Store;
use serde_json::{json, Value};

use crate::agents::UNDERSTAND_PROMPT;
use crate::budget::Budget;
use crate::computer;
use crate::error::{Result, RuntimeError};
use crate::tools::{self, ToolOutcome, TOOL_OUTPUT_LIMIT};
use crate::vision;

/// Screenshot recognition prompt: report structure and text first, then checkable details; no
/// speculation allowed.
const SCREENSHOT_PROMPT: &str = "This is a desktop screenshot. Describe what is on it objectively, so the user's intent can be inferred next:\n\
1. which app and which page it is (title bar / tab / header);\n\
2. the text, numbers, buttons and list items visible on the page, listed item by item per region, copied verbatim;\n\
3. if it is a chat, table or form, give each side's messages or each field and value one by one;\n\
4. write \"illegible\" for anything you cannot read clearly; do not guess and do not add anything.";

/// Long edge used for screenshot recognition: UI text is often small, so this sees finer detail
/// than the read_image default of 1024.
const SCREENSHOT_MAX_EDGE: u32 = 1568;

fn describe_capture(app: &str, title: &str, path: &Path) -> String {
    format!(
        "Captured {app} · {}: {}",
        if title.trim().is_empty() {
            "(untitled window)"
        } else {
            title.trim()
        },
        path.display()
    )
}

/// Placeholder text passed to the understanding agent in place of an instruction when the user
/// entered no text and only pressed "Understand".
///
/// It doubles as the marker for the blank-instruction scenario on the understanding side: on
/// seeing this text, the agent knows it has to infer the goal from what is on screen right now
/// instead of waiting for the user to add a sentence.
pub const BLANK_INSTRUCTION: &str =
    "(The user entered no text: look at the app and content I am currently using and infer what I want to do)";

/// Blank-instruction test: empty, or exactly the placeholder text.
pub fn is_blank_instruction(text: &str) -> bool {
    let trimmed = text.trim();
    trimmed.is_empty() || trimmed == BLANK_INSTRUCTION
}

/// Normalize user input into an instruction that can be passed downstream: empty means the
/// blank-instruction scenario.
pub fn normalize_instruction(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        BLANK_INSTRUCTION.to_string()
    } else {
        trimmed.to_string()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnderstandStage {
    Started,
    /// Waiting for the model's judgment (a model call usually takes several seconds, and the UI
    /// uses this event to show it is not stuck).
    Thinking,
    ToolCall,
    ToolResult,
    Decided,
    Fallback,
    /// Understanding failed (model / network and the like); no task has been created yet.
    Failed,
}

#[derive(Debug, Clone)]
pub struct UnderstandEvent {
    pub step: u32,
    pub stage: UnderstandStage,
    pub tool: Option<String>,
    pub summary: String,
}

/// Current screen captured before understanding starts: the text is either the screen content
/// (when a vision model is configured) or a one-line note, and `image` is the raw screenshot
/// (when the main model looks at the image itself).
#[derive(Debug, Clone)]
pub struct PrefetchedScene {
    pub text: String,
    pub image: Option<PathBuf>,
    /// True when the user attached the screenshot themselves (region chosen and annotated).
    pub attached: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkspaceChoice {
    Reuse(String),
    Create { name: String, keywords: Vec<String> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionChoice {
    Reuse(String),
    Create { title: String },
}

#[derive(Debug, Clone)]
pub struct Understanding {
    pub workspace: WorkspaceChoice,
    pub session: SessionChoice,
    pub plan: Plan,
    pub confidence: f32,
    pub reason: String,
    /// Human-readable probing trail, e.g. `take_screenshot -> Captured 2560x1600`.
    pub trail: Vec<String>,
    /// Model usage during understanding: the task does not exist yet, so it is held here and
    /// folded into the stats once the task is created.
    pub usage: ContextUsage,
}

pub struct UnderstandingInput<'a> {
    pub store: &'a Store,
    pub context: &'a EnvironmentSnapshot,
    pub instruction: &'a str,
    pub supplement: Option<&'a str>,
    pub pinned_workspace: Option<WorkspaceId>,
    /// Authorized folders shown in the context (tools no longer enforce them).
    pub allowed_dirs: Vec<PathBuf>,
    /// Vision model; the main model is used when this is empty.
    pub vision_model: Option<String>,
    pub screenshot_dir: PathBuf,
    pub max_steps: u32,
    /// Context budget: before each step's call it decides, from the estimate, whether earlier
    /// tool output gets compacted.
    pub budget: Budget,
    /// Current screen captured before the first round: having it saves the whole model round
    /// trip of "ask for a screenshot first".
    pub prefetched: Option<PrefetchedScene>,
}

pub struct UnderstandingAgent {
    model: Arc<dyn LanguageModel>,
    /// Web search / fetch; absent without an API key.
    web_tools: Option<Arc<WebTools>>,
}

impl UnderstandingAgent {
    pub fn new(model: Arc<dyn LanguageModel>, web_tools: Option<Arc<WebTools>>) -> Self {
        Self { model, web_tools }
    }

    /// Run the whole probe-then-decide loop. `notify` streams progress to the UI in real time.
    pub async fn run(
        &self,
        input: &UnderstandingInput<'_>,
        notify: &(dyn Fn(UnderstandEvent) + Send + Sync),
    ) -> Result<Understanding> {
        notify(UnderstandEvent {
            step: 0,
            stage: UnderstandStage::Started,
            tool: None,
            summary: "started understanding the task".into(),
        });

        let mut tools = tool_defs();
        // Without an API key there is no tools API: do not advertise calls that can only fail.
        if self.web_tools.is_none() {
            tools.retain(|tool| tool.name != "search" && tool.name != "fetch");
        }
        let tool_names: Vec<String> = tools.iter().map(|tool| tool.name.clone()).collect();
        let mut items = vec![InputItem::Text(briefing(input))];
        if let Some(path) = input
            .prefetched
            .as_ref()
            .and_then(|scene| scene.image.clone())
        {
            items.push(InputItem::Image {
                path,
                mime: "image/png".into(),
            });
        }
        let mut trail: Vec<String> = Vec::new();
        let mut usage = ContextUsage::default();
        let blank = is_blank_instruction(input.instruction);
        // With a blank instruction the current screen is the primary evidence: submission is
        // refused until the screen has been seen, with at most one nudge.
        // A prefetched screenshot counts as having seen the screen: submission may go ahead.
        let mut scene_checked = !blank || input.prefetched.is_some();
        let mut nudged = false;

        for step in 1..=input.max_steps.max(1) {
            let request = CompletionRequest {
                instructions: UNDERSTAND_PROMPT.to_string(),
                input: Vec::new(),
                model: None,
                temperature: Some(0.2),
                // Reasoning models spend budget on thinking; leave ample headroom so the reply
                // is not truncated.
                max_output_tokens: 8192,
                previous_response_id: None,
                tools: tools.clone(),
            };
            notify(UnderstandEvent {
                step,
                stage: UnderstandStage::Thinking,
                tool: None,
                summary: format!("step {step}: the model is deciding"),
            });
            let call_started = std::time::Instant::now();
            let output = input
                .budget
                .complete(self.model.as_ref(), request, &mut items, &mut usage)
                .await?;
            tracing::info!(
                step,
                elapsed_ms = call_started.elapsed().as_millis() as u64,
                "understanding round model call finished"
            );
            if let Some(cost) = output.usage.as_ref() {
                usage.record(cost.input_tokens, cost.output_tokens, cost.cached_tokens);
            }

            // The server requires reasoning and function_call items to be replayed verbatim.
            for raw in &output.replay_items {
                items.push(InputItem::Raw(raw.clone()));
            }

            if let Some(decision) = take_decision(&output.tool_calls) {
                if blank && !scene_checked && !nudged {
                    nudged = true;
                    notify(UnderstandEvent {
                        step,
                        stage: UnderstandStage::ToolResult,
                        tool: Some("submit_plan".into()),
                        summary: "blank instruction: look at the current screen first, submission not accepted yet".into(),
                    });
                    // The submission needs a function_call_output reply too, to keep the
                    // protocol complete.
                    let refusal = "Refused: with a blank instruction you must look at the current screen first. The environment snapshot is already provided, so call take_screenshot, and only then submit_plan; historical workspaces/sessions only decide where work lands and cannot be used to infer the goal.";
                    match output
                        .tool_calls
                        .iter()
                        .find(|call| call.name == "submit_plan")
                    {
                        Some(call) => items.push(tools::function_output(call, refusal.into())),
                        None => items.push(InputItem::Text(refusal.into())),
                    }
                    continue;
                }
                notify(UnderstandEvent {
                    step,
                    stage: UnderstandStage::Decided,
                    tool: Some("submit_plan".into()),
                    summary: decision.reason.clone(),
                });
                return Ok(Understanding {
                    workspace: decision.workspace,
                    session: decision.session,
                    plan: decision.plan,
                    confidence: decision.confidence,
                    reason: decision.reason,
                    trail,
                    usage,
                });
            }

            if output.tool_calls.is_empty() {
                // The model called no tool and did not submit: fall back to its text so the task
                // can still move on.
                let understanding = fallback_understanding(input, &output.text);
                notify(UnderstandEvent {
                    step,
                    stage: UnderstandStage::Fallback,
                    tool: None,
                    summary: "the model did not call submit_plan, fell back to its text".into(),
                });
                return Ok(Understanding {
                    trail,
                    usage,
                    ..understanding
                });
            }

            for call in &output.tool_calls {
                if !tool_names.iter().any(|name| name == &call.name) {
                    let text = format!("unknown tool: {}", call.name);
                    items.push(tools::function_output(call, text.clone()));
                    continue;
                }
                if matches!(
                    call.name.as_str(),
                    "get_environment" | "take_screenshot" | "computer_accessibility"
                ) {
                    // A failed screenshot still counts as an attempt: a permissions problem
                    // must not leave the user stuck in understanding.
                    scene_checked = true;
                }
                notify(UnderstandEvent {
                    step,
                    stage: UnderstandStage::ToolCall,
                    tool: Some(call.name.clone()),
                    summary: tools::describe_call(call),
                });
                let tool_started = std::time::Instant::now();
                let outcome = self.call_tool(call, input).await?;
                tracing::info!(
                    step,
                    tool = %call.name,
                    elapsed_ms = tool_started.elapsed().as_millis() as u64,
                    "understanding tool call finished"
                );
                trail.push(format!(
                    "{} → {}",
                    call.name,
                    tools::first_line(&outcome.text)
                ));
                notify(UnderstandEvent {
                    step,
                    stage: UnderstandStage::ToolResult,
                    tool: Some(call.name.clone()),
                    summary: tools::first_line(&outcome.text),
                });
                items.push(tools::function_output(call, outcome.text));
                if let Some(path) = outcome.image {
                    items.push(InputItem::Image {
                        path,
                        mime: "image/png".into(),
                    });
                }
            }
        }

        Err(RuntimeError::Invalid(format!(
            "understanding ran {} steps without submitting a plan",
            input.max_steps
        )))
    }

    async fn call_tool(
        &self,
        call: &ToolCall,
        input: &UnderstandingInput<'_>,
    ) -> Result<ToolOutcome> {
        let args = &call.arguments;
        let tool = call.name.as_str();
        let text = match tool {
            "get_environment" => environment_text(input),
            "read_clipboard" => match relay_platform::clipboard_text() {
                Some(text) if !text.trim().is_empty() => {
                    format!(
                        "clipboard ({} characters):\n{}",
                        text.chars().count(),
                        clip(&text)
                    )
                }
                _ => "clipboard is empty or unreadable".to_string(),
            },
            "list_workspaces" => list_workspaces(input)?,
            "list_sessions" => list_sessions(input, args)?,
            "list_tasks" => list_tasks(input, args)?,
            "computer_accessibility" => {
                // Understanding is read-only: semantic queries and screenshots are available
                // in understanding rounds, while clicks wait for the execution round after
                // confirmation.
                if args.get("op").and_then(Value::as_str) == Some("action") {
                    "Understanding is read-only: put op=action in the plan for the execution round; use query / inspect / tree / screenshot / wait here.".to_string()
                } else {
                    computer::invoke(tool, args).await
                }
            }
            "list_files" => list_files(input, args)?,
            "read_file" => read_file(input, args)?,
            "take_screenshot" => {
                let scope = args
                    .get("scope")
                    .and_then(Value::as_str)
                    .unwrap_or("window");
                // The frontmost window was already captured in the first round: never capture
                // or recognize the same frame twice.
                if scope == "window" {
                    if let Some(scene) = &input.prefetched {
                        return Ok(ToolOutcome {
                            text: scene.text.clone(),
                            image: None,
                        });
                    }
                }
                let app = args
                    .get("app")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty());
                let dir = input.screenshot_dir.as_path();
                let shot: std::result::Result<(String, Vec<PathBuf>), relay_platform::ScreenError> =
                    match scope {
                        "app" => match app {
                            None => {
                                return Ok(ToolOutcome {
                                    text: "scope=app requires the app argument, e.g. {\"scope\":\"app\",\"app\":\"WeChat\"}"
                                        .into(),
                                    image: None,
                                })
                            }
                            Some(name) => relay_platform::capture_app_png(name, dir).map(|shot| {
                                (
                                    describe_capture(&shot.app, &shot.title, &shot.path),
                                    vec![shot.path],
                                )
                            }),
                        },
                        _ => relay_platform::capture_user_window_png(dir).map(|shot| {
                            let text = if shot.app.trim().is_empty() {
                                format!(
                                    "could not identify a specific window, captured the main display instead: {}",
                                    shot.path.display()
                                )
                            } else {
                                describe_capture(&shot.app, &shot.title, &shot.path)
                            };
                            (text, vec![shot.path])
                        }),
                    };
                let (header, paths) = match shot {
                    Ok(value) => value,
                    Err(err) => {
                        return Ok(ToolOutcome {
                            text: format!(
                                "screenshot failed: {err} (on macOS, grant Relay access in System Settings -> Privacy & Security -> Screen Recording)"
                            ),
                            image: None,
                        })
                    }
                };

                // Screenshots are always turned into text first: use the dedicated vision model
                // when one is configured, otherwise ask the main model directly. That keeps the
                // tool result as screen content instead of a file name, and the execution side
                // can reuse it.
                let prepared = tokio::task::spawn_blocking({
                    let paths = paths.clone();
                    move || {
                        paths
                            .iter()
                            .take(vision::MAX_IMAGES)
                            .map(|path| vision::prepare(path, SCREENSHOT_MAX_EDGE))
                            .collect::<std::result::Result<Vec<_>, _>>()
                    }
                })
                .await
                .unwrap_or_else(|err| Err(format!("screenshot preprocessing task failed: {err}")));
                let prepared = match prepared {
                    Ok(images) => images,
                    Err(err) => {
                        return Ok(ToolOutcome {
                            text: format!("{header}\nscreen recognition failed: {err}"),
                            image: None,
                        })
                    }
                };
                return match vision::describe(
                    self.model.as_ref(),
                    &prepared,
                    SCREENSHOT_PROMPT,
                    SCREENSHOT_MAX_EDGE,
                    input.vision_model.as_deref(),
                )
                .await
                {
                    Ok(description) => Ok(ToolOutcome {
                        text: format!("{header}\n--- screen content ---\n{description}"),
                        image: None,
                    }),
                    Err(err) => Ok(ToolOutcome {
                        text: format!(
                            "{header}\nscreen recognition failed: {err}\n(the main model may not support image input; set vision_model in the configuration)"
                        ),
                        image: None,
                    }),
                };
            }
            "search" => match &self.web_tools {
                Some(web) => tools::search_tool(web, args).await,
                None => "search is unavailable: configure an API key whose endpoint serves /tools/search".into(),
            },
            "fetch" => match &self.web_tools {
                Some(web) => tools::fetch_tool(web, args).await,
                None => "fetch is unavailable: configure an API key whose endpoint serves /tools/fetch".into(),
            },
            "read_image" => {
                tools::read_image_tool(
                    self.model.as_ref(),
                    input.vision_model.as_deref(),
                    &input.allowed_dirs,
                    args,
                )
                .await
            }
            other => format!("unknown tool: {other}"),
        };
        Ok(ToolOutcome {
            text: tools::truncate(&text, TOOL_OUTPUT_LIMIT),
            image: None,
        })
    }
}

const BLANK_SCENE: &str = r#"## User instruction
(Empty: the user entered no text and only pressed "Understand")

## Blank instruction scenario
The user wrote nothing: the goal must be inferred from "what they are using right now", never from history.
The evidence has a strict priority, highest first:

1. **Current screen (highest priority, decides "what to do")**: frontmost app, window title, screen content, selected files.
   The environment snapshot is already provided with the instruction; if the screen was captured automatically, judge from it directly, otherwise use `take_screenshot` to look first.
2. **Content of the current object (extra detail)**: `read_clipboard` for what was just copied, plus the file being edited or viewed when needed.
3. **History (decides only "where it goes / whether to continue a session", never the goal)**: `list_workspaces` / `list_sessions` / `list_tasks`.

Hard rules:
- Order: screen -> content -> history (history only fixes the target). **Do not consult history before you have seen the screen, and above all do not `submit_plan`.**
- `take_screenshot` and `read_clipboard` can be called in the same round; do not run a whole extra round just to wait for one result.
- When the screen and history conflict, the screen always wins. If the screen shows an error log, you must not keep working on the electricity-bill ledger just because the latest task was one.
- With a blank instruction, call each history tool at most once; submit as soon as the screen gives enough, and do not dig through history repeatedly to "find a project".
- When the screen contains images, or the user has selected image files (invoices, screenshots, scans), use `read_image` to read their content (with `prompt` stating what to extract); do not guess from the file name.
- Pick from the screen "the one thing they will inevitably do next" and write it as a single actionable goal, for example:
  - looking at a bill, order or sign-up message in chat/mail -> turn it into a ledger or a to-do file;
  - looking at a pile of screenshots or scattered notes -> merge them into one list;
  - looking at an error or a log -> find the cause and write down a fix;
  - looking at a long article, policy or document -> summarize it into key points;
  - looking at tables or invoices -> extract the key fields into a table;
  - looking at one message in a chat window -> reply to it directly and verify with a screenshot; never write "ask the user to reply".
- Phrase `summary` as "I saw Y in X, so I infer you want Z", and state the evidence in `references` (app name, window title, key information on screen).
- When the screen holds no useful information (only the desktop, a blank window, a login page), keep `confidence` at or below 0.35 and offer two candidate directions in `question` for the user to choose; do not force a guess.
- When the screen sits on a chat, a mail thread or a form, the goal is usually "reply for me / fill this in": write the plan as
  "open -> screenshot to locate -> type -> send -> screenshot to verify" and carry it out yourself; never write "ask the user to reply".
- Before any sending action, state in `summary` "to whom / what / how it will be verified"; never delete user data, and never send mass messages the user did not ask for.
"#;

fn briefing(input: &UnderstandingInput<'_>) -> String {
    let mut out = String::new();
    if is_blank_instruction(input.instruction) {
        out.push_str(BLANK_SCENE);
    } else {
        out.push_str("## User instruction\n");
        out.push_str(input.instruction.trim());
        out.push('\n');
    }
    if let Some(supplement) = input.supplement.map(str::trim).filter(|s| !s.is_empty()) {
        out.push_str("\n## Supplement\n");
        out.push_str(supplement);
        out.push('\n');
    }
    if let Some(pinned) = input.pinned_workspace {
        if let Ok(Some(workspace)) = input.store.workspace(pinned) {
            out.push_str(&format!(
                "\n## Workspace chosen by the user\n{} ({})\n",
                workspace.name,
                workspace.root_dir.display()
            ));
        }
    }
    // The environment snapshot ships with the first round: the frontmost app and window title
    // are available locally right away, and making the model spend a whole extra round on them
    // only makes "Understand" wait a few seconds for nothing.
    out.push_str("\n(The environment information below is already provided with the instruction; do not call get_environment just to fetch it again.)\n");
    out.push_str(&environment_text(input));
    if let Some(scene) = &input.prefetched {
        if scene.attached {
            out.push_str(
                "\n## Screenshot attached by the user (region chosen, with annotations)\n",
            );
            out.push_str(&scene.text);
            out.push_str(
                "\nThe user picked this region and drew on it: rectangles, circles, arrows and text notes mark what matters. Read the annotations as the user's emphasis, not as part of the app content. The full display is not included; do not call take_screenshot again unless another window or a wider view is needed.\n",
            );
        } else {
            out.push_str("\n## Current screen (captured automatically)\n");
            out.push_str(&scene.text);
            out.push_str("\nThe screen is already captured: do not call take_screenshot again unless you need another app's window; use computer_accessibility op=screenshot for the whole display or for pixel coordinates.\n");
        }
    }
    let tail = if input.prefetched.is_some() {
        "\n## Notes\nThe environment and the screen content are both above. Add read_clipboard / read_file / list_* as needed; \
submit_plan as soon as the information is enough, and never route workspaces on a guess.\n"
    } else {
        "\n## Notes\nThe environment information is above; the screen content is not yet. Use take_screenshot to look at the screen when needed, \
then add read_clipboard / read_file / list_* as needed; submit_plan as soon as the information is enough, and never route workspaces on a guess.\n"
    };
    out.push_str(tail);
    out
}

/// Capture the current window before the first round: convert it to text when a vision model is
/// configured, otherwise hand the image itself to the main model.
///
/// Returns `None` on failure (no screen-recording permission, no capturable window, recognition
/// error); the understanding agent then falls back to its own `take_screenshot` path, and the
/// task still moves on.
pub(crate) async fn prefetch_scene(
    model: &dyn LanguageModel,
    vision_model: Option<&str>,
    dir: &Path,
) -> Option<PrefetchedScene> {
    let capture_dir = dir.to_path_buf();
    let shot =
        tokio::task::spawn_blocking(move || relay_platform::capture_user_window_png(&capture_dir))
            .await
            .ok()?
            .ok()?;
    let header = describe_capture(&shot.app, &shot.title, &shot.path);

    if vision_model.is_none() {
        return Some(PrefetchedScene {
            text: format!("{header}\n(this image is attached to this message as an image)"),
            image: Some(shot.path),
            attached: false,
        });
    }

    let path = shot.path.clone();
    let prepared = tokio::task::spawn_blocking(move || vision::prepare(&path, SCREENSHOT_MAX_EDGE))
        .await
        .ok()?
        .ok()?;
    match vision::describe(
        model,
        std::slice::from_ref(&prepared),
        SCREENSHOT_PROMPT,
        SCREENSHOT_MAX_EDGE,
        vision_model,
    )
    .await
    {
        Ok(description) => Some(PrefetchedScene {
            text: format!("{header}\n--- screen content ---\n{description}"),
            image: None,
            attached: false,
        }),
        Err(err) => {
            tracing::warn!(%err, "first-round screen recognition failed, falling back to the model taking its own screenshot");
            None
        }
    }
}

/// Scene for a screenshot the user attached in the panel: the selected region plus their markup.
pub(crate) async fn attached_scene(
    model: &dyn LanguageModel,
    vision_model: Option<&str>,
    path: &Path,
) -> PrefetchedScene {
    let header = format!(
        "User-provided screenshot (cropped from the desktop, with annotations): {}",
        path.display()
    );
    if vision_model.is_none() {
        return PrefetchedScene {
            text: format!(
                "{header}\n(this image is attached to this message as an image; the user's shapes and notes are part of the picture)"
            ),
            image: Some(path.to_path_buf()),
            attached: true,
        };
    }

    let owned = path.to_path_buf();
    let prepared =
        tokio::task::spawn_blocking(move || vision::prepare(&owned, SCREENSHOT_MAX_EDGE))
            .await
            .ok()
            .and_then(|prepared| prepared.ok());
    let Some(prepared) = prepared else {
        return PrefetchedScene {
            text: format!("{header}\n(the image could not be prepared for recognition)"),
            image: None,
            attached: true,
        };
    };
    match vision::describe(
        model,
        std::slice::from_ref(&prepared),
        ATTACHED_SCREENSHOT_PROMPT,
        SCREENSHOT_MAX_EDGE,
        vision_model,
    )
    .await
    {
        Ok(description) => PrefetchedScene {
            text: format!("{header}\n--- screenshot content ---\n{description}"),
            image: None,
            attached: true,
        },
        Err(err) => {
            tracing::warn!(%err, "attached screenshot recognition failed");
            PrefetchedScene {
                text: format!("{header}\n(recognition failed; the image is not attached)"),
                image: None,
                attached: true,
            }
        }
    }
}

const ATTACHED_SCREENSHOT_PROMPT: &str = "This is a screenshot the user cropped from their desktop and annotated to explain a task.\n\
1. Identify the app and page (title bar / tab / header).\n\
2. Read every annotation first: rectangle / circle / arrow / text note. State what each one points at and quote the notes verbatim.\n\
3. Then transcribe the text, numbers, fields and buttons inside the cropped region, item by item per area.\n\
4. Write \"illegible\" for anything unclear; do not guess.\n\
Return the annotations and the content, not speculation about intent.";

fn tool_defs() -> Vec<ToolDef> {
    vec![
        computer::accessibility_def(),
        ToolDef::new(
            "get_environment",
            "Read the current frontmost app, window title, selected files and authorized folders (live). This is already provided with the instruction, so it is rarely needed; use it only when you suspect the snapshot is stale.",
            json!({"type":"object","properties":{},"required":[]}),
        ),
        ToolDef::new(
            "take_screenshot",
            "Capture a window and look at it directly: the frontmost window the user is viewing by default, or a named app. \
For a full-display capture, or when you need coordinates from pixels, use computer_accessibility with op=screenshot (it reports scale and origin). \
With a blank instruction this is the primary evidence for the goal, and it can be called in the same round as read_clipboard.",
            json!({
                "type":"object",
                "properties":{
                    "scope":{
                        "type":"string",
                        "enum":["window","app"],
                        "default":"window",
                        "description":"window=the user's frontmost window (default; falls back to the whole display when it cannot be captured); app=a window of the named app"
                    },
                    "app":{
                        "type":"string",
                        "description":"App name when scope=app, e.g. \"WeChat\" or \"Safari\"; case-insensitive, partial matches work"
                    }
                },
                "required":[]
            }),
        ),
        tools::read_image_def(),
        tools::search_def(),
        tools::fetch_def(),
        ToolDef::new(
            "read_clipboard",
            "Read the current clipboard text.",
            json!({"type":"object","properties":{},"required":[]}),
        ),
        ToolDef::new(
            "list_workspaces",
            "List all workspaces: name, directory, keywords, rules summary, session count and number of running tasks. With a blank instruction, call it only when deciding where work lands; it cannot be used to infer the goal.",
            json!({"type":"object","properties":{},"required":[]}),
        ),
        ToolDef::new(
            "list_sessions",
            "List a workspace's sessions (ranked by relevance to the current instruction) to decide whether to continue an existing one.",
            json!({
                "type":"object",
                "properties":{
                    "workspace":{"type":"string","description":"Workspace name or id; omit for all of them"},
                    "limit":{"type":"integer","description":"Maximum number to return, default 8"}
                },
                "required":[]
            }),
        ),
        ToolDef::new(
            "list_tasks",
            "List tasks: scope=running for tasks that are executing or awaiting confirmation, scope=recent for recently completed tasks.",
            json!({
                "type":"object",
                "properties":{
                    "scope":{"type":"string","enum":["running","recent"],"description":"Default running"},
                    "workspace":{"type":"string","description":"Restrict to a workspace name or id"},
                    "limit":{"type":"integer","description":"Maximum number of entries, default 10"}
                },
                "required":[]
            }),
        ),
        tools::list_files_def(),
        tools::read_file_def(),
        ToolDef::new(
            "submit_plan",
            "Submit the final understanding: choose or create the workspace and session, and give a plan the agent will carry out itself. Call it as soon as the information is enough; do not spin.",
            json!({
                "type":"object",
                "properties":{
                    "workspace_mode":{"type":"string","enum":["reuse","create"]},
                    "workspace_id":{"type":"string","description":"workspace id to reuse"},
                    "workspace_name":{"type":"string","description":"name when creating; include the business context"},
                    "workspace_keywords":{"type":"array","items":{"type":"string"},"description":"keywords for future search when creating, 2-6 of them"},
                    "session_mode":{"type":"string","enum":["reuse","create"]},
                    "session_id":{"type":"string","description":"session id to reuse"},
                    "session_title":{"type":"string","description":"session title when creating"},
                    "title":{"type":"string","description":"one-line task title"},
                    "summary":{"type":"string","description":"goal and approach, one paragraph"},
                    "steps":{"type":"array","items":{"type":"string"},"description":"2-6 step plan: every step is carried out by the agent itself with bash/CLI; never write \"ask the user to...\" or \"wait for the user to...\""},
                    "references":{"type":"array","items":{"type":"string"},"description":"reference materials or environment clues; empty array when there are none"},
                    "delivery":{"type":"string","enum":["open_file","open_folder","notify_only"]},
                    "confidence":{"type":"number","description":"0-1, overall confidence in the routing and the plan"},
                    "reason":{"type":"string","description":"one sentence on why this workspace and session"},
                    "question":{"type":"string","description":"question for the user when information is insufficient; leave empty otherwise"}
                },
                "required":["workspace_mode","title","summary","steps","delivery","confidence","reason"]
            }),
        ),
    ]
}

fn environment_text(input: &UnderstandingInput<'_>) -> String {
    let mut out = String::new();
    out.push_str("## Environment snapshot at trigger time\n");
    out.push_str(&input.context.to_prompt());
    let snapshot_is_self = relay_platform::is_shell_app(&input.context.app_name);
    if snapshot_is_self {
        out.push_str("\n(The frontmost app in the snapshot is Relay itself or a system shell, which does not represent what the user is looking at; rely on the live window below.)\n");
    }
    if let Some(window) =
        relay_platform::user_front_window().or_else(relay_platform::last_user_window)
    {
        out.push_str(
            "\n## Window the user is looking at (Relay panels and system shells excluded)\n",
        );
        out.push_str(&format!("app: {}\ntitle: {}\n", window.app, window.title));
    }
    out
}

fn list_workspaces(input: &UnderstandingInput<'_>) -> Result<String> {
    let workspaces = input.store.workspaces()?;
    if workspaces.is_empty() {
        return Ok("There are no workspaces yet; create one with workspace_mode=create.".into());
    }
    let unfinished = input.store.unfinished_tasks()?;
    let mut out = String::new();
    for (index, workspace) in workspaces.iter().enumerate() {
        let sessions = input.store.sessions(workspace.id, 3).unwrap_or_default();
        let running = unfinished
            .iter()
            .filter(|task| task.workspace_id == workspace.id && task.state.is_active())
            .count();
        out.push_str(&format!(
            "{}. {} | id={} | dir={}\n",
            index + 1,
            workspace.name,
            workspace.id,
            workspace.root_dir.display()
        ));
        if !workspace.keywords.is_empty() {
            out.push_str(&format!("   keywords: {}\n", workspace.keywords.join(", ")));
        }
        let rules = workspace.rules.trim();
        if !rules.is_empty() {
            out.push_str(&format!(
                "   rules summary: {}\n",
                tools::truncate(rules, 160)
            ));
        }
        if let Some(latest) = sessions.first() {
            out.push_str(&format!(
                "   latest session: {} (of {} sessions, last active {})\n",
                latest.title,
                sessions.len(),
                human_age(latest.last_active_at)
            ));
        }
        out.push_str(&format!("   running tasks: {running}\n"));
    }
    Ok(out)
}

fn list_sessions(input: &UnderstandingInput<'_>, args: &Value) -> Result<String> {
    let limit = arg_usize(args, "limit", 8).min(20);
    let workspaces = input.store.workspaces()?;
    let target = args.get("workspace").and_then(Value::as_str);
    let selected: Vec<&Workspace> = match target.map(str::trim).filter(|s| !s.is_empty()) {
        Some(needle) => workspaces
            .iter()
            .filter(|w| w.name.contains(needle) || w.id.to_string() == needle)
            .collect(),
        None => workspaces.iter().collect(),
    };
    if selected.is_empty() {
        let names: Vec<String> = workspaces.iter().map(|w| w.name.clone()).collect();
        return Ok(format!(
            "No matching workspace: {target:?}. Available workspaces: {}",
            if names.is_empty() {
                "(no workspaces yet)".to_string()
            } else {
                names.join(", ")
            }
        ));
    }
    let mut out = String::new();
    for workspace in selected {
        let sessions = input.store.sessions(workspace.id, 40).unwrap_or_default();
        if sessions.is_empty() {
            out.push_str(&format!("{}: no sessions yet\n", workspace.name));
            continue;
        }
        let ranked = relay_router::rank_sessions(input.instruction, &sessions, Utc::now());
        out.push_str(&format!("## {}\n", workspace.name));
        for ranked_session in ranked.iter().take(limit) {
            let session = &ranked_session.session;
            out.push_str(&format!(
                "- {} | id={} | active {}\n  summary: {}\n",
                session.title,
                session.id,
                human_age(session.last_active_at),
                tools::truncate(session.summary.trim(), 120)
            ));
            if let Some(context) = session
                .last_context
                .as_deref()
                .filter(|s| !s.trim().is_empty())
            {
                out.push_str(&format!(
                    "  last environment: {}\n",
                    tools::truncate(context.trim(), 80)
                ));
            }
        }
    }
    Ok(out)
}

fn list_tasks(input: &UnderstandingInput<'_>, args: &Value) -> Result<String> {
    let scope = args
        .get("scope")
        .and_then(Value::as_str)
        .unwrap_or("running");
    let limit = arg_usize(args, "limit", 10).min(20);
    let workspace_filter = args.get("workspace").and_then(Value::as_str);
    let workspaces = input.store.workspaces()?;
    let name_of = |id: WorkspaceId| {
        workspaces
            .iter()
            .find(|w| w.id == id)
            .map(|w| w.name.clone())
            .unwrap_or_else(|| "unknown".into())
    };

    let tasks = if scope == "recent" {
        input.store.recent_tasks(limit as u32 * 2)?
    } else {
        input.store.unfinished_tasks()?
    };
    let mut out = String::new();
    for task in tasks {
        if let Some(needle) = workspace_filter {
            let name = name_of(task.workspace_id);
            if !name.contains(needle) && task.workspace_id.to_string() != needle {
                continue;
            }
        }
        if scope != "recent"
            && !task.state.is_active()
            && task.state.as_str() != "awaiting_confirmation"
        {
            continue;
        }
        out.push_str(&format!(
            "- [{}] {} | workspace {}\n  updated {} | {}\n",
            task.state.label(),
            task.title,
            name_of(task.workspace_id),
            human_age(task.updated_at),
            task.id.to_string().chars().take(8).collect::<String>()
        ));
        if out.lines().count() > limit * 3 {
            break;
        }
    }
    if out.trim().is_empty() {
        out.push_str(if scope == "recent" {
            "No recent tasks."
        } else {
            "No tasks are running right now."
        });
    }
    Ok(out)
}

fn list_files(input: &UnderstandingInput<'_>, args: &Value) -> Result<String> {
    Ok(tools::list_files_tool(
        &input.store.workspaces()?,
        None,
        args,
    ))
}

fn read_file(input: &UnderstandingInput<'_>, args: &Value) -> Result<String> {
    Ok(tools::read_file_tool(&input.allowed_dirs, args))
}

struct Decision {
    workspace: WorkspaceChoice,
    session: SessionChoice,
    plan: Plan,
    confidence: f32,
    reason: String,
}

/// Pull `submit_plan` out of the tool calls; returns None when it is absent (continue to the next
/// round).
fn take_decision(calls: &[ToolCall]) -> Option<Decision> {
    let call = calls.iter().find(|call| call.name == "submit_plan")?;
    let args = &call.arguments;

    let text = |key: &str| -> String {
        args.get(key)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_string()
    };
    let list = |key: &str| -> Vec<String> {
        args.get(key)
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(|item| item.trim().to_string())
                    .filter(|item| !item.is_empty())
                    .collect()
            })
            .unwrap_or_default()
    };

    let workspace = if text("workspace_mode") == "create" {
        WorkspaceChoice::Create {
            name: text("workspace_name"),
            keywords: list("workspace_keywords"),
        }
    } else {
        WorkspaceChoice::Reuse(text("workspace_id"))
    };
    let session = if text("session_mode") == "reuse" && !text("session_id").is_empty() {
        SessionChoice::Reuse(text("session_id"))
    } else {
        SessionChoice::Create {
            title: text("session_title"),
        }
    };

    let delivery = text("delivery").parse().unwrap_or(DeliveryMode::OpenFile);
    let mut summary = text("summary");
    let question = text("question");
    if !question.is_empty() {
        summary = format!("{summary}\nNeeds clarification: {question}");
    }

    let plan = Plan {
        title: text("title"),
        summary,
        steps: list("steps"),
        references: list("references"),
        delivery,
        raw: call.arguments.to_string(),
    };

    Some(Decision {
        workspace,
        session,
        plan,
        confidence: args
            .get("confidence")
            .and_then(Value::as_f64)
            .unwrap_or(0.5) as f32,
        reason: text("reason"),
    })
}

/// Fallback when the model does not submit by protocol: reuse the legacy text protocol and derive
/// the session name from the instruction, leaving workspace creation to the Runtime.
fn fallback_understanding(input: &UnderstandingInput<'_>, raw: &str) -> Understanding {
    let plan =
        crate::agents::parse_plan(raw, &relay_router::derive_session_title(input.instruction));
    let title = relay_router::derive_session_title(input.instruction);
    let workspace = if let Some(pinned) = input.pinned_workspace {
        WorkspaceChoice::Reuse(pinned.to_string())
    } else {
        WorkspaceChoice::Create {
            name: String::new(),
            keywords: Vec::new(),
        }
    };
    Understanding {
        workspace,
        session: SessionChoice::Create { title },
        plan,
        confidence: 0.3,
        reason: "the model did not call submit_plan, fell back to its text".into(),
        trail: Vec::new(),
        usage: ContextUsage::default(),
    }
}

fn arg_usize(args: &Value, key: &str, default: usize) -> usize {
    args.get(key)
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .unwrap_or(default)
}

fn human_age(at: chrono::DateTime<Utc>) -> String {
    let seconds = (Utc::now() - at).num_seconds().max(0);
    if seconds < 60 {
        "just now".into()
    } else if seconds < 3600 {
        format!("{} minutes ago", seconds / 60)
    } else if seconds < 86_400 {
        format!("{} hours ago", seconds / 3600)
    } else {
        format!("{} days ago", seconds / 86_400)
    }
}

fn clip(text: &str) -> String {
    tools::truncate(text.trim(), 1200)
}
