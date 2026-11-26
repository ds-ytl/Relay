use std::path::{Path, PathBuf};

use relay_core::{
    ChatMessageRecord, ControlPhase, ControlSignal, DeliveryMode, MessageRole, Plan, Task, TaskId,
    TaskReport, Verdict,
};
use relay_openai::{InputItem, ToolCall};
use serde_json::Value;

pub const UNDERSTAND_PROMPT: &str = include_str!("../../../prompts/understand.md");
pub const EXECUTOR_PROMPT: &str = include_str!("../../../prompts/executor.md");
pub const VERIFIER_PROMPT: &str = include_str!("../../../prompts/verifier.md");

/// One tool call inside the transcript.
#[derive(Debug, Clone)]
pub struct TranscriptStep {
    pub tool: String,
    /// The command, or a summary of the arguments.
    pub input: String,
    /// Tail of the output (truncated again when rendered).
    pub output: String,
    pub exit_code: Option<i32>,
}

/// One execution round: the process reply of this round plus every tool called in it.
#[derive(Debug, Clone)]
pub struct TranscriptEntry {
    pub turn: u32,
    /// Process reply of this round: kept across rounds so the model remembers what it said it
    /// would do.
    pub reply: String,
    pub steps: Vec<TranscriptStep>,
    pub phase: String,
}

/// Answers for tool calls in a recorded input that never got a result:
/// a restart landed between dispatch and the checkpoint write.
///
/// They have to be appended before anything new, because a bare `function_call` is rejected by
/// the API, and dropping the recorded prefix instead would throw away the whole token cache.
pub fn interrupted_outputs(input: &[Value]) -> Vec<Value> {
    use std::collections::HashSet;
    let answered: HashSet<&str> = input
        .iter()
        .filter(|item| item["type"] == "function_call_output")
        .filter_map(|item| item["call_id"].as_str())
        .collect();
    input
        .iter()
        .filter(|item| item["type"] == "function_call")
        .filter_map(|item| item["call_id"].as_str())
        .filter(|call_id| !answered.contains(call_id))
        .map(|call_id| {
            serde_json::json!({
                "type": "function_call_output",
                "call_id": call_id,
                "output": INTERRUPTED_OUTPUT,
            })
        })
        .collect()
}

/// Receipt for a call cut off by a restart: tells the model the outcome is unknown, so it checks
/// the current state instead of assuming the call did nothing.
const INTERRUPTED_OUTPUT: &str = "interrupted: Relay restarted before this call returned, so the outcome is unknown. Inspect the current state (files, task directory) before repeating it.";

/// Heading of the note appended when a recorded request had to replace a missing checkpoint: the
/// reply and tool results of that round were never recorded, so the model must not assume they
/// happened. It stays a fixed string, so it caches like any other suffix.
pub const INTERRUPTED_HEADING: &str = "## Interrupted round";

/// The note itself, as a new user message after the replayed prefix.
pub fn interrupted_note() -> Value {
    serde_json::json!({
        "role": "user",
        "content": [{
            "type": "input_text",
            "text": format!(
                "{INTERRUPTED_HEADING}\nRelay restarted right after this request was sent, so that round may have been cut off: anything without a result above did not run as recorded, and the state may already have changed. Re-check the files and the task state, then continue."
            ),
        }],
    })
}

/// Per-segment caps when rendering the transcript: they trim the display only, never the
/// original record.
const TRANSCRIPT_REPLY_LIMIT: usize = 400;
const TRANSCRIPT_INPUT_LIMIT: usize = 400;
const TRANSCRIPT_OUTPUT_LIMIT: usize = 800;
/// How many rounds the transcript looks back at most (a second gate next to the token budget).
const MAX_TRANSCRIPT_ROUNDS: usize = 12;
/// Character cap for each side of a question/answer pair in the dialogue history.
const DIALOGUE_TURN_LIMIT: usize = 600;
/// Cap for the first process block in the dialogue history: it only has to convey intent.
const DIALOGUE_PROCESS_LIMIT: usize = 160;
/// How many question/answer rounds the dialogue history looks back at most.
pub(crate) const DIALOGUE_MAX_TURNS: usize = 8;

/// Input for the execution turn, split into four segments by prefix stability (highest first),
/// each rendered as its own Text block.
///
/// Model calls are stateless replays and the server hits its token cache by prefix, where the
/// matched length is decided by the first differing byte. The segment order therefore has to
/// put the least volatile content first.
///
/// 1. `head`: earlier turns of the same session. Byte-identical within a task and only appended
///    within a session, so it is reusable across tasks;
/// 2. `stable`: task, goal, plan, directory, reply rules. Byte-identical within a task;
/// 3. `transcript`: already executed rounds. Only appended at the end with each new round;
/// 4. `tail`: progress and artifact list for this round. Changes every round, so it must be last.
///
/// Counter-example: putting a line that changes every round (such as "round N") first means
/// nothing after it ever hits the cache.
pub struct ExecutorInput {
    pub head: String,
    pub stable: String,
    pub transcript: String,
    pub tail: String,
}

impl ExecutorInput {
    /// Turn this into request input: empty segments take no slot. Also returns the number of
    /// prefix segments, which tells compaction how much cacheable prefix to keep.
    pub fn into_items(self) -> (Vec<InputItem>, usize, String) {
        let mut items = Vec::new();
        for text in [self.head, self.stable] {
            if !text.trim().is_empty() {
                items.push(InputItem::Text(text));
            }
        }
        let prefix_items = items.len();
        if !self.transcript.trim().is_empty() {
            items.push(InputItem::Text(self.transcript));
        }
        if !self.tail.trim().is_empty() {
            items.push(InputItem::Text(self.tail.clone()));
        }
        (items, prefix_items, self.tail)
    }
}

fn confirmed_context(task: &Task, plan: &Plan) -> String {
    let mut stable = String::new();
    stable.push_str(&format!("## Task\n{}\n\n", plan.title));
    if crate::understanding::is_blank_instruction(&task.instruction) {
        // Task submitted with a blank instruction: placeholder text is useless for execution,
        // and the goal is the plan itself.
        stable
            .push_str("## User goal\n(The user entered no text, so the goal is the task title above and the confirmed plan)\n\n");
    } else if task.instruction.trim() != plan.title.trim() {
        // Skip it when identical to the title: the task section already says the same thing.
        stable.push_str(&format!("## User goal\n{}\n\n", task.instruction.trim()));
    }
    if let Some(supplement) = task.supplement.as_deref().filter(|s| !s.trim().is_empty()) {
        stable.push_str(&format!("## Supplement\n{}\n\n", supplement.trim()));
    }
    stable.push_str("## Confirmed plan\n");
    stable.push_str(&relay_core::plan_text(plan));
    stable.push('\n');
    if !task.context.selected_files.is_empty() {
        stable.push_str("Files selected at submission:\n");
        for path in &task.context.selected_files {
            stable.push_str(&format!("- {}\n", path.display()));
        }
    }
    stable
}

#[allow(clippy::too_many_arguments)]
pub fn executor_input(
    task: &Task,
    plan: &Plan,
    task_dir: &Path,
    turn: u32,
    max_rounds: u32,
    dialogue: &str,
    transcript: &[TranscriptEntry],
    known_artifacts: &[PathBuf],
    transcript_budget: u32,
) -> ExecutorInput {
    let mut stable = confirmed_context(task, plan);
    stable.push_str(&format!("\nTask directory: {}\n", task_dir.display()));
    stable.push_str(
        "\nCall tools when you need to act (bash / read_file / list_files / read_image / computer_accessibility / computer_input). \
Before acting in each round, write a 1-3 sentence process reply describing what this round will do; whether or not you finish, always call submit_result at the end to write back the phase, the conclusion and the artifacts. \
With phase=done, first output a Markdown final reply, then call submit_result; \
do not use code fences or paste commands in the reply: every command goes through the bash tool.\n",
    );

    // Volatile content goes last: artifacts first (they change now and then), then progress
    // (which changes every round).
    let mut tail = String::new();
    if !known_artifacts.is_empty() {
        tail.push_str("## Artifacts produced so far\n");
        for path in known_artifacts {
            tail.push_str(&format!("- {}\n", path.display()));
        }
        tail.push('\n');
    }
    tail.push_str(&format!("## Progress\nround {turn} / {max_rounds}\n"));

    ExecutorInput {
        head: dialogue.to_string(),
        stable,
        transcript: render_transcript(transcript, transcript_budget),
        tail,
    }
}

/// Earlier exchanges of the same session: merged into one round per question/answer pair, since
/// the tool transcript already lives in the task steps.
///
/// Rounds are packed newest first within the token budget until it is full. Rendering is
/// deterministic for a given set of messages and finished rounds never change again, so the
/// result is byte-identical within a task and append-only across tasks: a stable cache prefix.
pub fn dialogue_history(messages: &[ChatMessageRecord], skip_task: TaskId, budget: u32) -> String {
    let cutoff = messages
        .iter()
        .find(|m| m.task_id == skip_task && m.role == MessageRole::User)
        .map(|m| m.id);
    dialogue_with_outcomes(
        messages,
        skip_task,
        cutoff,
        budget,
        &std::collections::HashMap::new(),
    )
}

pub(crate) fn dialogue_with_outcomes(
    messages: &[ChatMessageRecord],
    skip_task: TaskId,
    cutoff: Option<i64>,
    budget: u32,
    outcomes: &std::collections::HashMap<TaskId, String>,
) -> String {
    struct Turn {
        user: String,
        /// One entry per streamed block: process replies first, the closing conclusion last.
        assistant: Vec<String>,
    }

    // Group by task into question/answer pairs: a task's streamed reply blocks merge into a
    // single assistant message.
    let mut turns: Vec<(TaskId, Turn)> = Vec::new();
    for record in messages {
        if record.task_id == skip_task {
            continue;
        }
        let body = record.content.trim();
        if body.is_empty() {
            continue;
        }
        match record.role {
            MessageRole::User if cutoff.is_none_or(|id| record.id < id) => turns.push((
                record.task_id,
                Turn {
                    user: body.to_string(),
                    assistant: Vec::new(),
                },
            )),
            MessageRole::Assistant => {
                let found = turns
                    .iter_mut()
                    .rev()
                    .find(|(task_id, _)| *task_id == record.task_id);
                if let Some((_, turn)) = found {
                    turn.assistant.push(body.to_string());
                }
            }
            // plan / thinking / tool / notice / error are process traces and stay out of the
            // dialogue history.
            _ => {}
        }
    }

    let mut blocks: Vec<String> = Vec::new();
    let mut used = 0u32;
    for (id, turn) in turns.iter().rev() {
        if blocks.len() >= DIALOGUE_MAX_TURNS {
            break;
        }
        let mut block = format!("\nUser: {}", truncate(&turn.user, DIALOGUE_TURN_LIMIT));
        let reply = render_reply(&turn.assistant);
        if !reply.is_empty() {
            block.push_str(&format!("\nAssistant: {reply}"));
        }
        if let Some(outcome) = outcomes.get(id) {
            block.push_str(outcome);
        }
        let cost = crate::budget::text_tokens(&block);
        if !blocks.is_empty() && used + cost > budget {
            break;
        }
        used += cost;
        blocks.push(block);
    }
    if blocks.is_empty() {
        return String::new();
    }
    blocks.reverse();
    let mut out = String::from(
        "## Conversation history\n(earlier rounds of this session, in chronological order; rely on task state and reports for execution results)\n",
    );
    for block in blocks {
        out.push_str(&block);
    }
    out.push('\n');
    out
}

/// An assistant round may have several blocks (process reply + closing conclusion). Joining all
/// of them blows the budget, while joining only one lets process chatter drown the conclusion —
/// so the first block keeps one line of intent and the last keeps the full conclusion.
fn render_reply(blocks: &[String]) -> String {
    match blocks.len() {
        0 => String::new(),
        1 => truncate(blocks[0].trim(), DIALOGUE_TURN_LIMIT),
        _ => {
            let first = truncate(blocks[0].trim(), DIALOGUE_PROCESS_LIMIT);
            let last = truncate(blocks[blocks.len() - 1].trim(), DIALOGUE_TURN_LIMIT);
            format!("{first}\n(process omitted)\n{last}")
        }
    }
}

/// Render executed rounds: command + output + notes, packed newest first within the token
/// budget.
fn render_transcript(transcript: &[TranscriptEntry], budget: u32) -> String {
    if transcript.is_empty() {
        return String::new();
    }
    let (_, mut blocks) = transcript_blocks(transcript, budget);
    blocks.reverse();
    let mut out = String::from("## Execution history\n");
    for block in blocks {
        out.push_str(&block);
    }
    out
}

/// Earlier rounds pushed out by the budget must be folded into the rolling summary before they
/// are dropped.
pub(crate) fn transcript_blocks(
    transcript: &[TranscriptEntry],
    budget: u32,
) -> (usize, Vec<String>) {
    let mut blocks: Vec<String> = Vec::new();
    let mut used = 0u32;
    for entry in transcript.iter().rev() {
        if blocks.len() >= MAX_TRANSCRIPT_ROUNDS {
            break;
        }
        let block = render_transcript_entry(entry);
        let cost = crate::budget::text_tokens(&block);
        // Always keep the most recent round: even with a tiny budget the model has to know what
        // happened last round.
        if !blocks.is_empty() && used + cost > budget {
            break;
        }
        used += cost;
        blocks.push(block);
    }
    (transcript.len() - blocks.len(), blocks)
}

fn render_transcript_entry(entry: &TranscriptEntry) -> String {
    let mut out = format!("\n### Round {} (phase={})\n", entry.turn, entry.phase);
    if !entry.reply.trim().is_empty() {
        out.push_str("notes: ");
        out.push_str(&truncate(entry.reply.trim(), TRANSCRIPT_REPLY_LIMIT));
        out.push('\n');
    }
    for step in &entry.steps {
        out.push_str(&format!(
            "[{}] {}\n",
            step.tool,
            truncate(step.input.trim(), TRANSCRIPT_INPUT_LIMIT)
        ));
        if let Some(code) = step.exit_code {
            if code != 0 {
                out.push_str(&format!("exit={code}\n"));
            }
        }
        out.push_str("output: ");
        out.push_str(&truncate(step.output.trim(), TRANSCRIPT_OUTPUT_LIMIT));
        out.push('\n');
    }
    out
}

pub fn verifier_input(
    task: &Task,
    plan: &Plan,
    artifacts: &[PathBuf],
    last_output: &str,
) -> String {
    let mut out = confirmed_context(task, plan);
    out.push_str("\n## Artifact list\n");
    if artifacts.is_empty() {
        out.push_str("(no new files detected)\n");
    } else {
        for path in artifacts {
            out.push_str(&format!("- {}\n", path.display()));
        }
    }
    out.push_str("\n## Last round output\n");
    out.push_str(&truncate(last_output, 3000));
    out.push_str("\n\nAfter checking, you must call submit_verification to submit the conclusion (verdict / summary / checks / deliver); do not output explanatory text.");
    out
}

/// Parse the Understand Agent's structured text; missing fields fall back to defaults so the
/// task can continue.
pub fn parse_plan(raw: &str, fallback_title: &str) -> Plan {
    let mut title = None;
    let mut summary = String::new();
    let mut steps = Vec::new();
    let mut references = Vec::new();
    let mut delivery = DeliveryMode::OpenFile;
    let mut section = Section::None;

    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(value) = strip_field(trimmed, &["TITLE:"]) {
            if !value.is_empty() {
                title = Some(value.to_string());
            }
            section = Section::None;
            continue;
        }
        if let Some(value) = strip_field(trimmed, &["SUMMARY:"]) {
            summary = value.to_string();
            section = Section::Summary;
            continue;
        }
        if strip_field(trimmed, &["STEPS:"]).is_some() {
            section = Section::Steps;
            continue;
        }
        if strip_field(trimmed, &["REFERENCES:"]).is_some() {
            section = Section::References;
            continue;
        }
        if let Some(value) = strip_field(trimmed, &["DELIVERY:"]) {
            if let Ok(parsed) = value.parse() {
                delivery = parsed;
            }
            section = Section::None;
            continue;
        }
        match section {
            Section::Summary => {
                if !summary.is_empty() {
                    summary.push(' ');
                }
                summary.push_str(trimmed);
            }
            Section::Steps => {
                if let Some(item) = bullet(trimmed) {
                    steps.push(item);
                }
            }
            Section::References => {
                if let Some(item) = bullet(trimmed) {
                    if !item.is_empty() && !item.eq_ignore_ascii_case("none") {
                        references.push(item);
                    }
                }
            }
            _ => {}
        }
    }

    Plan {
        title: title
            .filter(|t| !t.is_empty())
            .unwrap_or_else(|| fallback_title.to_string()),
        summary: if summary.is_empty() {
            fallback_title.to_string()
        } else {
            summary
        },
        steps,
        references,
        delivery,
        raw: raw.to_string(),
    }
}

fn strip_field<'a>(line: &'a str, keys: &[&str]) -> Option<&'a str> {
    for key in keys {
        if let Some(prefix) = line.get(..key.len()) {
            if prefix.eq_ignore_ascii_case(key) {
                return Some(line[key.len()..].trim());
            }
        }
    }
    None
}

fn bullet(line: &str) -> Option<String> {
    let trimmed = line.trim_start_matches(['-', '*', '•', ' ']);
    let trimmed = trimmed.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verification {
    pub verdict: Verdict,
    pub summary: String,
    pub checks: Vec<String>,
    pub deliver: Option<PathBuf>,
}

/// Parse the execution turn's `submit_result`: phase, conclusion, artifacts and delivery mode.
pub fn parse_result(call: &ToolCall) -> ControlSignal {
    let args = &call.arguments;
    let text = |key: &str| -> String {
        args.get(key)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_string()
    };
    let phase = text("phase").parse().unwrap_or(ControlPhase::Continue);
    let artifacts = args
        .get("artifacts")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(|item| item.trim().to_string())
                .filter(|item| !item.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let delivery = args
        .get("delivery")
        .and_then(Value::as_str)
        .and_then(|value| value.parse().ok());
    ControlSignal {
        phase,
        summary: text("summary"),
        artifacts,
        delivery,
    }
}

/// Parse the verification turn's `submit_verification`; a missing field counts as FAIL, never a
/// guessed PASS.
pub fn parse_verification_call(call: &ToolCall) -> Verification {
    let args = &call.arguments;
    let text = |key: &str| -> String {
        args.get(key)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_string()
    };
    let verdict = if text("verdict").eq_ignore_ascii_case("pass") {
        Verdict::Pass
    } else {
        Verdict::Fail
    };
    let checks = args
        .get("checks")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(|item| item.trim().to_string())
                .filter(|item| !item.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let deliver = text("deliver");
    let deliver = if deliver.is_empty() || deliver.eq_ignore_ascii_case("none") {
        None
    } else {
        Some(PathBuf::from(deliver))
    };
    let summary = text("summary");
    Verification {
        verdict,
        summary: if summary.is_empty() {
            match verdict {
                Verdict::Pass => "verification passed".to_string(),
                Verdict::Fail => "verification failed".to_string(),
            }
        } else {
            summary
        },
        checks,
        deliver,
    }
}

pub fn parse_verification(raw: &str) -> Verification {
    let mut verdict = Verdict::Fail;
    let mut summary = String::new();
    let mut checks = Vec::new();
    let mut deliver = None;
    let mut section = Section::None;

    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(value) = strip_field(trimmed, &["VERDICT:"]) {
            if value.to_uppercase().contains("PASS") {
                verdict = Verdict::Pass;
            }
            section = Section::None;
            continue;
        }
        if let Some(value) = strip_field(trimmed, &["SUMMARY:"]) {
            summary = value.to_string();
            section = Section::Summary;
            continue;
        }
        if strip_field(trimmed, &["CHECKS:"]).is_some() {
            section = Section::Checks;
            continue;
        }
        if let Some(value) = strip_field(trimmed, &["DELIVER:"]) {
            if !value.is_empty() && !value.eq_ignore_ascii_case("none") {
                deliver = Some(PathBuf::from(value));
            }
            section = Section::None;
            continue;
        }
        match section {
            Section::Checks => {
                if let Some(item) = bullet(trimmed) {
                    checks.push(item);
                }
            }
            Section::Summary => {
                if !summary.is_empty() {
                    summary.push(' ');
                }
                summary.push_str(trimmed);
            }
            _ => {}
        }
    }

    if summary.is_empty() {
        summary = match verdict {
            Verdict::Pass => "verification passed".to_string(),
            Verdict::Fail => "verification failed".to_string(),
        };
    }

    Verification {
        verdict,
        summary,
        checks,
        deliver,
    }
}

pub fn report_from(verification: Verification, delivered: Vec<String>) -> TaskReport {
    TaskReport {
        verdict: Some(verification.verdict),
        summary: verification.summary,
        checks: verification.checks,
        delivered,
    }
}

pub fn build_task_report(verification: Verification) -> TaskReport {
    report_from(verification, Vec::new())
}

pub fn workspace_listing(root: &Path, limit: usize) -> Vec<String> {
    let mut entries: Vec<String> = Vec::new();
    let Ok(read) = std::fs::read_dir(root) else {
        return entries;
    };
    for entry in read.flatten() {
        if entries.len() >= limit {
            break;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') {
            continue;
        }
        let meta = entry.metadata().ok();
        let kind = if meta.as_ref().map(|m| m.is_dir()).unwrap_or(false) {
            "directory"
        } else {
            "file"
        };
        entries.push(format!("{name} ({kind})"));
    }
    entries.sort();
    entries
}

fn truncate(value: &str, max: usize) -> String {
    if value.chars().count() <= max {
        return value.to_string();
    }
    let mut out: String = value.chars().take(max).collect();
    out.push_str("\n…(truncated)");
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Section {
    None,
    Summary,
    Steps,
    References,
    Checks,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_plan_sections() {
        let raw = "TITLE: Fill in the project registration form\nSUMMARY: Sort out the WeChat attachments and fill in the form.\nSTEPS:\n- Collect the attachments\n- Fill in the table\nREFERENCES:\n- Project A materials\nDELIVERY: open_folder";
        let plan = parse_plan(raw, "fallback");
        assert_eq!(plan.title, "Fill in the project registration form");
        assert_eq!(plan.steps.len(), 2);
        assert_eq!(plan.references, vec!["Project A materials"]);
        assert_eq!(plan.delivery, DeliveryMode::OpenFolder);
    }

    #[test]
    fn plan_falls_back_when_unstructured() {
        let plan = parse_plan(
            "I understand you want to tidy up the files.",
            "tidy up the files",
        );
        assert_eq!(plan.title, "tidy up the files");
        assert!(plan.steps.is_empty());
    }

    #[test]
    fn parses_verification_fail() {
        let raw =
            "VERDICT: FAIL\nSUMMARY: The table is missing two columns\nCHECKS:\n- The file exists\n- The fields are incomplete\nDELIVER: none";
        let v = parse_verification(raw);
        assert_eq!(v.verdict, Verdict::Fail);
        assert_eq!(v.checks.len(), 2);
        assert!(v.deliver.is_none());
    }

    #[test]
    fn parses_verification_pass_with_deliverable() {
        let raw = "VERDICT: PASS\nSUMMARY: Done\nCHECKS:\n- The file opens\nDELIVER: project-registration.xlsx";
        let v = parse_verification(raw);
        assert_eq!(v.verdict, Verdict::Pass);
        assert_eq!(v.deliver, Some(PathBuf::from("project-registration.xlsx")));
    }

    #[test]
    fn lists_workspace_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "x").unwrap();
        std::fs::create_dir(dir.path().join("materials")).unwrap();
        let listing = workspace_listing(dir.path(), 10);
        assert!(listing.iter().any(|l| l.contains("a.txt")));
        assert!(listing.iter().any(|l| l.contains("materials")));
    }
}
