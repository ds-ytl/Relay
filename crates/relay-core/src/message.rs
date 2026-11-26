use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::ids::{SessionId, TaskId};

/// Role of a session message block: one record is one bubble / card in the UI.
///
/// Messages are displayed in persistence order, so user messages, thinking, tool calls and replies
/// can interleave in real chronological order, and history replay shows the same data as streaming.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    User,
    Plan,
    Thinking,
    Assistant,
    Tool,
    Notice,
    Error,
}

impl MessageRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Plan => "plan",
            Self::Thinking => "thinking",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
            Self::Notice => "notice",
            Self::Error => "error",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "user" => Self::User,
            "plan" => Self::Plan,
            "thinking" => Self::Thinking,
            "assistant" => Self::Assistant,
            "tool" => Self::Tool,
            "notice" => Self::Notice,
            "error" => Self::Error,
            _ => return None,
        })
    }

    /// Thinking/assistant are streamed in as blocks; every other role is written once, in full.
    pub fn is_stream(self) -> bool {
        matches!(self, Self::Thinking | Self::Assistant)
    }
}

/// Execution result of a tool message; the script body lives in `ChatMessageRecord::content` and
/// the result lives here.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolMeta {
    #[serde(default)]
    pub done: bool,
    #[serde(default)]
    pub exit_code: Option<i32>,
    #[serde(default)]
    pub timed_out: bool,
    #[serde(default)]
    pub duration_ms: u64,
    #[serde(default)]
    pub output: String,
}

/// One persisted session message; the single data source for both history and streaming display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessageRecord {
    pub id: i64,
    pub session_id: SessionId,
    pub task_id: TaskId,
    pub turn: u32,
    pub role: MessageRole,
    pub content: String,
    pub meta: Option<ToolMeta>,
    pub created_at: DateTime<Utc>,
}

/// Body of a plan message: summary + steps + references, so a replay shows the intent at a glance.
pub fn plan_text(plan: &crate::plan::Plan) -> String {
    let mut body = if plan.summary.trim().is_empty() {
        plan.title.clone()
    } else {
        plan.summary.clone()
    };
    for (index, step) in plan.steps.iter().enumerate() {
        body.push_str(&format!("\n{}. {}", index + 1, step));
    }
    if !plan.references.is_empty() {
        body.push_str(&format!("\n\nReferences: {}", plan.references.join(" · ")));
    }
    body
}

/// Body of a user message: the original instruction + any supplement.
pub fn user_text(instruction: &str, supplement: Option<&str>) -> String {
    let mut body = instruction.trim().to_string();
    if let Some(supplement) = supplement {
        let supplement = supplement.trim();
        if !supplement.is_empty() {
            body.push_str("\n\nSupplement: ");
            body.push_str(supplement);
        }
    }
    body
}
