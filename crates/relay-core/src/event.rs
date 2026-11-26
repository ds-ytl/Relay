use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::ids::{SessionId, TaskId, WorkspaceId};
use crate::model::{TaskReport, Verdict};
use crate::plan::Plan;
use crate::state::TaskState;

/// Task events the runtime broadcasts: UI popups, the control center and the CLI share one stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeEvent {
    pub task_id: TaskId,
    pub at: DateTime<Utc>,
    pub kind: RuntimeEventKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RuntimeEventKind {
    Queued,
    Routed {
        workspace_id: WorkspaceId,
        workspace_name: String,
        session_id: SessionId,
        session_title: String,
        confidence: f32,
        reason: String,
    },
    PlanReady {
        plan: Plan,
    },
    ConfirmationRequested {
        title: String,
        context_line: String,
        /// Permission summary for this execution: sandbox, authorized folders, network, timeout and
        /// round limit.
        permissions: String,
        /// Workspace reference materials that were matched (at most 3).
        references: Vec<String>,
    },
    StateChanged {
        from: TaskState,
        to: TaskState,
    },
    /// Visible output delta from the executing agent; the message block is already persisted, the
    /// delta only drives the UI.
    AssistantDelta {
        turn: u32,
        delta: String,
        /// The corresponding `messages` row: the UI appends in place by id instead of rebuilding the
        /// whole message.
        #[serde(default)]
        message_id: Option<i64>,
    },
    /// Thinking delta (reasoning summary / reasoning text).
    ThinkingDelta {
        turn: u32,
        delta: String,
        #[serde(default)]
        message_id: Option<i64>,
    },
    StepStarted {
        turn: u32,
        /// Which command within this turn (the executing agent may call bash several times per turn).
        index: u32,
        script_preview: String,
        #[serde(default)]
        message_id: Option<i64>,
    },
    StepFinished {
        turn: u32,
        index: u32,
        exit_code: Option<i32>,
        timed_out: bool,
        phase: String,
        output_tail: String,
        #[serde(default)]
        message_id: Option<i64>,
    },
    Notice {
        message: String,
    },
    Completed {
        report: TaskReport,
        verdict: Option<Verdict>,
        artifacts: Vec<PathBuf>,
        /// The task's final reply; the tray shows it until the next task starts.
        #[serde(default)]
        reply: String,
    },
    Failed {
        error: String,
    },
    Cancelled,
}

impl RuntimeEventKind {
    pub fn tag(&self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Routed { .. } => "routed",
            Self::PlanReady { .. } => "plan_ready",
            Self::ConfirmationRequested { .. } => "confirmation_requested",
            Self::StateChanged { .. } => "state_changed",
            Self::AssistantDelta { .. } => "assistant_delta",
            Self::ThinkingDelta { .. } => "thinking_delta",
            Self::StepStarted { .. } => "step_started",
            Self::StepFinished { .. } => "step_finished",
            Self::Notice { .. } => "notice",
            Self::Completed { .. } => "completed",
            Self::Failed { .. } => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

impl RuntimeEvent {
    pub fn new(task_id: TaskId, kind: RuntimeEventKind) -> Self {
        Self {
            task_id,
            at: Utc::now(),
            kind,
        }
    }

    pub fn summary(&self) -> String {
        match &self.kind {
            RuntimeEventKind::Queued => "Task queued".into(),
            RuntimeEventKind::Routed {
                workspace_name,
                session_title,
                ..
            } => {
                format!("Routed to {workspace_name} / {session_title}")
            }
            RuntimeEventKind::PlanReady { plan } => format!("Plan: {}", plan.title),
            RuntimeEventKind::ConfirmationRequested { title, .. } => {
                format!("Awaiting confirmation: {title}")
            }
            RuntimeEventKind::StateChanged { to, .. } => to.label().into(),
            RuntimeEventKind::AssistantDelta { turn, delta, .. } => {
                format!("Turn {turn} output +{} chars", delta.chars().count())
            }
            RuntimeEventKind::ThinkingDelta { turn, delta, .. } => {
                format!("Turn {turn} thinking +{} chars", delta.chars().count())
            }
            RuntimeEventKind::StepStarted { turn, index, .. } => {
                format!("Turn {turn} · step {index} started")
            }
            RuntimeEventKind::StepFinished {
                turn,
                index,
                exit_code,
                ..
            } => format!("Turn {turn} · step {index} finished (exit={exit_code:?})"),
            RuntimeEventKind::Notice { message } => message.clone(),
            RuntimeEventKind::Completed { report, .. } => report.summary.clone(),
            RuntimeEventKind::Failed { error } => format!("Failed: {error}"),
            RuntimeEventKind::Cancelled => "Cancelled".into(),
        }
    }
}
