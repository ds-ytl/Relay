use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::context::EnvironmentSnapshot;
use crate::ids::{SessionId, TaskId, WorkspaceId};
use crate::plan::Plan;
use crate::state::TaskState;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workspace {
    pub id: WorkspaceId,
    pub name: String,
    pub root_dir: PathBuf,
    /// Working rules: constraints the model reads every time it enters this workspace.
    pub rules: String,
    /// Routing keywords; a hit on the current window/app name raises the match score.
    pub keywords: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: SessionId,
    pub workspace_id: WorkspaceId,
    pub title: String,
    pub summary: String,
    pub created_at: DateTime<Utc>,
    pub last_active_at: DateTime<Utc>,
    /// Environment summary from the previous trigger, used to resume within the same session.
    pub last_context: Option<String>,
}

/// Context usage: cumulative token consumption and the peak input of a single request.
///
/// `peak_input_tokens` reflects the request closest to the window limit, and the UI shows the
/// share from it; `compactions` records how many times history was compacted, which helps tell
/// whether a task has been running against the window limit for a long time.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// Input of the most recent request (any phase), for troubleshooting; the UI should not use it
    /// to show occupancy.
    pub last_input_tokens: u32,
    /// Largest input of a single request: the one closest to the window limit.
    pub peak_input_tokens: u32,
    /// Cumulative input tokens that hit the prefix cache.
    pub cached_tokens: u64,
    /// Cache tokens hit by the most recent request: pairs with `last_input_tokens`.
    pub last_cached_tokens: u32,
    /// Input size of the most recent "main execution request": the UI's "context occupancy" looks
    /// only at this.
    pub context_tokens: u32,
    /// Cache tokens hit by that main request; used with `context_tokens` to compute the hit rate.
    pub context_cached_tokens: u32,
    /// Number of times model summary compaction was triggered.
    pub compactions: u32,
}

impl ContextUsage {
    /// Record the usage of one model call; not reached when the server returns no usage.
    pub fn record(&mut self, input: u32, output: u32, cached: u32) {
        self.input_tokens += input as u64;
        self.output_tokens += output as u64;
        self.cached_tokens += cached as u64;
        self.last_input_tokens = input;
        self.last_cached_tokens = cached;
        self.peak_input_tokens = self.peak_input_tokens.max(input);
    }

    /// Record a "main execution request": besides totals and peak, refresh the current context size.
    ///
    /// Understanding / summary / verification are auxiliary calls and only use `record`: their
    /// requests are usually far smaller than an execution turn, and counting them into the context
    /// occupancy would make the badge drop by a chunk at the end, as if history had been lost.
    pub fn record_context(&mut self, input: u32, output: u32, cached: u32) {
        self.record(input, output, cached);
        self.context_tokens = input;
        self.context_cached_tokens = cached;
    }

    /// Merge another set of stats (for example usage that happened during understanding, before the
    /// task existed).
    pub fn merge(&mut self, other: Self) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.cached_tokens += other.cached_tokens;
        if other.last_input_tokens > 0 {
            self.last_input_tokens = other.last_input_tokens;
            self.last_cached_tokens = other.last_cached_tokens;
        }
        self.peak_input_tokens = self.peak_input_tokens.max(other.peak_input_tokens);
        // Take the largest context size seen: one small request never drags down the same execution.
        if other.context_tokens > self.context_tokens {
            self.context_tokens = other.context_tokens;
            self.context_cached_tokens = other.context_cached_tokens;
        }
        self.compactions += other.compactions;
    }

    /// Cache hit rate of the most recent request (0.0–1.0); returns 0 when there is no input.
    pub fn cache_ratio(&self) -> f32 {
        let (hit, total) = if self.context_tokens > 0 {
            (self.context_cached_tokens, self.context_tokens)
        } else {
            (self.last_cached_tokens, self.last_input_tokens)
        };
        if total == 0 {
            0.0
        } else {
            (hit as f32 / total as f32).clamp(0.0, 1.0)
        }
    }

    /// "Current context occupancy" shown in the UI: main execution request → peak → any request,
    /// degrading step by step.
    pub fn occupancy(&self) -> u32 {
        if self.context_tokens > 0 {
            self.context_tokens
        } else if self.peak_input_tokens > 0 {
            self.peak_input_tokens
        } else {
            self.last_input_tokens
        }
    }

    /// Share of the window taken by the current context occupancy; returns 0 when the window is
    /// unknown (0).
    pub fn ratio(&self, window: u32) -> f32 {
        if window == 0 {
            0.0
        } else {
            self.occupancy() as f32 / window as f32
        }
    }

    pub fn is_empty(&self) -> bool {
        self.input_tokens == 0 && self.output_tokens == 0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: TaskId,
    pub workspace_id: WorkspaceId,
    pub session_id: SessionId,
    pub title: String,
    pub instruction: String,
    pub supplement: Option<String>,
    pub context: EnvironmentSnapshot,
    /// Context usage stats: cumulative consumption, peak input and compaction count.
    pub usage: ContextUsage,
    pub plan: Option<Plan>,
    pub state: TaskState,
    pub report: Option<TaskReport>,
    pub delivery: DeliveryMode,
    pub error: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub finished_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryMode {
    /// Open the artifact file directly when done.
    OpenFile,
    /// Open the folder containing the artifact when done.
    OpenFolder,
    /// Notify only; open no windows.
    NotifyOnly,
}

impl DeliveryMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OpenFile => "open_file",
            Self::OpenFolder => "open_folder",
            Self::NotifyOnly => "notify_only",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::OpenFile => "Open the file when done",
            Self::OpenFolder => "Open the containing folder when done",
            Self::NotifyOnly => "Send a completion notification only",
        }
    }
}

impl std::str::FromStr for DeliveryMode {
    type Err = crate::error::CoreError;

    fn from_str(s: &str) -> crate::error::Result<Self> {
        match s.trim() {
            "open_file" => Ok(Self::OpenFile),
            "open_folder" => Ok(Self::OpenFolder),
            "notify_only" | "" => Ok(Self::NotifyOnly),
            other => Err(crate::error::CoreError::Invalid(format!(
                "unknown delivery mode: {other}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskStep {
    pub id: String,
    pub task_id: TaskId,
    pub turn: u32,
    pub script: String,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
    pub timed_out: bool,
    pub duration_ms: u64,
    pub phase: Option<ControlPhase>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    File,
    Folder,
    Log,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    pub id: String,
    pub task_id: TaskId,
    pub path: PathBuf,
    pub kind: ArtifactKind,
    pub size: u64,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TaskReport {
    pub verdict: Option<Verdict>,
    pub summary: String,
    pub checks: Vec<String>,
    pub delivered: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlPhase {
    /// Continue with the next execution turn.
    Continue,
    /// Wait for user confirmation before continuing.
    AwaitApproval,
    /// Pause deliberately and wait for the user to resume.
    Paused,
    /// The task is complete.
    Done,
    /// The task failed.
    Failed,
}

impl ControlPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Continue => "continue",
            Self::AwaitApproval => "await_approval",
            Self::Paused => "paused",
            Self::Done => "done",
            Self::Failed => "failed",
        }
    }
}

impl std::str::FromStr for ControlPhase {
    type Err = crate::error::CoreError;

    fn from_str(s: &str) -> crate::error::Result<Self> {
        match s {
            "continue" => Ok(Self::Continue),
            "await_approval" => Ok(Self::AwaitApproval),
            "paused" => Ok(Self::Paused),
            "done" => Ok(Self::Done),
            "failed" => Ok(Self::Failed),
            other => Err(crate::error::CoreError::Invalid(format!(
                "unknown control phase: {other}"
            ))),
        }
    }
}

/// Standalone state protocol that the runtime reads from `.relay/state.json` in the task directory
/// after each Bash turn.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ControlSignal {
    #[serde(default = "default_phase")]
    pub phase: ControlPhase,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub artifacts: Vec<String>,
    #[serde(default)]
    pub delivery: Option<DeliveryMode>,
}

fn default_phase() -> ControlPhase {
    ControlPhase::Continue
}

impl Default for ControlPhase {
    fn default() -> Self {
        Self::Continue
    }
}

#[cfg(test)]
mod usage_tests {
    use super::ContextUsage;

    #[test]
    fn auxiliary_calls_do_not_shrink_the_context_badge() {
        let mut usage = ContextUsage::default();
        // Execution turn: 20k input, 18k cached.
        usage.record_context(20_000, 500, 18_000);
        assert_eq!(usage.occupancy(), 20_000);
        // The verification turn's request is tiny: it must not drag the badge back.
        usage.record(1_200, 80, 1_000);
        assert_eq!(usage.occupancy(), 20_000);
        assert!((usage.cache_ratio() - 0.9).abs() < 1e-6);
        // Totals and peak are still recorded.
        assert_eq!(usage.input_tokens, 21_200);
        assert_eq!(usage.peak_input_tokens, 20_000);
    }

    #[test]
    fn merge_keeps_the_largest_context() {
        let mut exec = ContextUsage::default();
        exec.record_context(8_000, 100, 6_000);
        let mut verify = ContextUsage::default();
        verify.record(900, 40, 700);
        exec.merge(verify);
        assert_eq!(exec.occupancy(), 8_000);
        assert_eq!(exec.context_cached_tokens, 6_000);
    }

    #[test]
    fn occupancy_falls_back_to_peak_then_last() {
        let mut usage = ContextUsage::default();
        usage.record(3_000, 10, 1_500);
        assert_eq!(usage.occupancy(), 3_000);
        assert!((usage.cache_ratio() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn cache_ratio_is_clamped() {
        let mut usage = ContextUsage::default();
        usage.record_context(1_000, 10, 2_000);
        assert_eq!(usage.cache_ratio(), 1.0);
    }
}
