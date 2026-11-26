//! Runtime: task state machine and pipeline (understand → confirm → execute → verify → deliver).
//!
//! - Understand phase: the agent calls read-only tools over several turns (environment/screenshot/workspace/session/task/materials),
//!   then submits "workspace + session + plan" in one `submit_plan` call; there is no separate keyword router;
//! - Execute phase: each turn the agent first writes a progress reply, then calls tools (bash / read_file / list_files /
//!   read_image / computer_accessibility / computer_input) and reports phase plus artifacts through `submit_result`;
//!   on completion it first emits the final Markdown reply;
//! - Verify phase: the agent cross-checks the artifacts with read-only tools and submits a structured verdict via `submit_verification`.
//! The legacy `relay-state` / VERDICT text protocol is only a fallback; code blocks in reply text are never run as commands.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use chrono::Utc;
use relay_context::{CaptureOptions, ContextProvider};
use relay_core::{
    plan_text, user_text, Artifact, ArtifactKind, ChatMessageRecord, ContextUsage, ControlPhase,
    DeliveryMode, EnvironmentSnapshot, MessageRole, Plan, RuntimeEvent, RuntimeEventKind, Session,
    SessionId, Task, TaskId, TaskState, TaskStep, ToolMeta, Verdict, Workspace, WorkspaceId,
};
use relay_delivery::{self, DeliveryOutcome};
use relay_executor::{scan_artifacts, BashExecutor, ExecutionRequest, ExecutionResult};
use relay_openai::{CompletionRequest, InputItem, LanguageModel, ModelOutput, StreamDelta};
use relay_storage::Store;
use serde_json::Value;
use tokio::sync::{broadcast, oneshot};

use crate::agents::{self, TranscriptEntry};
use crate::budget::{self, Budget};
use crate::computer;
use crate::config::{resolve_within, RuntimeConfig};
use crate::error::{Result, RuntimeError};
use crate::model_history::{RecordedModel, CURRENT_TASK};
use crate::tools;
use crate::understanding::{
    SessionChoice, UnderstandEvent, UnderstandStage, UnderstandingAgent, UnderstandingInput,
    WorkspaceChoice,
};

const EVENT_BUFFER: usize = 256;
/// Output budget requested for an executor call: enough for a progress note plus one tool call,
/// with headroom for a reasoning model's thinking.
const EXECUTOR_OUTPUT_TOKENS: u32 = 8192;

/// How the pipeline starts: whether the plan already came out of the understanding phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PipelineMode {
    /// submit already ran understanding; the plan lives in task.plan, but the first confirmation is still required.
    Preplanned,
    /// Resume from a pause: reuse the plan and skip confirmation.
    Resume,
    /// Execute directly inside a session: skip understanding and verification; delivery happens as soon as execution finishes.
    Direct,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Confirmation {
    /// Confirm execution.
    Approve,
    /// Re-run understanding with revised requirements.
    Revise(String),
    /// Cancel the task.
    Cancel,
    /// Resume from a pause.
    Resume,
}

#[derive(Debug, Clone)]
pub struct TaskRequest {
    pub instruction: String,
    pub supplement: Option<String>,
    /// Workspace picked by the user in the dropdown.
    pub pinned_workspace: Option<WorkspaceId>,
    /// Session pinned when continuing a chat in the session window: with `direct` that session is reused for execution;
    /// otherwise the understanding phase still produces a plan, but its owner is no longer re-routed.
    pub pinned_session: Option<SessionId>,
    /// Skip the first confirmation (CLI / automation).
    pub auto_confirm: bool,
    /// Direct execution inside a session: skip understanding and verification; the plan comes from this turn's message only.
    pub direct: bool,
    /// Screenshots the user attached in the panel (cropped desktop captures with annotations).
    pub attachments: Vec<PathBuf>,
}

impl TaskRequest {
    pub fn new(instruction: impl Into<String>) -> Self {
        Self {
            instruction: instruction.into(),
            supplement: None,
            pinned_workspace: None,
            pinned_session: None,
            auto_confirm: false,
            direct: false,
            attachments: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TaskView {
    pub task: Task,
    pub workspace_name: String,
    pub session_title: String,
}

pub struct Runtime {
    inner: Arc<Inner>,
}

struct Inner {
    store: Arc<Store>,
    model: Arc<RecordedModel>,
    context: Arc<dyn ContextProvider>,
    config: RuntimeConfig,
    executor: BashExecutor,
    events: broadcast::Sender<RuntimeEvent>,
    /// Live trace of the understanding process (tool calls / decisions), rendered by the panel.
    understanding: broadcast::Sender<UnderstandEvent>,
    pending: Mutex<HashMap<TaskId, oneshot::Sender<Confirmation>>>,
    cancels: Mutex<HashMap<TaskId, Arc<AtomicBool>>>,
    auto_confirm: Mutex<HashSet<TaskId>>,
    /// Tasks within one session run serially; different sessions never block each other.
    session_gates: Mutex<HashMap<SessionId, Arc<tokio::sync::Mutex<()>>>>,
}

impl Clone for Runtime {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl Runtime {
    pub fn new(
        store: Arc<Store>,
        model: Arc<dyn LanguageModel>,
        context: Arc<dyn ContextProvider>,
        config: RuntimeConfig,
    ) -> Result<Self> {
        config.ensure_dirs()?;
        let helper = config.state_helper();
        write_state_helper(&helper)?;
        // Older versions stored only tasks and command records: backfill them into a message stream on startup so past sessions replay in the new UI.
        if let Err(err) = store.backfill_messages() {
            tracing::warn!(%err, "failed to backfill history session messages");
        }
        let executor = BashExecutor::new(config.policy.clone()).with_state_helper(&helper);
        let (events, _) = broadcast::channel(EVENT_BUFFER);
        let (understanding, _) = broadcast::channel(EVENT_BUFFER);
        let model = Arc::new(RecordedModel::new(model, store.clone()));
        let runtime = Self {
            inner: Arc::new(Inner {
                store,
                model,
                context,
                config,
                executor,
                events,
                understanding,
                pending: Mutex::new(HashMap::new()),
                cancels: Mutex::new(HashMap::new()),
                auto_confirm: Mutex::new(HashSet::new()),
                session_gates: Mutex::new(HashMap::new()),
            }),
        };
        runtime.recover_if_alone();
        Ok(runtime)
    }

    /// Interrupted tasks are recovered only after confirming that no other Relay instance is running.
    ///
    /// Otherwise even a quick CLI look at the task list would mark every GUI task awaiting confirmation as dead.
    fn recover_if_alone(&self) {
        let pid_path = self.inner.config.pid_path();
        if let Some(other) = read_pid(&pid_path) {
            if other != std::process::id() && process_alive(other) {
                tracing::info!(
                    pid = other,
                    "another Relay instance is running; skipping interrupted-task recovery"
                );
                return;
            }
        }
        if let Err(err) = std::fs::write(&pid_path, std::process::id().to_string()) {
            tracing::warn!(%err, "failed to write instance pid");
        }
        self.recover_interrupted();
    }

    /// Take over leftover tasks after a restart: paused ones stay resumable, the rest are marked as interrupted failures.
    fn recover_interrupted(&self) {
        let tasks = match self.inner.store.unfinished_tasks() {
            Ok(tasks) => tasks,
            Err(err) => {
                tracing::warn!(%err, "failed to read unfinished tasks; skipping crash recovery");
                return;
            }
        };
        for mut task in tasks {
            match task.state {
                TaskState::Paused => {
                    tracing::info!(task = %task.id, "task is paused and can be resumed from the management center");
                }
                _ => {
                    tracing::warn!(task = %task.id, state = task.state.as_str(), "failed to mark interrupted task");
                    self.fail_interrupted(
                        &mut task,
                        "Relay restarted, so the previous run was interrupted; resubmit it or resume from the pause point".to_string(),
                    );
                }
            }
        }
    }

    /// Used by startup recovery: only updates state and reports, with no desktop notifications to avoid a burst at startup.
    fn fail_interrupted(&self, task: &mut Task, reason: String) {
        if task.state.is_terminal() {
            return;
        }
        task.error = Some(reason.clone());
        task.report = Some(relay_core::TaskReport {
            verdict: Some(Verdict::Fail),
            summary: reason,
            checks: Vec::new(),
            delivered: Vec::new(),
        });
        task.finished_at = Some(Utc::now());
        task.updated_at = Utc::now();
        if self.transition(task, TaskState::Failed) {
            let _ = self.inner.store.save_task(task);
        }
    }

    /// Resume a paused task: re-acquire the session and continue the remaining turns.
    pub async fn resume(&self, task_id: TaskId) -> Result<()> {
        let mut task = self
            .inner
            .store
            .task(task_id)?
            .ok_or(RuntimeError::TaskNotFound(task_id))?;
        if task.state != TaskState::Paused {
            return Err(RuntimeError::Invalid(
                "only paused tasks can be resumed".into(),
            ));
        }
        let cancel = Arc::new(AtomicBool::new(false));
        self.inner.cancels.lock().unwrap().insert(task_id, cancel);
        self.emit(
            &task,
            RuntimeEventKind::Notice {
                message: "resume execution from the pause point".into(),
            },
        );

        let runtime = self.clone();
        let session_id = task.session_id;
        tokio::spawn(async move {
            let gate = runtime.session_gate(session_id);
            let guard = tokio::select! {
                guard = gate.lock() => Some(guard),
                _ = runtime.wait_cancel(task_id) => None,
            };
            match guard {
                Some(_guard) => {
                    if let Err(err) = runtime.pipeline(task_id, PipelineMode::Resume).await {
                        tracing::error!(%err, task = %task_id, "failed to resume task");
                        runtime.fail_task(task_id, err.to_string());
                    }
                }
                None => runtime.mark_cancelled_by_id(task_id),
            }
            runtime.release_session_gate(session_id, &gate);
        });
        let _ = &mut task;
        Ok(())
    }

    pub fn store(&self) -> &Store {
        &self.inner.store
    }

    pub fn config(&self) -> &RuntimeConfig {
        &self.inner.config
    }

    /// Instance pid file path (tests and diagnostics only).
    pub fn config_pid_path(&self) -> PathBuf {
        self.inner.config.pid_path()
    }

    /// Context window in tokens; 0 disables window accounting and compaction.
    pub fn context_window(&self) -> u32 {
        self.inner.config.context_window
    }

    /// Fraction of the window that triggers compaction; the UI uses it to color the usage badge.
    pub fn context_compact_ratio(&self) -> f32 {
        self.inner.config.context_compact_ratio
    }

    pub fn model_name(&self) -> &str {
        self.inner.model.name()
    }

    pub fn subscribe(&self) -> broadcast::Receiver<RuntimeEvent> {
        self.inner.events.subscribe()
    }

    /// Subscribe to the understanding process (tool-call trace).
    pub fn subscribe_understanding(&self) -> broadcast::Receiver<UnderstandEvent> {
        self.inner.understanding.subscribe()
    }

    /// Task entry point: capture environment → (understand →) create task → execute in the background.
    ///
    /// With `direct` and a pinned session the call goes straight into that session: understanding and verification are skipped and the plan comes from this turn's message;
    /// every other entry point gets workspace, session and plan from the understanding phase; there is no separate routing phase.
    pub async fn submit(&self, request: TaskRequest) -> Result<TaskId> {
        let task_id = TaskId::new();
        CURRENT_TASK
            .scope(task_id, self.submit_task(task_id, request))
            .await
    }

    async fn submit_task(&self, task_id: TaskId, request: TaskRequest) -> Result<TaskId> {
        // Blank instructions are allowed: when the user only presses "start understanding", the instruction becomes a placeholder
        // and the understanding agent infers the goal from what is on screen (see BLANK_SCENE).
        let instruction = crate::understanding::normalize_instruction(&request.instruction);

        // Continuing a chat: the user pinned a session and asked explicitly for "direct execution",
        // so understanding (routing/planning) and verification are skipped and the request goes straight to the executor agent.
        let direct = request.direct && request.pinned_session.is_some();
        let mut snapshot = if direct {
            EnvironmentSnapshot::default()
        } else {
            self.capture_context().await?
        };
        // Attachments are real files on disk; list them so read_image can reach them later in the task.
        for path in &request.attachments {
            if path.exists() && !snapshot.screenshots.contains(path) {
                snapshot.screenshots.push(path.clone());
            }
        }

        let (plan, workspace, session, reason, confidence, usage) = if direct {
            let session_id = request
                .pinned_session
                .expect("direct path already validated the session");
            let session = self.inner.store.session(session_id)?.ok_or_else(|| {
                RuntimeError::Invalid(format!("session does not exist: {session_id}"))
            })?;
            let workspace = self
                .inner
                .store
                .workspace(session.workspace_id)?
                .ok_or_else(|| {
                    RuntimeError::Invalid("the session's workspace does not exist".into())
                })?;
            let plan = direct_plan(&instruction);
            (
                plan,
                workspace,
                session,
                "direct execution in session".to_string(),
                1.0,
                ContextUsage::default(),
            )
        } else {
            let mut understanding = self
                .understand_task(
                    &instruction,
                    request.supplement.as_deref(),
                    &snapshot,
                    &request.attachments,
                    request.pinned_workspace,
                )
                .await?;

            // Continuing from the session window: the user pinned the session, so understanding must not re-route it.
            if let Some(session_id) = request.pinned_session {
                if let Ok(Some(session)) = self.inner.store.session(session_id) {
                    understanding.workspace =
                        WorkspaceChoice::Reuse(session.workspace_id.to_string());
                    understanding.session = SessionChoice::Reuse(session.id.to_string());
                    understanding.confidence = 1.0;
                    understanding.reason = "keep the current session".into();
                }
            }

            let plan = understanding.plan.clone();
            let workspace = self.ensure_workspace(&understanding.workspace, &snapshot)?;
            // With a blank instruction the placeholder text carries no meaning, so the session title falls back to the plan title.
            let title_source = if crate::understanding::is_blank_instruction(&instruction) {
                plan.title.clone()
            } else {
                instruction.clone()
            };
            let session =
                self.ensure_session(&understanding.session, &workspace, &title_source, &snapshot)?;
            (
                plan,
                workspace,
                session,
                understanding.reason.clone(),
                understanding.confidence,
                understanding.usage,
            )
        };

        let now = Utc::now();
        let task = Task {
            id: task_id,
            workspace_id: workspace.id,
            session_id: session.id,
            title: if plan.title.trim().is_empty() {
                short_title(&instruction)
            } else {
                plan.title.clone()
            },
            instruction: instruction.clone(),
            supplement: request.supplement.clone(),
            context: snapshot,
            usage,
            plan: Some(plan.clone()),
            state: TaskState::Pending,
            report: None,
            delivery: plan.delivery,
            error: None,
            created_at: now,
            updated_at: now,
            started_at: None,
            finished_at: None,
        };
        self.inner.store.save_task(&task)?;
        // Session messages are persisted from the very first moment: history replay and streaming read the same data.
        self.record_message(
            &task,
            0,
            MessageRole::User,
            &user_text(&task.instruction, task.supplement.as_deref()),
            None,
        );
        self.record_message(&task, 0, MessageRole::Plan, &plan_text(&plan), None);
        self.emit(
            &task,
            RuntimeEventKind::Routed {
                workspace_id: workspace.id,
                workspace_name: workspace.name.clone(),
                session_id: session.id,
                session_title: session.title.clone(),
                confidence,
                reason: reason.clone(),
            },
        );
        self.emit(&task, RuntimeEventKind::PlanReady { plan });
        self.emit(&task, RuntimeEventKind::Queued);
        tracing::info!(
            task = %task.id,
            session = %session.id,
            workspace = %workspace.name,
            confidence,
            reason = %reason,
            "task created"
        );

        if request.auto_confirm || self.inner.config.auto_confirm {
            self.inner.auto_confirm.lock().unwrap().insert(task.id);
        }
        let cancel = Arc::new(AtomicBool::new(false));
        self.inner.cancels.lock().unwrap().insert(task.id, cancel);

        let runtime = self.clone();
        let task_id = task.id;
        let session_id = task.session_id;
        let pipeline_mode = if direct {
            PipelineMode::Direct
        } else {
            PipelineMode::Preplanned
        };
        tokio::spawn(async move {
            let gate = runtime.session_gate(session_id);
            let guard = tokio::select! {
                guard = gate.lock() => Some(guard),
                _ = runtime.wait_cancel(task_id) => None,
            };
            match guard {
                Some(_guard) => {
                    if let Err(err) = runtime.pipeline(task_id, pipeline_mode).await {
                        tracing::error!(%err, task = %task_id, "task execution failed");
                        runtime.fail_task(task_id, err.to_string());
                    }
                }
                None => runtime.mark_cancelled_by_id(task_id),
            }
            runtime.release_session_gate(session_id, &gate);
        });

        Ok(task.id)
    }

    /// The user makes a choice on the confirmation card.
    /// The user rewrites the understanding result on the confirmation card: title, summary, steps and delivery mode are stored as edited.
    ///
    /// Only allowed while awaiting confirmation; swapping the plan of a running task would drift its meaning.
    pub fn update_plan(&self, task_id: TaskId, plan: Plan) -> Result<()> {
        let mut task = self
            .inner
            .store
            .task(task_id)?
            .ok_or(RuntimeError::TaskNotFound(task_id))?;
        if task.state != TaskState::AwaitingConfirmation {
            return Err(RuntimeError::Invalid(
                "only tasks awaiting confirmation can have their plan edited".into(),
            ));
        }
        if plan.title.trim().is_empty() {
            return Err(RuntimeError::Invalid("plan title cannot be empty".into()));
        }
        let steps: Vec<String> = plan
            .steps
            .iter()
            .map(|step| step.trim().to_string())
            .filter(|step| !step.is_empty())
            .collect();
        if steps.is_empty() {
            return Err(RuntimeError::Invalid(
                "at least one execution step must remain".into(),
            ));
        }
        let mut plan = plan;
        plan.title = plan.title.trim().to_string();
        plan.summary = plan.summary.trim().to_string();
        plan.steps = steps;
        self.apply_plan(&mut task, &plan)
    }

    pub fn confirm(&self, task_id: TaskId, decision: Confirmation) -> Result<()> {
        let tx = self
            .inner
            .pending
            .lock()
            .unwrap()
            .remove(&task_id)
            .ok_or(RuntimeError::NoPendingConfirmation(task_id))?;
        tx.send(decision)
            .map_err(|_| RuntimeError::NoPendingConfirmation(task_id))
    }

    /// Hand the task's cancel flag to the executor so a stop can kill the running process group directly.
    fn cancel_flag(&self, task_id: TaskId) -> Option<Arc<AtomicBool>> {
        self.inner
            .cancels
            .lock()
            .unwrap()
            .get(&task_id)
            .map(Arc::clone)
    }

    /// Cancel a task: write Cancelled immediately and signal the waiting pipeline to exit.
    pub fn cancel(&self, task_id: TaskId) -> Result<()> {
        if let Some(flag) = self.inner.cancels.lock().unwrap().get(&task_id) {
            flag.store(true, Ordering::SeqCst);
        }
        if let Some(tx) = self.inner.pending.lock().unwrap().remove(&task_id) {
            let _ = tx.send(Confirmation::Cancel);
        }
        if let Some(mut task) = self.inner.store.task(task_id)? {
            if !task.state.is_terminal() {
                if let Ok(state) = task.state.transition_to(TaskState::Cancelled) {
                    self.record_message(&task, 0, MessageRole::Notice, "task cancelled", None);
                    let from = task.state;
                    task.state = state;
                    task.updated_at = Utc::now();
                    task.finished_at = Some(Utc::now());
                    self.inner.store.save_task(&task)?;
                    self.emit(&task, RuntimeEventKind::StateChanged { from, to: state });
                    self.emit(&task, RuntimeEventKind::Cancelled);
                }
            }
        }
        Ok(())
    }

    pub fn task(&self, task_id: TaskId) -> Result<Option<Task>> {
        Ok(self.inner.store.task(task_id)?)
    }

    /// Most recent task waiting for user confirmation.
    ///
    /// The panel collapses during understanding/execution and reopens for confirmation; a reopened panel uses this call
    /// to restore the confirmation card instead of showing an empty input box.
    pub fn awaiting_confirmation(&self) -> Result<Option<Task>> {
        let mut tasks = self.inner.store.unfinished_tasks()?;
        tasks.retain(|task| task.state == TaskState::AwaitingConfirmation);
        tasks.sort_by_key(|task| std::cmp::Reverse(task.created_at));
        Ok(tasks.into_iter().next())
    }

    pub fn recent_tasks(&self, limit: u32) -> Result<Vec<TaskView>> {
        let tasks = self.inner.store.recent_tasks(limit)?;
        Ok(tasks.into_iter().map(|task| self.view(task)).collect())
    }

    pub fn tasks_for_session(&self, session_id: SessionId, limit: u32) -> Result<Vec<TaskView>> {
        let tasks = self.inner.store.tasks_by_session(session_id, limit)?;
        Ok(tasks.into_iter().map(|task| self.view(task)).collect())
    }

    /// Session message stream in storage order, i.e. the real sequence of user message / reasoning / tool / reply.
    pub fn messages(&self, session_id: SessionId) -> Result<Vec<ChatMessageRecord>> {
        Ok(self.inner.store.messages_for_session(session_id)?)
    }

    /// Single message: the UI fetches the latest content by id when a tool step starts or ends.
    pub fn message(&self, id: i64) -> Result<Option<ChatMessageRecord>> {
        Ok(self.inner.store.message(id)?)
    }

    pub fn workspaces(&self) -> Result<Vec<Workspace>> {
        Ok(self.inner.store.workspaces()?)
    }

    pub fn sessions(&self, workspace_id: WorkspaceId, limit: u32) -> Result<Vec<Session>> {
        Ok(self.inner.store.sessions(workspace_id, limit)?)
    }

    /// Execution log: state events plus per-turn script output, shown in the management center.
    pub fn events(&self, task_id: TaskId, limit: u32) -> Result<Vec<RuntimeEvent>> {
        Ok(self.inner.store.events_for(task_id, limit)?)
    }

    /// Human-readable log for the management center: scripts and output per turn first, recent events after.
    pub fn task_log(&self, task_id: TaskId) -> Result<Vec<String>> {
        let mut lines = Vec::new();
        let steps = self.inner.store.steps(task_id)?;
        if steps.is_empty() {
            lines.push(
                "no execution records (the task may still be understanding or was cancelled)."
                    .into(),
            );
        } else {
            for step in steps {
                lines.push(format!(
                    "turn {} · {} · exit={:?} · {}ms",
                    step.turn,
                    step.phase
                        .map(|phase| phase.as_str().to_string())
                        .unwrap_or_else(|| "-".into()),
                    step.exit_code,
                    step.duration_ms
                ));
                let script = step.script.lines().next().unwrap_or_default().trim();
                if !script.is_empty() {
                    lines.push(format!("  $ {script}"));
                }
                let mut output = step.stdout.trim_end().to_string();
                if !step.stderr.trim().is_empty() {
                    output.push_str(&format!("\n[stderr] {}", step.stderr.trim_end()));
                }
                for line in output
                    .lines()
                    .rev()
                    .take(6)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                {
                    lines.push(format!("    {line}"));
                }
                lines.push(String::new());
            }
        }
        let events = self.inner.store.events_for(task_id, 20)?;
        if !events.is_empty() {
            lines.push("events:".into());
            for event in events.into_iter().rev() {
                lines.push(format!(
                    "  {} · {}",
                    event.at.format("%H:%M:%S"),
                    event.summary()
                ));
            }
        }
        Ok(lines)
    }

    pub fn steps(&self, task_id: TaskId) -> Result<Vec<TaskStep>> {
        Ok(self.inner.store.steps(task_id)?)
    }

    pub fn artifacts(&self, task_id: TaskId) -> Result<Vec<Artifact>> {
        Ok(self.inner.store.artifacts(task_id)?)
    }

    fn view(&self, task: Task) -> TaskView {
        let workspace_name = self
            .inner
            .store
            .workspace(task.workspace_id)
            .ok()
            .flatten()
            .map(|w| w.name)
            .unwrap_or_else(|| "unknown workspace".into());
        let session_title = self
            .inner
            .store
            .session(task.session_id)
            .ok()
            .flatten()
            .map(|s| s.title)
            .unwrap_or_else(|| "unknown session".into());
        TaskView {
            task,
            workspace_name,
            session_title,
        }
    }

    async fn capture_context(&self) -> Result<EnvironmentSnapshot> {
        let mut options = CaptureOptions {
            authorized_dirs: self.inner.config.policy.allowed_dirs.clone(),
            include_clipboard: self.inner.config.include_clipboard,
            include_screenshot: self.inner.config.include_screenshot,
            screenshot_dir: Some(self.inner.config.data_dir.join("context")),
            clipboard_max_chars: 1500,
        };
        for workspace in self.inner.store.workspaces()? {
            if !options.authorized_dirs.contains(&workspace.root_dir) {
                options.authorized_dirs.push(workspace.root_dir);
            }
        }
        let provider = self.inner.context.clone();
        // On first run the OS may be waiting for user authorization (accessibility / automation), so submission must not hang forever.
        let captured = tokio::time::timeout(
            Duration::from_secs(6),
            tokio::task::spawn_blocking(move || provider.capture(&options)),
        )
        .await;
        let mut snapshot = match captured {
            Ok(Ok(Ok(snapshot))) => snapshot,
            Ok(Ok(Err(err))) => {
                tracing::warn!(%err, "context capture failed; continuing with empty context");
                EnvironmentSnapshot::default()
            }
            Ok(Err(err)) => {
                tracing::warn!(%err, "context capture task failed; continuing with empty context");
                EnvironmentSnapshot::default()
            }
            Err(_) => {
                tracing::warn!("context capture timed out (possibly waiting for OS authorization); continuing with empty context");
                EnvironmentSnapshot::default()
            }
        };
        // The panel steals focus: if the snapshot's frontmost app is Relay or a system shell, swap in the user's real window,
        // otherwise blank-instruction inference would only ever see ourselves.
        if snapshot.app_name.trim().is_empty() || relay_platform::is_shell_app(&snapshot.app_name) {
            // Prefer the live window, then fall back to the window remembered before the panel opened: at the moment the panel steals focus
            // the system may report only Dock or relay itself.
            let window =
                relay_platform::user_front_window().or_else(relay_platform::last_user_window);
            if let Some(window) = window {
                snapshot.app_name = window.app;
                snapshot.window_title = window.title;
            }
        }
        Ok(snapshot)
    }

    fn ensure_workspace(
        &self,
        choice: &WorkspaceChoice,
        snapshot: &EnvironmentSnapshot,
    ) -> Result<Workspace> {
        if let WorkspaceChoice::Reuse(id) = choice {
            if let Ok(parsed) = id.parse::<WorkspaceId>() {
                if let Some(workspace) = self.inner.store.workspace(parsed)? {
                    return self.sync_workspace(&workspace);
                }
            }
            tracing::warn!(workspace_id = %id, "workspace from understanding does not exist; creating a new one");
        }

        let (name, keywords) = match choice {
            WorkspaceChoice::Create { name, keywords } => {
                let name = if name.trim().is_empty() {
                    derive_workspace_name(snapshot)
                } else {
                    name.trim().to_string()
                };
                let mut keywords = keywords.clone();
                if keywords.is_empty() {
                    keywords = relay_router::tokenize(&name)
                        .into_iter()
                        .filter(|token| token.chars().count() >= 2)
                        .take(6)
                        .collect();
                }
                (name, keywords)
            }
            WorkspaceChoice::Reuse(_) => (derive_workspace_name(snapshot), Vec::new()),
        };

        // Workspace names are unique: reuse a match by name before creating one to avoid a UNIQUE constraint failure.
        if let Some(workspace) = self.inner.store.workspace_by_name(&name)? {
            return self.sync_workspace(&workspace);
        }

        let id = WorkspaceId::new();
        let root_dir = self.inner.config.workspace_root(&name, id);
        std::fs::create_dir_all(&root_dir)?;
        for dir in ["materials", "outputs"] {
            std::fs::create_dir_all(root_dir.join(dir))?;
        }
        let rules_path = root_dir.join("rules.md");
        let rules = format!(
            "# Workspace Rules\n\n- Trigger: {}\n- Default delivery: open the artifact when done\n- Work on copies; never overwrite source files\n- Put source material in materials/ and deliverables in outputs/\n",
            snapshot.summary_line()
        );
        std::fs::write(&rules_path, &rules)?;
        let now = Utc::now();
        let workspace = Workspace {
            id,
            name: name.clone(),
            root_dir,
            rules,
            keywords,
            created_at: now,
            updated_at: now,
        };
        if let Err(err) = self.inner.store.save_workspace(&workspace) {
            // A concurrent submission may have just created the same workspace; reuse it.
            if let Some(existing) = self.inner.store.workspace_by_name(&workspace.name)? {
                return self.sync_workspace(&existing);
            }
            return Err(err.into());
        }
        tracing::info!(workspace = %workspace.name, root = %workspace.root_dir.display(), "create workspace");
        Ok(workspace)
    }

    /// rules.md wins for a workspace: user edits take effect on the next task.
    fn sync_workspace(&self, workspace: &Workspace) -> Result<Workspace> {
        migrate_legacy_layout(&workspace.root_dir);
        let mut workspace = workspace.clone();
        for name in ["materials", "outputs"] {
            let _ = std::fs::create_dir_all(workspace.root_dir.join(name));
        }
        let rules_path = workspace.root_dir.join("rules.md");
        let file_rules = std::fs::read_to_string(&rules_path)
            .ok()
            .map(|text| text.trim().to_string())
            .filter(|text| !text.is_empty());
        if let Some(rules) = file_rules {
            if rules != workspace.rules {
                workspace.rules = rules;
                workspace.updated_at = Utc::now();
                let _ = self.inner.store.save_workspace(&workspace);
            }
        }
        Ok(workspace)
    }

    fn ensure_session(
        &self,
        choice: &SessionChoice,
        workspace: &Workspace,
        instruction: &str,
        snapshot: &EnvironmentSnapshot,
    ) -> Result<Session> {
        if let SessionChoice::Reuse(id) = choice {
            if let Ok(parsed) = id.parse::<SessionId>() {
                if let Some(session) = self.inner.store.session(parsed)? {
                    if session.workspace_id == workspace.id {
                        return Ok(session);
                    }
                }
            }
            tracing::warn!(session_id = %id, "session missing or owned by another workspace; creating a new one");
        }

        let title = match choice {
            SessionChoice::Create { title } if !title.trim().is_empty() => title.trim().to_string(),
            _ => relay_router::derive_session_title(instruction),
        };
        let now = Utc::now();
        let session = Session {
            id: SessionId::new(),
            workspace_id: workspace.id,
            title,
            summary: short_title(instruction),
            created_at: now,
            last_active_at: now,
            last_context: Some(snapshot.summary_line()),
        };
        self.inner.store.save_session(&session)?;
        Ok(session)
    }

    async fn pipeline(&self, task_id: TaskId, mode: PipelineMode) -> Result<()> {
        CURRENT_TASK
            .scope(task_id, self.run_pipeline(task_id, mode))
            .await
    }

    async fn run_pipeline(&self, task_id: TaskId, mode: PipelineMode) -> Result<()> {
        let resume = mode == PipelineMode::Resume;
        let skip_verify = mode == PipelineMode::Direct;
        let mut task = self
            .inner
            .store
            .task(task_id)?
            .ok_or(RuntimeError::TaskNotFound(task_id))?;
        let workspace = self
            .inner
            .store
            .workspace(task.workspace_id)?
            .ok_or_else(|| RuntimeError::Invalid("the task's workspace does not exist".into()))?;

        let task_dir = self.inner.config.task_dir(task.id);
        std::fs::create_dir_all(task_dir.join(".relay"))?;
        std::fs::create_dir_all(&workspace.root_dir)?;
        let cwd = workspace.root_dir.clone();

        task.started_at.get_or_insert_with(Utc::now);

        let mut plan = match task.plan.clone() {
            Some(plan) => plan,
            None => {
                // Fallback: legacy tasks without a plan (e.g. inserted by hand) get one understanding pass.
                if !self.transition(&mut task, TaskState::Understanding) {
                    return Ok(());
                }
                self.save(&task)?;
                let instruction = task.instruction.clone();
                let supplement = task.supplement.clone();
                let understanding = match self
                    .understand_task(
                        &instruction,
                        supplement.as_deref(),
                        &task.context,
                        &[],
                        Some(task.workspace_id),
                    )
                    .await
                {
                    Ok(understanding) => understanding,
                    Err(err) => {
                        self.fail_with(&mut task, format!("task understanding failed: {err}"));
                        return Ok(());
                    }
                };
                task.usage.merge(understanding.usage);
                self.save(&task)?;
                self.apply_plan(&mut task, &understanding.plan)?;
                understanding.plan
            }
        };

        // When submit already produced the plan, the state machine still passes through Understanding before confirmation.
        if matches!(mode, PipelineMode::Preplanned | PipelineMode::Direct) {
            if !self.transition(&mut task, TaskState::Understanding) {
                return Ok(());
            }
            self.save(&task)?;
        }

        if !resume {
            loop {
                if self.is_cancelled(task.id) {
                    self.mark_cancelled(&mut task);
                    return Ok(());
                }
                if !self.needs_confirmation(task.id) {
                    break;
                }
                if !self.transition(&mut task, TaskState::AwaitingConfirmation) {
                    return Ok(());
                }
                match self.ask_confirmation(&task, &plan, &workspace).await {
                    Confirmation::Approve | Confirmation::Resume => break,
                    Confirmation::Cancel => {
                        self.mark_cancelled(&mut task);
                        return Ok(());
                    }
                    Confirmation::Revise(supplement) => {
                        task.supplement = Some(supplement);
                        if !self.transition(&mut task, TaskState::Understanding) {
                            return Ok(());
                        }
                        let instruction = task.instruction.clone();
                        let supplement = task.supplement.clone();
                        let understanding = match self
                            .understand_task(
                                &instruction,
                                supplement.as_deref(),
                                &task.context,
                                &[],
                                Some(task.workspace_id),
                            )
                            .await
                        {
                            Ok(understanding) => understanding,
                            Err(err) => {
                                self.fail_with(
                                    &mut task,
                                    format!("re-running task understanding failed: {err}"),
                                );
                                return Ok(());
                            }
                        };
                        self.apply_plan(&mut task, &understanding.plan)?;
                        plan = understanding.plan;
                    }
                }
            }
        }

        // The user may have edited the plan on the confirmation card, so execution must use the latest stored copy.
        // The in-memory task must follow, or the final save would overwrite the edited version.
        if let Some(latest) = self.inner.store.task(task.id)? {
            if let Some(edited) = latest.plan {
                task.title = edited.title.clone();
                task.delivery = edited.delivery;
                task.plan = Some(edited.clone());
                plan = edited;
            }
        }

        if !self.transition(&mut task, TaskState::Executing) {
            return Ok(());
        }

        let since = SystemTime::now();
        let mut transcript: Vec<TranscriptEntry> = if resume {
            self.transcript_from_steps(task.id)?
        } else {
            Vec::new()
        };
        let mut artifacts: Vec<PathBuf> = if resume {
            self.inner
                .store
                .artifacts(task.id)?
                .into_iter()
                .map(|artifact| artifact.path)
                .collect()
        } else {
            Vec::new()
        };
        let mut last_output = String::new();
        let mut failure: Option<String> = None;
        let mut completed = false;

        // Session-level history: earlier turns in this session already did part of the work, so the new task continues instead of starting over.
        let (mut dialogue, previous_task) = self.dialogue_history(task.session_id, task.id)?;
        let mut dialogue_compacted = false;
        let transcript_budget = self.inner.config.transcript_budget();

        let mut exec_tools = tools::executor_tools();
        // No key, no tools API: keep search / fetch out of the tool list entirely.
        if self.inner.config.web_tools.is_none() {
            exec_tools.retain(|tool| tool.name != "search" && tool.name != "fetch");
        }
        let exec_tool_names: Vec<String> =
            exec_tools.iter().map(|tool| tool.name.clone()).collect();
        let exec_allowed = self.inner.config.policy.allowed_dirs.clone();
        let budget = Budget::new(
            self.inner.config.context_window,
            self.inner.config.context_compact_ratio,
        );

        let overhead = budget::request_overhead(agents::EXECUTOR_PROMPT, &exec_tools);
        let mut rolling_summary = String::new();
        let mut observed_input = 0;

        let provider = &self.inner.model.provider;
        let store = &self.inner.store;
        let source = if resume { Some(task.id) } else { previous_task };
        let header = CompletionRequest::text(agents::EXECUTOR_PROMPT, "")
            .with_tools(exec_tools.clone())
            .responses_payload(self.inner.model.name())?;
        // Highest turn already executed, used when no checkpoint names the turn.
        let last_turn = transcript.iter().map(|entry| entry.turn).max().unwrap_or(0);
        let mut checkpoint = match source {
            Some(id) => store.executor_checkpoint(id, provider)?,
            None => None,
        };
        // A crash can land between dispatch and the checkpoint write. The recorded request is
        // still exactly what the model saw, so replay it and answer whatever calls it left open;
        // re-rendering from steps would send a different, shorter context and lose every cached
        // token. Only when nothing was recorded does the task fall back to the rebuilt transcript.
        if let Some(id) = source.filter(|_| checkpoint.is_none()) {
            // Only an executor request is replayable: understanding and verification use a
            // different prompt and tool set, so their headers do not match.
            let replayable = store
                .recent_model_requests(id, provider, 8)?
                .into_iter()
                .find(|(_, stored)| {
                    ["model", "tools", "instructions"]
                        .iter()
                        .all(|key| stored[key] == header[key])
                });
            if let Some((request_id, _)) = replayable {
                let payload = store.model_request(request_id)?;
                let open = agents::interrupted_outputs(
                    payload["input"]
                        .as_array()
                        .map(Vec::as_slice)
                        .unwrap_or(&[]),
                );
                let mut suffix = open;
                suffix.push(agents::interrupted_note());
                tracing::info!(
                    task = %id,
                    request = request_id,
                    "replaying the recorded request after a restart: its checkpoint was never written"
                );
                checkpoint = Some(relay_storage::ExecutorCheckpoint {
                    turn: last_turn,
                    payload,
                    suffix,
                });
            }
        }
        let first_turn = if resume {
            checkpoint.as_ref().map_or(1, |c| c.turn.saturating_add(1))
        } else {
            1
        };
        for turn in first_turn..=self.inner.config.max_rounds {
            if self.is_cancelled(task.id) {
                self.mark_cancelled(&mut task);
                return Ok(());
            }

            let (omitted, _) = agents::transcript_blocks(&transcript, transcript_budget);
            if budget.enabled() && omitted > 0 {
                let mut history = vec![InputItem::Text(rolling_summary.clone())];
                for entry in &transcript[..omitted] {
                    history.push(InputItem::Text(format!(
                        "turn {} ({})\n{}",
                        entry.turn, entry.phase, entry.reply
                    )));
                    for step in &entry.steps {
                        history.push(InputItem::Text(format!(
                            "{} {}\nexit={:?}\n{}",
                            step.tool, step.input, step.exit_code, step.output
                        )));
                    }
                }
                let compacted =
                    budget::summarize_for_window(self.inner.model.as_ref(), &history, budget)
                        .await?;
                rolling_summary = compacted.record(&mut task.usage);
                transcript.drain(..omitted);
            }
            let (mut items, prefix_items, tail) = agents::executor_input(
                &task,
                &plan,
                &task_dir,
                turn,
                self.inner.config.max_rounds,
                &dialogue,
                &transcript,
                &artifacts,
                transcript_budget,
            )
            .into_items();

            if !rolling_summary.is_empty() {
                items.insert(
                    prefix_items,
                    InputItem::Text(format!("{}\n{}", budget::SUMMARY_HEADING, rolling_summary)),
                );
            }
            // Reopen and live continuation follow the same path. Prior messages retain their
            // exact wire representation and position, including the original user-message boundary.
            if turn != first_turn {
                checkpoint = store.executor_checkpoint(task.id, provider)?;
            }
            let mut archived = Vec::new();
            if let Some(mut saved) = checkpoint.take() {
                if ["model", "tools", "instructions"]
                    .iter()
                    .all(|key| saved.payload[key] == header[key])
                {
                    let input = saved.payload["input"].as_array_mut().ok_or_else(|| {
                        RuntimeError::Invalid("invalid stored executor context".into())
                    })?;
                    archived.extend(input.drain(..).chain(saved.suffix).map(InputItem::Raw));
                } else {
                    // A different model, tool set or prompt invalidates the cached prefix anyway,
                    // so the exchange cannot be replayed. Say so instead of paying blindly.
                    tracing::warn!(
                        turn,
                        "stored executor context belongs to another model or prompt; the turn starts a new prefix"
                    );
                }
            }
            // The checkpoint already contains the prior exchanges. Append only new task
            // information and the current progress, without duplicating the condensed transcript.
            let mut append_from = 0;
            if !archived.is_empty() {
                for item in &items[..prefix_items] {
                    let InputItem::Text(text) = item else {
                        continue;
                    };
                    let already_sent = archived.iter().any(|item| match item {
                        InputItem::Raw(message) if message["role"] == "user" => {
                            message["content"].as_array().is_some_and(|parts| {
                                parts.iter().any(|p| p["text"].as_str() == Some(text))
                            })
                        }
                        _ => false,
                    });
                    if !already_sent {
                        archived.push(item.clone());
                    }
                }
                archived.push(InputItem::Text(tail.clone()));
                append_from = items.len();
            }
            let mut archived_cost = budget::estimate(&archived);
            // One turn = one "multiple tool calls → submit_result" cycle of the executor agent.
            // Segments follow cache-prefix order: dialogue history → task brief → earlier transcript → current-turn progress.
            let mut round_steps: Vec<agents::TranscriptStep> = Vec::new();
            let mut round_reply = String::new();
            let mut round_control: Option<relay_core::ControlSignal> = None;
            let mut exec_seq = 0u32;
            // When the current model call is rejected for exceeding the window, shrink once and retry.
            let mut retry_output_limit = None;
            // Output room learned from a truncated call: kept for the rest of the turn so the model
            // does not have to burn the whole budget twice.
            let mut output_floor = 0u32;
            let mut truncated_retries = 0u8;

            loop {
                if self.is_cancelled(task.id) {
                    self.mark_cancelled(&mut task);
                    return Ok(());
                }
                if exec_seq >= self.inner.config.max_tool_steps_per_round {
                    tracing::warn!(
                        turn,
                        exec_seq,
                        "tool-call limit reached in this turn; ending the turn"
                    );
                    break;
                }

                if !archived.is_empty()
                    && (retry_output_limit.is_some()
                        || budget.needs_summary(
                            overhead
                                .saturating_add(archived_cost)
                                .saturating_add(budget::estimate(&items[append_from..])),
                            8192,
                        ))
                {
                    // Start a new prefix only when the previous one no longer fits.
                    let compacted =
                        budget::summarize_for_window(self.inner.model.as_ref(), &archived, budget)
                            .await?;
                    rolling_summary = compacted.record(&mut task.usage);
                    items.insert(
                        prefix_items,
                        InputItem::Text(format!(
                            "{}\n{}",
                            budget::SUMMARY_HEADING,
                            rolling_summary
                        )),
                    );
                    archived.clear();
                    append_from = 0;
                    archived_cost = 0;
                    observed_input = 0;
                }
                let estimated =
                    (overhead + archived_cost + budget::estimate(&items[append_from..]))
                        .max(observed_input);
                // Even before any tool call, the first turn can already carry an oversized dialogue or transcript.
                if archived.is_empty()
                    && (budget.needs_summary(estimated, 8192) || retry_output_limit.is_some())
                {
                    let has_history = !transcript.is_empty()
                        || items.iter().any(|item| matches!(item, InputItem::Raw(_)));
                    if has_history {
                        match budget::summarize_for_window(
                            self.inner.model.as_ref(),
                            &items[prefix_items..],
                            budget,
                        )
                        .await
                        {
                            Ok(compacted) => {
                                rolling_summary = compacted.record(&mut task.usage);
                                items = budget::compacted_items(
                                    &items,
                                    rolling_summary.clone(),
                                    prefix_items,
                                );
                                items.push(InputItem::Text(tail.clone()));
                                transcript.clear();
                                round_steps.clear();
                                observed_input = 0;
                            }
                            Err(err) if retry_output_limit.is_some() => return Err(err.into()),
                            Err(err) => {
                                tracing::warn!(%err, turn, "failed to summarize execution history")
                            }
                        }
                    }
                    // The cache prefix usually keeps raw text. When the history itself is too long, keep the confirmed plan first.
                    if prefix_items == 2
                        && !dialogue_compacted
                        && (retry_output_limit.is_some()
                            || budget.needs_summary(overhead + budget::estimate(&items), 8192))
                    {
                        let compacted = budget::summarize_for_window(
                            self.inner.model.as_ref(),
                            &items[..1],
                            budget,
                        )
                        .await?;
                        dialogue = format!(
                            "## Conversation history (summary)\n{}",
                            compacted.record(&mut task.usage)
                        );
                        dialogue_compacted = true;
                        items[0] = InputItem::Text(dialogue.clone());
                        observed_input = 0;
                    }
                }
                let requested_output =
                    retry_output_limit.unwrap_or(EXECUTOR_OUTPUT_TOKENS.max(output_floor));
                let output_limit = budget.checked_output_limit(
                    requested_output,
                    overhead + archived_cost + budget::estimate(&items[append_from..]),
                )?;
                let request = CompletionRequest {
                    instructions: agents::EXECUTOR_PROMPT.to_string(),
                    input: archived
                        .iter()
                        .chain(&items[append_from..])
                        .cloned()
                        .collect(),
                    model: None,
                    temperature: Some(0.2),
                    max_output_tokens: output_limit,
                    previous_response_id: None,
                    tools: exec_tools.clone(),
                };
                let (sent_request_id, output) = match self
                    .streamed_complete(&task, turn, request)
                    .await
                {
                    Ok(output) => output,
                    Err(err)
                        if retry_output_limit.is_none()
                            && budget::is_context_overflow(&err.to_string()) =>
                    {
                        retry_output_limit = Some((output_limit / 2).clamp(1, 1024));
                        continue;
                    }
                    // Truncated with nothing usable (a thinking model can spend the whole
                    // budget on reasoning): raise the output budget and call again, as long as
                    // the window has room left.
                    Err(err) if err.is_output_truncated() => {
                        let raised = budget.checked_output_limit(
                            requested_output.saturating_mul(2),
                            overhead + archived_cost + budget::estimate(&items[append_from..]),
                        );
                        let more_room = truncated_retries < budget::MAX_TRUNCATION_RETRIES;
                        match raised {
                            Ok(raised) if more_room && raised > output_limit => {
                                truncated_retries += 1;
                                output_floor = raised;
                                continue;
                            }
                            _ => {
                                failure = Some(format!(
                                    "executor agent call failed: {err}; max_output_tokens was already widened to {output_limit}"
                                ));
                                break;
                            }
                        }
                    }
                    Err(err) => {
                        failure = Some(format!("executor agent call failed: {err}"));
                        break;
                    }
                };
                retry_output_limit = None;
                if let Some(cost) = output.usage.as_ref() {
                    // Main execution request: refresh the current context usage in sync; the UI badge trusts only this number.
                    task.usage.record_context(
                        cost.input_tokens,
                        cost.output_tokens,
                        cost.cached_tokens,
                    );
                    // The server-reported input size is a lower bound for the estimate: when the estimate is low, compact early from it.
                    observed_input = observed_input.max(cost.input_tokens);
                    self.save(&task)?;
                }
                // Progress replies also go into the transcript: the model must know what it promised last turn.
                if !output.text.trim().is_empty() {
                    if !round_reply.is_empty() {
                        round_reply.push('\n');
                    }
                    round_reply.push_str(output.text.trim());
                }
                let response_start = items.len();
                if !output.text.trim().is_empty()
                    && !output.replay_items.iter().any(|v| v["role"] == "assistant")
                {
                    items.push(InputItem::Raw(
                        serde_json::json!({"role":"assistant","content":output.text}),
                    ));
                }
                // In reasoning mode, reasoning and function_call must be replayed verbatim.
                for raw in &output.replay_items {
                    items.push(InputItem::Raw(raw.clone()));
                }

                // Only structured tool calls count: code blocks in the reply text (including bash fences) are never run as commands,
                // otherwise any sample snippet the model writes would actually execute.
                let calls = output.tool_calls.clone();

                let mut stop_round = false;
                'calls: for call in &calls {
                    match call.name.as_str() {
                        "bash" => {
                            let command = call
                                .arguments
                                .get("command")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string();
                            if command.trim().is_empty() {
                                if !call.call_id.is_empty() {
                                    items.push(tools::function_output(
                                        call,
                                        "bash command must not be empty.".into(),
                                    ));
                                }
                                continue;
                            }

                            exec_seq += 1;
                            let message_id = self.record_tool_start(&task, turn, &command);
                            self.emit(
                                &task,
                                RuntimeEventKind::StepStarted {
                                    turn,
                                    index: exec_seq,
                                    script_preview: command
                                        .lines()
                                        .next()
                                        .unwrap_or("")
                                        .chars()
                                        .take(120)
                                        .collect(),
                                    message_id,
                                },
                            );

                            let result = match self
                                .inner
                                .executor
                                .execute(ExecutionRequest {
                                    script: command.clone(),
                                    cwd: cwd.clone(),
                                    task_dir: task_dir.clone(),
                                    env: task_env(&task, &workspace, &task_dir),
                                    timeout: Duration::from_secs(
                                        self.inner.config.policy.timeout_secs,
                                    ),
                                    // Several commands per turn: number the file names so they do not overwrite each other.
                                    turn: turn * 100 + exec_seq,
                                    cancel: self.cancel_flag(task.id),
                                })
                                .await
                            {
                                Ok(result) => result,
                                Err(relay_executor::ExecutorError::Cancelled) => {
                                    // The user pressed stop: the current process is already killed and the task wraps up at once.
                                    self.mark_cancelled(&mut task);
                                    return Ok(());
                                }
                                Err(err) => {
                                    failure = Some(format!("bash execution failed: {err}"));
                                    break 'calls;
                                }
                            };

                            let phase = result
                                .control
                                .as_ref()
                                .map(|control| control.phase)
                                .unwrap_or(ControlPhase::Continue);
                            self.persist_step(&task, turn, &command, &result, phase)?;
                            self.finish_tool(
                                message_id,
                                result.exit_code,
                                result.timed_out,
                                result.duration_ms,
                                &result.output_tail(4000),
                            );
                            self.emit(
                                &task,
                                RuntimeEventKind::StepFinished {
                                    turn,
                                    index: exec_seq,
                                    exit_code: result.exit_code,
                                    timed_out: result.timed_out,
                                    phase: phase.as_str().to_string(),
                                    output_tail: result.output_tail(600),
                                    message_id,
                                },
                            );

                            last_output = result.output_tail(4000);
                            round_steps.push(agents::TranscriptStep {
                                tool: "bash".into(),
                                input: command.clone(),
                                output: result.output_tail(4000),
                                exit_code: result.exit_code,
                            });
                            if let Some(control) = &result.control {
                                artifacts.extend(
                                    self.record_control_artifacts(&task, control, &cwd, &task_dir),
                                );
                                round_control = Some(control.clone());
                                // relay-state reported something other than continue: end this turn here and let the outer state machine decide.
                                if control.phase != ControlPhase::Continue {
                                    stop_round = true;
                                }
                            }
                            artifacts.extend(self.record_scan(&task, &cwd, since));
                            artifacts.extend(self.record_scan(&task, &task_dir, since));
                            artifacts.sort();
                            artifacts.dedup();

                            // Feed the execution result back to the model so it can decide the next step.
                            if !call.call_id.is_empty() {
                                items.push(tools::function_output(call, bash_feedback(&result)));
                            }
                        }
                        "read_file" => {
                            exec_seq += 1;
                            let script = tools::describe_call(call);
                            let message_id = self.record_tool_start(&task, turn, &script);
                            let text = tools::read_file_tool(&exec_allowed, &call.arguments);
                            self.emit(
                                &task,
                                RuntimeEventKind::StepStarted {
                                    turn,
                                    index: exec_seq,
                                    script_preview: script,
                                    message_id,
                                },
                            );
                            self.finish_tool(message_id, Some(0), false, 0, &text);
                            self.emit(
                                &task,
                                RuntimeEventKind::StepFinished {
                                    turn,
                                    index: exec_seq,
                                    exit_code: Some(0),
                                    timed_out: false,
                                    phase: "tool".into(),
                                    output_tail: tools::first_line(&text),
                                    message_id,
                                },
                            );
                            round_steps.push(agents::TranscriptStep {
                                tool: call.name.clone(),
                                input: call.arguments.to_string(),
                                output: text.clone(),
                                exit_code: Some(0),
                            });
                            items.push(tools::function_output(call, text));
                        }
                        "list_files" => {
                            exec_seq += 1;
                            let script = tools::describe_call(call);
                            let message_id = self.record_tool_start(&task, turn, &script);
                            let workspaces = self.inner.store.workspaces().unwrap_or_default();
                            let text = tools::list_files_tool(
                                &workspaces,
                                Some(&workspace.root_dir),
                                &call.arguments,
                            );
                            self.emit(
                                &task,
                                RuntimeEventKind::StepStarted {
                                    turn,
                                    index: exec_seq,
                                    script_preview: script,
                                    message_id,
                                },
                            );
                            self.finish_tool(message_id, Some(0), false, 0, &text);
                            self.emit(
                                &task,
                                RuntimeEventKind::StepFinished {
                                    turn,
                                    index: exec_seq,
                                    exit_code: Some(0),
                                    timed_out: false,
                                    phase: "tool".into(),
                                    output_tail: tools::first_line(&text),
                                    message_id,
                                },
                            );
                            round_steps.push(agents::TranscriptStep {
                                tool: call.name.clone(),
                                input: call.arguments.to_string(),
                                output: text.clone(),
                                exit_code: Some(0),
                            });
                            items.push(tools::function_output(call, text));
                        }
                        "read_image" => {
                            exec_seq += 1;
                            let script = tools::describe_call(call);
                            let message_id = self.record_tool_start(&task, turn, &script);
                            self.emit(
                                &task,
                                RuntimeEventKind::StepStarted {
                                    turn,
                                    index: exec_seq,
                                    script_preview: script,
                                    message_id,
                                },
                            );
                            let text = tools::read_image_tool(
                                self.inner.model.as_ref(),
                                self.inner.config.vision_model.as_deref(),
                                &exec_allowed,
                                &call.arguments,
                            )
                            .await;
                            self.finish_tool(message_id, Some(0), false, 0, &text);
                            self.emit(
                                &task,
                                RuntimeEventKind::StepFinished {
                                    turn,
                                    index: exec_seq,
                                    exit_code: Some(0),
                                    timed_out: false,
                                    phase: "tool".into(),
                                    output_tail: tools::first_line(&text),
                                    message_id,
                                },
                            );
                            round_steps.push(agents::TranscriptStep {
                                tool: call.name.clone(),
                                input: call.arguments.to_string(),
                                output: text.clone(),
                                exit_code: Some(0),
                            });
                            items.push(tools::function_output(call, text));
                        }
                        "search" | "fetch" => {
                            exec_seq += 1;
                            let script = tools::describe_call(call);
                            let message_id = self.record_tool_start(&task, turn, &script);
                            self.emit(
                                &task,
                                RuntimeEventKind::StepStarted {
                                    turn,
                                    index: exec_seq,
                                    script_preview: script,
                                    message_id,
                                },
                            );
                            let text = match self.inner.config.web_tools.as_deref() {
                                Some(web) if call.name == "search" => {
                                    tools::search_tool(web, &call.arguments).await
                                }
                                Some(web) => tools::fetch_tool(web, &call.arguments).await,
                                None => format!(
                                    "{} is unavailable: no web tools client is configured (an API key is required)",
                                    call.name
                                ),
                            };
                            self.finish_tool(message_id, Some(0), false, 0, &text);
                            self.emit(
                                &task,
                                RuntimeEventKind::StepFinished {
                                    turn,
                                    index: exec_seq,
                                    exit_code: Some(0),
                                    timed_out: false,
                                    phase: "tool".into(),
                                    output_tail: tools::first_line(&text),
                                    message_id,
                                },
                            );
                            round_steps.push(agents::TranscriptStep {
                                tool: call.name.clone(),
                                input: call.arguments.to_string(),
                                output: text.clone(),
                                exit_code: Some(0),
                            });
                            items.push(tools::function_output(call, text));
                        }
                        "computer_accessibility" | "computer_input" => {
                            exec_seq += 1;
                            let script = tools::describe_call(call);
                            let message_id = self.record_tool_start(&task, turn, &script);
                            self.emit(
                                &task,
                                RuntimeEventKind::StepStarted {
                                    turn,
                                    index: exec_seq,
                                    script_preview: script,
                                    message_id,
                                },
                            );
                            let text = computer::invoke(&call.name, &call.arguments).await;
                            self.finish_tool(message_id, Some(0), false, 0, &text);
                            self.emit(
                                &task,
                                RuntimeEventKind::StepFinished {
                                    turn,
                                    index: exec_seq,
                                    exit_code: Some(0),
                                    timed_out: false,
                                    phase: "tool".into(),
                                    output_tail: tools::first_line(&text),
                                    message_id,
                                },
                            );
                            round_steps.push(agents::TranscriptStep {
                                tool: call.name.clone(),
                                input: call.arguments.to_string(),
                                output: text.clone(),
                                exit_code: Some(0),
                            });
                            items.push(tools::function_output(call, text));
                        }
                        "submit_result" => {
                            let signal = agents::parse_result(call);
                            artifacts.extend(
                                self.record_control_artifacts(&task, &signal, &cwd, &task_dir),
                            );
                            artifacts.sort();
                            artifacts.dedup();
                            if !signal.summary.trim().is_empty() {
                                last_output = signal.summary.clone();
                            }
                            if !call.call_id.is_empty() {
                                items.push(tools::function_output(
                                    call,
                                    format!(
                                        "recorded: phase={} summary={}",
                                        signal.phase.as_str(),
                                        signal.summary
                                    ),
                                ));
                            }
                            round_control = Some(signal);
                            stop_round = true;
                        }
                        other => {
                            if !call.call_id.is_empty() {
                                items.push(tools::function_output(
                                    call,
                                    format!(
                                        "unknown tool: {other}. Available tools: {}",
                                        exec_tool_names.join(", ")
                                    ),
                                ));
                            }
                        }
                    }
                }

                if failure.is_none() {
                    let suffix = items[response_start..]
                        .iter()
                        .filter_map(|item| match item {
                            InputItem::Raw(value) => Some(value.clone()),
                            _ => None,
                        })
                        .collect::<Vec<_>>();
                    self.inner.store.checkpoint_executor(
                        task.id,
                        sent_request_id,
                        turn,
                        &suffix,
                    )?;
                }
                if failure.is_some() || stop_round || calls.is_empty() {
                    break;
                }
            }

            if failure.is_some() {
                break;
            }

            if let Some(control) = &round_control {
                if !control.summary.is_empty() && !round_reply.contains(&control.summary) {
                    round_reply.push('\n');
                    round_reply.push_str(&control.summary);
                }
            }
            let phase = round_control
                .as_ref()
                .map(|control| control.phase)
                .unwrap_or(ControlPhase::Continue);
            let round_had_steps = !round_steps.is_empty();
            if round_had_steps || !round_reply.trim().is_empty() || round_control.is_some() {
                transcript.push(TranscriptEntry {
                    turn,
                    reply: round_reply,
                    steps: round_steps,
                    phase: phase.as_str().to_string(),
                });
            }

            match phase {
                ControlPhase::Continue => {
                    if !round_had_steps && failure.is_none() {
                        tracing::debug!(
                            turn,
                            "no command executed this turn; moving to the next turn"
                        );
                    }
                }
                ControlPhase::AwaitApproval => {
                    // The direct session path has no confirmation card: waiting for approval would end the turn, so let the user reply instead,
                    // otherwise the task would keep holding the session lock and later messages would stay queued.
                    if skip_verify {
                        completed = true;
                        break;
                    }
                    if !self.transition(&mut task, TaskState::AwaitingConfirmation) {
                        return Ok(());
                    }
                    match self.ask_confirmation(&task, &plan, &workspace).await {
                        Confirmation::Approve | Confirmation::Resume => {
                            if !self.transition(&mut task, TaskState::Executing) {
                                return Ok(());
                            }
                        }
                        Confirmation::Revise(supplement) => {
                            task.supplement = Some(supplement);
                            if !self.transition(&mut task, TaskState::Executing) {
                                return Ok(());
                            }
                        }
                        Confirmation::Cancel => {
                            self.mark_cancelled(&mut task);
                            return Ok(());
                        }
                    }
                }
                ControlPhase::Paused => {
                    // Same as above: inside a session "pause" means this turn is over; wait for the user to continue in the input box.
                    if skip_verify {
                        completed = true;
                        break;
                    }
                    if !self.transition(&mut task, TaskState::Paused) {
                        return Ok(());
                    }
                    self.emit(
                        &task,
                        RuntimeEventKind::Notice {
                            message: round_control
                                .as_ref()
                                .map(|control| control.summary.clone())
                                .filter(|summary| !summary.is_empty())
                                .unwrap_or_else(|| "task paused; waiting to be resumed".into()),
                        },
                    );
                    match self.ask_confirmation(&task, &plan, &workspace).await {
                        Confirmation::Cancel => {
                            self.mark_cancelled(&mut task);
                            return Ok(());
                        }
                        Confirmation::Revise(supplement) => {
                            // The user changed the goal before mid-run approval: continue with the latest plan, not the old one.
                            task.supplement = Some(supplement);
                            if let Some(latest) = self.inner.store.task(task.id)? {
                                if let Some(edited) = latest.plan {
                                    task.title = edited.title.clone();
                                    task.delivery = edited.delivery;
                                    task.plan = Some(edited.clone());
                                    plan = edited;
                                }
                            }
                            if !self.transition(&mut task, TaskState::Executing) {
                                return Ok(());
                            }
                        }
                        _ => {
                            if let Some(latest) = self.inner.store.task(task.id)? {
                                if let Some(edited) = latest.plan {
                                    task.title = edited.title.clone();
                                    task.delivery = edited.delivery;
                                    task.plan = Some(edited.clone());
                                    plan = edited;
                                }
                            }
                            if !self.transition(&mut task, TaskState::Executing) {
                                return Ok(());
                            }
                        }
                    }
                }
                ControlPhase::Done => {
                    completed = true;
                    break;
                }
                ControlPhase::Failed => {
                    failure = Some(
                        round_control
                            .as_ref()
                            .map(|control| control.summary.clone())
                            .filter(|summary| !summary.trim().is_empty())
                            .unwrap_or_else(|| "the executed script reported failure".into()),
                    );
                    break;
                }
            }
        }

        if self.is_cancelled(task.id) {
            self.mark_cancelled(&mut task);
            return Ok(());
        }
        if !completed && failure.is_none() {
            failure = Some(format!(
                "not finished after reaching the maximum execution turns ({})",
                self.inner.config.max_rounds
            ));
        }

        if !self.transition(&mut task, TaskState::Verifying) {
            return Ok(());
        }
        let mut verify_usage = ContextUsage::default();
        let verification = if skip_verify {
            // Direct session path: no verifier agent; the execution result is the delivery verdict.
            agents::Verification {
                verdict: Verdict::Pass,
                summary: if last_output.trim().is_empty() {
                    plan.summary.clone()
                } else {
                    last_output.clone()
                },
                checks: Vec::new(),
                deliver: None,
            }
        } else {
            match self
                .verify(&task, &plan, &artifacts, &last_output, &workspace)
                .await
            {
                Ok((verification, usage)) => {
                    verify_usage = usage;
                    verification
                }
                Err(err) => agents::Verification {
                    verdict: Verdict::Fail,
                    summary: format!("verification call failed: {err}"),
                    checks: Vec::new(),
                    deliver: None,
                },
            }
        };

        task.usage.merge(verify_usage);
        self.save(&task)?;

        let mut report = agents::report_from(verification.clone(), Vec::new());
        if let Some(error) = failure {
            report.verdict = Some(Verdict::Fail);
            report.summary = error.clone();
            self.fail_with_report(&mut task, error, report);
            return Ok(());
        }

        match verification.verdict {
            Verdict::Pass => {
                if !self.transition(&mut task, TaskState::Delivering) {
                    return Ok(());
                }
                let preferred = verification
                    .deliver
                    .as_ref()
                    .and_then(|p| resolve_within(&cwd, p));
                let outcome =
                    self.deliver(&task, &report.summary, &artifacts, preferred.as_deref());
                report.delivered = outcome
                    .opened
                    .as_ref()
                    .map(|p| vec![p.display().to_string()])
                    .unwrap_or_default();
                // When the model only called submit_result with no reply delta, turn its verdict line into the reply,
                // otherwise this turn would show only tool records and no verdict.
                if !report.summary.trim().is_empty()
                    && !self
                        .inner
                        .store
                        .has_assistant_text(task.id)
                        .unwrap_or(false)
                {
                    self.record_message(
                        &task,
                        0,
                        MessageRole::Assistant,
                        report.summary.trim(),
                        None,
                    );
                }
                task.report = Some(report.clone());
                task.finished_at = Some(Utc::now());
                task.updated_at = Utc::now();
                self.save(&task)?;
                if self.transition(&mut task, TaskState::Succeeded) {
                    let context_line = task.context.summary_line();
                    let _ = self.inner.store.touch_session(
                        task.session_id,
                        &task.title,
                        Some(context_line.as_str()),
                    );
                }
                // The final reply travels with the completion event so the tray can show it
                // without re-reading the session; a turn that only called submit_result falls
                // back to its verdict line.
                let reply = self
                    .inner
                    .store
                    .last_assistant_text(task.id)
                    .ok()
                    .flatten()
                    .filter(|text| !text.trim().is_empty())
                    .unwrap_or_else(|| report.summary.clone());
                self.emit(
                    &task,
                    RuntimeEventKind::Completed {
                        report,
                        verdict: Some(Verdict::Pass),
                        artifacts,
                        reply,
                    },
                );
            }
            Verdict::Fail => {
                self.fail_with_report(&mut task, verification.summary.clone(), report);
            }
        }
        Ok(())
    }

    fn apply_plan(&self, task: &mut Task, plan: &relay_core::Plan) -> Result<()> {
        task.plan = Some(plan.clone());
        task.delivery = plan.delivery;
        task.title = plan.title.clone();
        task.updated_at = Utc::now();
        self.save(task)?;
        self.emit(task, RuntimeEventKind::PlanReady { plan: plan.clone() });
        Ok(())
    }

    /// Understanding agent: multi-turn tool exploration plus a submit_plan decision (workspace/session/plan in one go).
    async fn understand_task(
        &self,
        instruction: &str,
        supplement: Option<&str>,
        snapshot: &EnvironmentSnapshot,
        attachments: &[PathBuf],
        pinned: Option<WorkspaceId>,
    ) -> Result<crate::understanding::Understanding> {
        let session_dir = self
            .inner
            .config
            .data_dir
            .join("understanding")
            .join(uuid::Uuid::new_v4().to_string());
        std::fs::create_dir_all(&session_dir)?;

        // The authorized-directory list is only context for the tools: they no longer block paths, and any path is readable.
        let mut allowed: Vec<PathBuf> = self.inner.config.policy.allowed_dirs.clone();
        allowed.push(self.inner.config.data_dir.clone());
        for workspace in self.inner.store.workspaces().unwrap_or_default() {
            allowed.push(workspace.root_dir.clone());
        }
        let allowed: Vec<PathBuf> = allowed
            .into_iter()
            .filter_map(|path| path.canonicalize().ok())
            .collect();

        // Capture the current screen up front: capture and recognition happen before the model's first decision,
        // saving a full round trip between asking for a screenshot and submitting.
        // A user-attached screenshot beats the automatic capture: it is the region they chose,
        // with the strokes that say what to change.
        let prefetched = if let Some(path) = attachments.iter().find(|path| path.exists()) {
            Some(
                crate::understanding::attached_scene(
                    self.inner.model.as_ref(),
                    self.inner.config.vision_model.as_deref(),
                    path,
                )
                .await,
            )
        } else if self.inner.config.prefetch_screen {
            crate::understanding::prefetch_scene(
                self.inner.model.as_ref(),
                self.inner.config.vision_model.as_deref(),
                &session_dir,
            )
            .await
        } else {
            None
        };

        let input = UnderstandingInput {
            store: self.inner.store.as_ref(),
            context: snapshot,
            instruction,
            supplement,
            pinned_workspace: pinned,
            allowed_dirs: allowed,
            vision_model: self.inner.config.vision_model.clone(),
            screenshot_dir: session_dir,
            max_steps: self.inner.config.max_understand_steps,
            prefetched,
            budget: Budget::new(
                self.inner.config.context_window,
                self.inner.config.context_compact_ratio,
            ),
        };
        let agent = UnderstandingAgent::new(
            self.inner.model.clone(),
            self.inner.config.web_tools.clone(),
        );
        let sender = self.inner.understanding.clone();
        let notify_sender = sender.clone();
        let notify = move |event: UnderstandEvent| {
            let _ = notify_sender.send(event);
        };
        let result = agent.run(&input, &notify).await;
        if let Err(err) = &result {
            // Always wrap up, even on failure, or subscribers (tray / trace) would stay stuck in "understanding".
            let _ = sender.send(UnderstandEvent {
                step: 0,
                stage: UnderstandStage::Failed,
                tool: None,
                summary: format!("understanding failed: {err}"),
            });
        }
        result
    }

    /// Streaming model call for execution: deltas are broadcast to the UI at once; the final output still uses the existing parser.
    async fn streamed_complete(
        &self,
        task: &Task,
        turn: u32,
        request: CompletionRequest,
    ) -> Result<(i64, ModelOutput)> {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<StreamDelta>();
        let events = self.inner.events.clone();
        let store = self.inner.store.clone();
        let task_id = task.id;
        let session_id = task.session_id;
        let forward = tokio::spawn(async move {
            // Deltas are grouped into message blocks per role: create the row first to get an id, then append as they arrive.
            // Events carry only the id plus the delta, so the UI grows in place without rebuilding the whole history.
            let mut block: Option<MessageBlock> = None;
            while let Some(delta) = rx.recv().await {
                let (role, delta) = match delta {
                    StreamDelta::Text(delta) => (MessageRole::Assistant, delta),
                    StreamDelta::Reasoning(delta) => (MessageRole::Thinking, delta),
                };
                if block.as_ref().map(|block| block.role) != Some(role) {
                    flush_block(&store, &mut block);
                    let id = match store.open_message(session_id, task_id, turn, role, "", None) {
                        Ok(id) => Some(id),
                        Err(err) => {
                            tracing::warn!(%err, %task_id, "failed to write streaming message");
                            None
                        }
                    };
                    block = Some(MessageBlock {
                        role,
                        id,
                        pending: String::new(),
                    });
                }
                let message_id = match block.as_mut() {
                    Some(current) => {
                        current.pending.push_str(&delta);
                        current.id
                    }
                    None => None,
                };
                let kind = match role {
                    MessageRole::Assistant => RuntimeEventKind::AssistantDelta {
                        turn,
                        delta,
                        message_id,
                    },
                    _ => RuntimeEventKind::ThinkingDelta {
                        turn,
                        delta,
                        message_id,
                    },
                };
                let _ = events.send(RuntimeEvent::new(task_id, kind));
                let full = block
                    .as_ref()
                    .map(|current| current.pending.len() >= 128)
                    .unwrap_or(false);
                if full {
                    flush_block(&store, &mut block);
                }
            }
            flush_block(&store, &mut block);
        });
        let output = self.inner.model.stream(task.id, request, tx).await;
        let _ = forward.await;
        Ok(output?)
    }

    async fn verify(
        &self,
        task: &Task,
        plan: &relay_core::Plan,
        artifacts: &[PathBuf],
        last_output: &str,
        workspace: &Workspace,
    ) -> Result<(agents::Verification, ContextUsage)> {
        let input = agents::verifier_input(task, plan, artifacts, last_output);
        let tools = tools::verifier_tools();
        let tool_names: Vec<String> = tools.iter().map(|tool| tool.name.clone()).collect();
        let allowed = self.inner.config.policy.allowed_dirs.clone();
        let budget = Budget::new(
            self.inner.config.context_window,
            self.inner.config.context_compact_ratio,
        );
        let mut items = vec![InputItem::Text(input)];
        let mut usage = ContextUsage::default();
        for _ in 1..=self.inner.config.max_verify_steps.max(1) {
            let request = CompletionRequest {
                instructions: agents::VERIFIER_PROMPT.to_string(),
                input: Vec::new(),
                model: None,
                temperature: Some(0.2),
                max_output_tokens: 4096,
                previous_response_id: None,
                tools: tools.clone(),
            };
            let output = budget
                .complete(self.inner.model.as_ref(), request, &mut items, &mut usage)
                .await?;
            if let Some(cost) = output.usage.as_ref() {
                usage.record(cost.input_tokens, cost.output_tokens, cost.cached_tokens);
            }
            // Server requirement: reasoning and function_call must be replayed verbatim.
            for raw in &output.replay_items {
                items.push(InputItem::Raw(raw.clone()));
            }

            if let Some(call) = output
                .tool_calls
                .iter()
                .find(|call| call.name == "submit_verification")
            {
                return Ok((agents::parse_verification_call(call), usage));
            }
            if output.tool_calls.is_empty() {
                // Legacy protocol compatibility: emit VERDICT / SUMMARY / CHECKS / DELIVER text directly.
                if output.text.trim().is_empty() {
                    return Err(RuntimeError::Invalid(
                        "verification submitted no verdict".into(),
                    ));
                }
                return Ok((agents::parse_verification(&output.text), usage));
            }

            for call in &output.tool_calls {
                let text = match call.name.as_str() {
                    "read_file" => tools::read_file_tool(&allowed, &call.arguments),
                    "list_files" => {
                        let workspaces = self.inner.store.workspaces().unwrap_or_default();
                        tools::list_files_tool(
                            &workspaces,
                            Some(&workspace.root_dir),
                            &call.arguments,
                        )
                    }
                    "read_image" => {
                        tools::read_image_tool(
                            self.inner.model.as_ref(),
                            self.inner.config.vision_model.as_deref(),
                            &allowed,
                            &call.arguments,
                        )
                        .await
                    }
                    "bash" => {
                        // Verification must be read-only: even if the model calls bash, no command runs.
                        "bash is not allowed during verification; inspect with list_files / read_file / read_image, then submit_verification.".to_string()
                    }
                    other => format!(
                        "unknown tool: {other}. Available tools: {}",
                        tool_names.join(", ")
                    ),
                };
                items.push(tools::function_output(call, text));
            }
        }

        Err(RuntimeError::Invalid(format!(
            "verification did not submit a verdict within {} steps",
            self.inner.config.max_verify_steps
        )))
    }

    async fn ask_confirmation(
        &self,
        task: &Task,
        plan: &relay_core::Plan,
        workspace: &Workspace,
    ) -> Confirmation {
        let (tx, rx) = oneshot::channel();
        self.inner.pending.lock().unwrap().insert(task.id, tx);
        self.emit(
            task,
            RuntimeEventKind::ConfirmationRequested {
                title: plan.title.clone(),
                context_line: task.context.summary_line(),
                permissions: self.permissions_summary(),
                references: agents::workspace_listing(&workspace.root_dir, 3),
            },
        );
        let decision = self.await_decision(task.id, rx).await;
        self.inner.pending.lock().unwrap().remove(&task.id);
        decision
    }

    async fn await_decision(
        &self,
        task_id: TaskId,
        mut rx: oneshot::Receiver<Confirmation>,
    ) -> Confirmation {
        loop {
            if self.is_cancelled(task_id) {
                return Confirmation::Cancel;
            }
            tokio::select! {
                result = &mut rx => {
                    return result.unwrap_or(Confirmation::Cancel);
                }
                _ = tokio::time::sleep(Duration::from_millis(200)) => {}
            }
        }
    }

    fn needs_confirmation(&self, task_id: TaskId) -> bool {
        self.inner.config.policy.require_confirmation
            && !self.inner.auto_confirm.lock().unwrap().contains(&task_id)
    }

    /// Permission summary on the confirmation card, so the user sees the execution boundary before approving.
    fn permissions_summary(&self) -> String {
        let policy = &self.inner.config.policy;
        let mut parts = vec![if policy.sandbox {
            "sandbox isolation"
        } else {
            "no sandbox (path policy only)"
        }
        .to_string()];
        parts.push(format!("{} authorized dirs", policy.allowed_dirs.len()));
        parts.push(if policy.allow_network {
            "network allowed".into()
        } else {
            "network blocked".to_string()
        });
        parts.push(format!("{}s per turn", policy.timeout_secs));
        parts.push(format!("max {} turns", self.inner.config.max_rounds));
        parts.join(" · ")
    }

    /// On resume, rebuild the transcript from executed turns so the model can see what was done before.
    ///
    /// Each command in `steps` is one row; grouping by `turn` restores the original "several commands per turn" shape.
    fn transcript_from_steps(&self, task_id: TaskId) -> Result<Vec<TranscriptEntry>> {
        let steps = self.inner.store.steps(task_id)?;
        let mut transcript: Vec<TranscriptEntry> = Vec::new();
        for step in steps {
            let mut output = step.stdout.trim_end().to_string();
            if !step.stderr.trim().is_empty() {
                output.push_str("\n[stderr]\n");
                output.push_str(step.stderr.trim_end());
            }
            let entry = agents::TranscriptStep {
                tool: "bash".into(),
                input: step.script,
                output,
                exit_code: step.exit_code,
            };
            match transcript.last_mut() {
                Some(last) if last.turn == step.turn => last.steps.push(entry),
                _ => transcript.push(TranscriptEntry {
                    turn: step.turn,
                    reply: String::new(),
                    steps: vec![entry],
                    phase: step
                        .phase
                        .map(|phase| phase.as_str().to_string())
                        .unwrap_or_else(|| "continue".into()),
                }),
            }
        }
        Ok(transcript)
    }

    /// Earlier dialogue in the same session: gives the executor agent continuity across follow-up tasks.
    fn dialogue_history(
        &self,
        session_id: SessionId,
        current: TaskId,
    ) -> Result<(String, Option<TaskId>)> {
        let store = &self.inner.store;
        let messages =
            store.prior_dialogue_messages(session_id, current, agents::DIALOGUE_MAX_TURNS)?;
        let mut outcomes = HashMap::new();
        for message in messages.iter().filter(|m| m.role == MessageRole::User) {
            let Some(task) = store.task(message.task_id)? else {
                continue;
            };
            let mut outcome = format!("\ntask state: {}", task.state.label());
            if let Some(report) = &task.report {
                outcome.push_str(&format!(
                    "\ntask report: {}",
                    truncate_text(&report.summary, 500)
                ));
            }
            if let Some(error) = &task.error {
                outcome.push_str(&format!("\nfailure reason: {}", truncate_text(error, 400)));
            }
            for artifact in store.artifacts(task.id)?.iter().take(8) {
                outcome.push_str(&format!("\nrecorded artifact: {}", artifact.path.display()));
            }
            outcomes.insert(task.id, outcome);
        }
        let previous = messages
            .iter()
            .rev()
            .find(|m| m.role == MessageRole::User)
            .map(|m| m.task_id);
        Ok((
            agents::dialogue_with_outcomes(
                &messages,
                current,
                None,
                self.inner.config.dialogue_budget(),
                &outcomes,
            ),
            previous,
        ))
    }

    fn session_gate(&self, session_id: SessionId) -> Arc<tokio::sync::Mutex<()>> {
        self.inner
            .session_gates
            .lock()
            .unwrap()
            .entry(session_id)
            .or_default()
            .clone()
    }

    fn release_session_gate(&self, session_id: SessionId, gate: &Arc<tokio::sync::Mutex<()>>) {
        let mut gates = self.inner.session_gates.lock().unwrap();
        let idle = gates
            .get(&session_id)
            .map(|current| Arc::ptr_eq(current, gate) && Arc::strong_count(gate) == 2)
            .unwrap_or(false);
        if idle {
            gates.remove(&session_id);
        }
    }

    /// While queued, check the cancel flag every 200ms so a cancelled task does not keep the session.
    async fn wait_cancel(&self, task_id: TaskId) {
        while !self.is_cancelled(task_id) {
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }

    fn mark_cancelled_by_id(&self, task_id: TaskId) {
        if let Ok(Some(mut task)) = self.inner.store.task(task_id) {
            self.mark_cancelled(&mut task);
        }
    }

    fn is_cancelled(&self, task_id: TaskId) -> bool {
        self.inner
            .cancels
            .lock()
            .unwrap()
            .get(&task_id)
            .map(|flag| flag.load(Ordering::SeqCst))
            .unwrap_or(false)
    }

    fn persist_step(
        &self,
        task: &Task,
        turn: u32,
        script: &str,
        result: &ExecutionResult,
        phase: ControlPhase,
    ) -> Result<()> {
        let step = TaskStep {
            id: uuid::Uuid::new_v4().to_string(),
            task_id: task.id,
            turn,
            script: script.to_string(),
            stdout: result.stdout.clone(),
            stderr: result.stderr.clone(),
            exit_code: result.exit_code,
            timed_out: result.timed_out,
            duration_ms: result.duration_ms,
            phase: Some(phase),
            created_at: Utc::now(),
        };
        self.inner.store.add_step(&step)?;
        Ok(())
    }

    fn record_control_artifacts(
        &self,
        task: &Task,
        control: &relay_core::ControlSignal,
        cwd: &Path,
        task_dir: &Path,
    ) -> Vec<PathBuf> {
        let mut out = Vec::new();
        for raw in &control.artifacts {
            let candidate = PathBuf::from(raw);
            let candidates = if candidate.is_absolute() {
                vec![candidate]
            } else {
                vec![cwd.join(&candidate), task_dir.join(&candidate)]
            };
            if let Some(found) = candidates.into_iter().find(|p| p.exists()) {
                self.record_artifact(task.id, &found);
                out.push(found);
            } else {
                tracing::warn!(path = %raw, "declared artifact does not exist");
            }
        }
        out
    }

    fn record_scan(&self, task: &Task, root: &Path, since: SystemTime) -> Vec<PathBuf> {
        let files = scan_artifacts(root, since);
        let mut out = Vec::new();
        for path in files {
            if path.ends_with("state.json") || path.to_string_lossy().contains("/.relay/") {
                continue;
            }
            self.record_artifact(task.id, &path);
            out.push(path);
        }
        out
    }

    fn record_artifact(&self, task_id: TaskId, path: &Path) {
        if let Ok(existing) = self.inner.store.artifacts(task_id) {
            if existing.iter().any(|artifact| artifact.path == path) {
                return;
            }
        }
        let meta = std::fs::metadata(path);
        let kind = if meta.as_ref().map(|m| m.is_dir()).unwrap_or(false) {
            ArtifactKind::Folder
        } else if path.extension().and_then(|e| e.to_str()) == Some("log") {
            ArtifactKind::Log
        } else if meta.is_ok() {
            ArtifactKind::File
        } else {
            ArtifactKind::Other
        };
        let artifact = Artifact {
            id: uuid::Uuid::new_v4().to_string(),
            task_id,
            path: path.to_path_buf(),
            kind,
            size: meta.map(|m| m.len()).unwrap_or(0),
            created_at: Utc::now(),
        };
        if let Err(err) = self.inner.store.add_artifact(&artifact) {
            tracing::warn!(%err, "failed to record artifact");
        }
    }

    fn deliver(
        &self,
        task: &Task,
        body: &str,
        artifacts: &[PathBuf],
        preferred: Option<&Path>,
    ) -> DeliveryOutcome {
        let mode = if self.inner.config.policy.auto_open {
            task.delivery
        } else {
            DeliveryMode::NotifyOnly
        };
        if !self.inner.config.notifications {
            return DeliveryOutcome::default();
        }
        relay_delivery::deliver(
            &task.title,
            body,
            mode,
            artifacts,
            preferred,
            self.inner.config.policy.auto_open,
        )
    }

    fn transition(&self, task: &mut Task, next: TaskState) -> bool {
        if task.state == next {
            return true;
        }
        if task.state.is_terminal() {
            return false;
        }
        let from = task.state;
        match task.state.transition_to(next) {
            Ok(state) => {
                task.state = state;
                task.updated_at = Utc::now();
                if let Err(err) = self.inner.store.save_task(task) {
                    tracing::warn!(%err, "failed to save task state");
                }
                if state.is_terminal() {
                    tracing::info!(task = %task.id, state = state.as_str(), "task finished");
                } else {
                    tracing::debug!(task = %task.id, from = from.as_str(), to = state.as_str(), "task state transition");
                }
                self.emit(task, RuntimeEventKind::StateChanged { from, to: state });
                true
            }
            Err(err) => {
                tracing::warn!(%err, from = from.as_str(), to = next.as_str(), "illegal state transition");
                false
            }
        }
    }

    fn mark_cancelled(&self, task: &mut Task) {
        if task.state.is_terminal() {
            return;
        }
        let from = task.state;
        if let Ok(state) = task.state.transition_to(TaskState::Cancelled) {
            self.record_message(task, 0, MessageRole::Notice, "task cancelled", None);
            task.state = state;
            task.updated_at = Utc::now();
            task.finished_at = Some(Utc::now());
            let _ = self.inner.store.save_task(task);
            self.emit(task, RuntimeEventKind::StateChanged { from, to: state });
            self.emit(task, RuntimeEventKind::Cancelled);
        }
    }

    fn fail_with(&self, task: &mut Task, error: String) {
        let report = relay_core::TaskReport {
            verdict: Some(Verdict::Fail),
            summary: error.clone(),
            checks: Vec::new(),
            delivered: Vec::new(),
        };
        self.fail_with_report(task, error, report);
    }

    fn fail_with_report(&self, task: &mut Task, error: String, mut report: relay_core::TaskReport) {
        if task.state.is_terminal() {
            return;
        }
        report.verdict = Some(Verdict::Fail);
        self.record_message(task, 0, MessageRole::Error, &error, None);
        task.error = Some(error.clone());
        task.report = Some(report);
        task.finished_at = Some(Utc::now());
        task.updated_at = Utc::now();
        if self.transition(task, TaskState::Failed) {
            let _ = self.inner.store.save_task(task);
        }
        self.emit(
            task,
            RuntimeEventKind::Failed {
                error: error.clone(),
            },
        );
        let _ = relay_delivery::notify(&format!("{} · not completed", task.title), &error);
    }

    fn fail_task(&self, task_id: TaskId, error: String) {
        if let Ok(Some(mut task)) = self.inner.store.task(task_id) {
            if task.state.is_terminal() {
                return;
            }
            self.fail_with(&mut task, error);
        }
    }

    /// Persist session messages: failures are only logged; display data must never affect the task itself.
    fn record_message(
        &self,
        task: &Task,
        turn: u32,
        role: MessageRole,
        content: &str,
        meta: Option<&ToolMeta>,
    ) -> Option<i64> {
        match self
            .inner
            .store
            .open_message(task.session_id, task.id, turn, role, content, meta)
        {
            Ok(id) => Some(id),
            Err(err) => {
                tracing::warn!(%err, task = %task.id, role = role.as_str(), "failed to write session message");
                None
            }
        }
    }

    fn record_tool_start(&self, task: &Task, turn: u32, script: &str) -> Option<i64> {
        self.record_message(
            task,
            turn,
            MessageRole::Tool,
            script,
            Some(&ToolMeta::default()),
        )
    }

    fn finish_tool(
        &self,
        message_id: Option<i64>,
        exit_code: Option<i32>,
        timed_out: bool,
        duration_ms: u64,
        output: &str,
    ) {
        let Some(message_id) = message_id else {
            return;
        };
        let meta = ToolMeta {
            done: true,
            exit_code,
            timed_out,
            duration_ms,
            output: output.to_string(),
        };
        if let Err(err) = self.inner.store.update_message_meta(message_id, &meta) {
            tracing::warn!(%err, "failed to update tool message");
        }
    }

    fn save(&self, task: &Task) -> Result<()> {
        self.inner.store.save_task(task)?;
        Ok(())
    }

    fn emit(&self, task: &Task, kind: RuntimeEventKind) {
        let transient = matches!(
            kind,
            RuntimeEventKind::AssistantDelta { .. } | RuntimeEventKind::ThinkingDelta { .. }
        );
        let event = RuntimeEvent::new(task.id, kind);
        if !transient {
            if let Err(err) = self.inner.store.log_event(&event) {
                tracing::warn!(%err, "failed to write event");
            }
        }
        let _ = self.inner.events.send(event);
    }
}

/// One streaming block: consecutive deltas of the same role go into the same `messages` row.
struct MessageBlock {
    role: MessageRole,
    id: Option<i64>,
    pending: String,
}

/// Append buffered deltas to the message row; if the row is gone, just drop the buffer.
fn flush_block(store: &Store, block: &mut Option<MessageBlock>) {
    let Some(current) = block.as_mut() else {
        return;
    };
    if let Some(id) = current.id {
        if let Err(err) = store.append_message(id, &current.pending) {
            tracing::warn!(%err, "failed to append session message");
        }
    }
    current.pending.clear();
}

fn write_state_helper(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, include_str!("../../../runtime/bin/relay-state"))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(path)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(path, perms)?;
    }
    Ok(())
}

/// Rename entries created by earlier localized builds so existing workspaces keep working.
/// The legacy names are written as escapes to keep this source ASCII-only.
fn migrate_legacy_layout(root: &Path) {
    for (legacy, current) in [
        ("\u{89c4}\u{5219}.md", "rules.md"),
        ("\u{8d44}\u{6599}", "materials"),
        ("\u{6210}\u{679c}", "outputs"),
    ] {
        let from = root.join(legacy);
        let to = root.join(current);
        if from.exists() && !to.exists() {
            if let Err(err) = std::fs::rename(&from, &to) {
                tracing::warn!(
                    from = %from.display(),
                    to = %to.display(),
                    %err,
                    "failed to migrate a legacy workspace entry"
                );
            }
        }
    }
}

fn task_env(task: &Task, workspace: &Workspace, task_dir: &Path) -> Vec<(String, String)> {
    vec![
        ("RELAY_TASK_ID".into(), task.id.to_string()),
        ("RELAY_SESSION_ID".into(), task.session_id.to_string()),
        ("RELAY_WORKSPACE_ID".into(), workspace.id.to_string()),
        ("RELAY_WORKSPACE_NAME".into(), workspace.name.clone()),
        (
            "RELAY_WORKSPACE_DIR".into(),
            workspace.root_dir.to_string_lossy().to_string(),
        ),
        (
            "RELAY_TASK_DIR".into(),
            task_dir.to_string_lossy().to_string(),
        ),
    ]
}

/// Fallback when the model names no workspace: frontmost app name, then window title, then "Default Workspace".

/// Compress one execution result into short feedback the model can read.
fn bash_feedback(result: &ExecutionResult) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "exit_code={:?} timed_out={} duration={}ms\n",
        result.exit_code, result.timed_out, result.duration_ms
    ));
    let stdout = result.stdout.trim_end();
    if !stdout.is_empty() {
        out.push_str("--- stdout ---\n");
        out.push_str(&truncate_text(stdout, 3000));
        out.push('\n');
    }
    let stderr = result.stderr.trim_end();
    if !stderr.is_empty() {
        out.push_str("--- stderr ---\n");
        out.push_str(&truncate_text(stderr, 1500));
        out.push('\n');
    }
    if let Some(control) = &result.control {
        out.push_str(&format!(
            "--- relay-state ---\nphase={} summary={}\n",
            control.phase.as_str(),
            control.summary
        ));
    }
    out
}

fn truncate_text(value: &str, max: usize) -> String {
    if value.chars().count() <= max {
        return value.to_string();
    }
    let mut out: String = value.chars().take(max).collect();
    out.push('…');
    out
}

fn read_pid(path: &Path) -> Option<u32> {
    std::fs::read_to_string(path)
        .ok()?
        .trim()
        .parse::<u32>()
        .ok()
}

#[cfg(unix)]
fn process_alive(pid: u32) -> bool {
    let Ok(output) = std::process::Command::new("kill")
        .args(["-0", &pid.to_string()])
        .output()
    else {
        return false;
    };
    if output.status.success() {
        return true;
    }
    // EPERM means the process exists but we may not signal it (e.g. pid 1).
    let stderr = String::from_utf8_lossy(&output.stderr).to_lowercase();
    stderr.contains("not permitted")
}

#[cfg(windows)]
fn process_alive(pid: u32) -> bool {
    std::process::Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}")])
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).contains(&pid.to_string()))
        .unwrap_or(false)
}

#[cfg(not(any(unix, windows)))]
fn process_alive(_pid: u32) -> bool {
    false
}

fn derive_workspace_name(snapshot: &EnvironmentSnapshot) -> String {
    let app = snapshot.app_name.trim();
    if !app.is_empty() {
        return app.to_string();
    }
    let title = snapshot.window_title.trim();
    if !title.is_empty() {
        return title.chars().take(16).collect();
    }
    "Default Workspace".to_string()
}

/// Minimal plan for direct session execution: the goal is this turn's message; no routing or decomposition.
fn direct_plan(instruction: &str) -> Plan {
    Plan {
        title: short_title(instruction),
        summary: "execute this session message directly".into(),
        steps: Vec::new(),
        references: Vec::new(),
        delivery: DeliveryMode::NotifyOnly,
        raw: String::new(),
    }
}

fn short_title(instruction: &str) -> String {
    let line = instruction.lines().next().unwrap_or("").trim();
    let mut title: String = line.chars().take(30).collect();
    if line.chars().count() > 30 {
        title.push('…');
    }
    if title.is_empty() {
        "Untitled task".into()
    } else {
        title
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn migrates_legacy_workspace_layout() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join("\u{8d44}\u{6599}")).unwrap();
        std::fs::create_dir_all(root.join("\u{6210}\u{679c}")).unwrap();
        std::fs::write(root.join("\u{89c4}\u{5219}.md"), "# old rules").unwrap();

        migrate_legacy_layout(root);

        assert!(root.join("materials").is_dir());
        assert!(root.join("outputs").is_dir());
        assert!(root.join("rules.md").is_file());
        assert!(!root.join("\u{8d44}\u{6599}").exists());
        assert!(!root.join("\u{6210}\u{679c}").exists());
        assert!(!root.join("\u{89c4}\u{5219}.md").exists());
    }

    use super::*;
    use relay_context::ManualContextProvider;
    use relay_openai::MockModel;

    fn test_runtime_at(data_dir: std::path::PathBuf) -> Runtime {
        let store = Arc::new(Store::open(data_dir.join("relay.sqlite3")).unwrap());
        let mut config = RuntimeConfig::new(&data_dir);
        config.notifications = false;
        config.policy.sandbox = false;
        Runtime::new(
            store,
            Arc::new(MockModel::new(Vec::<String>::new())),
            Arc::new(ManualContextProvider::from_env()),
            config,
        )
        .unwrap()
    }

    fn test_runtime() -> Runtime {
        let dir = tempfile::tempdir().unwrap().keep();
        let store = Arc::new(Store::open(dir.join("relay.sqlite3")).unwrap());
        let mut config = RuntimeConfig::new(&dir);
        config.notifications = false;
        config.policy.sandbox = false;
        Runtime::new(
            store,
            Arc::new(MockModel::new(Vec::<String>::new())),
            Arc::new(ManualContextProvider::from_env()),
            config,
        )
        .unwrap()
    }

    fn seed_unfinished_task(runtime: &Runtime) -> TaskId {
        let now = Utc::now();
        let task = Task {
            id: TaskId::new(),
            workspace_id: WorkspaceId::new(),
            session_id: SessionId::new(),
            title: "task waiting for confirmation".into(),
            instruction: "write a file".into(),
            supplement: None,
            context: EnvironmentSnapshot::default(),
            usage: ContextUsage::default(),
            plan: None,
            state: TaskState::Understanding,
            report: None,
            delivery: relay_core::DeliveryMode::NotifyOnly,
            error: None,
            created_at: now,
            updated_at: now,
            started_at: Some(now),
            finished_at: None,
        };
        runtime.store().save_task(&task).unwrap();
        task.id
    }

    #[tokio::test]
    async fn recovery_is_skipped_while_another_instance_is_alive() {
        let runtime = test_runtime();
        let task_id = seed_unfinished_task(&runtime);
        // pid 1 (launchd) always exists and is not the current process.
        std::fs::write(runtime.config_pid_path(), "1").unwrap();

        // Simulate "a CLI taking a read-only look at the task list": a new Runtime starts.
        let fresh = test_runtime_at(runtime.config().data_dir.clone());
        let task = fresh.task(task_id).unwrap().unwrap();
        assert_eq!(
            task.state,
            TaskState::Understanding,
            "another instance running means its tasks must not be marked as interrupted failures"
        );
    }

    #[tokio::test]
    async fn recovery_runs_when_no_other_instance_is_alive() {
        let runtime = test_runtime();
        let task_id = seed_unfinished_task(&runtime);
        // A pid that cannot possibly be alive.
        std::fs::write(runtime.config_pid_path(), "4294967290").unwrap();

        let fresh = test_runtime_at(runtime.config().data_dir.clone());
        let task = fresh.task(task_id).unwrap().unwrap();
        assert_eq!(
            task.state,
            TaskState::Failed,
            "interrupted tasks are recovered when no other instance is alive"
        );
    }

    #[tokio::test]
    async fn same_session_gate_is_shared_and_blocks_parallel_run() {
        let runtime = test_runtime();
        let session = SessionId::new();
        let first = runtime.session_gate(session);
        let second = runtime.session_gate(session);
        assert!(
            Arc::ptr_eq(&first, &second),
            "the same session must reuse the same gate"
        );

        let guard = first.clone().lock_owned().await;
        assert!(
            second.clone().try_lock_owned().is_err(),
            "a second task in the same session must not run in parallel with the first"
        );
        drop(guard);
        let guard = second.lock_owned().await;
        drop(guard);

        let other = runtime.session_gate(SessionId::new());
        assert!(
            other.try_lock_owned().is_ok(),
            "different sessions must not block each other"
        );
    }

    #[tokio::test]
    async fn idle_session_gate_is_released() {
        let runtime = test_runtime();
        let session = SessionId::new();
        let gate = runtime.session_gate(session);
        assert_eq!(runtime.inner.session_gates.lock().unwrap().len(), 1);
        runtime.release_session_gate(session, &gate);
        assert!(
            runtime.inner.session_gates.lock().unwrap().is_empty(),
            "idle session gates are released to avoid unbounded growth"
        );
    }
}
