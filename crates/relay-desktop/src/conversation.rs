//! Session UI: workspace -> session tree on the left, agent conversation stream on the right.
//!
//! All data comes from local SQLite: every task inside a session is one turn,
//! user messages carry the raw instruction, while the agent side shows plan, run log and result in order.
//! The composer stays on the current session: new tasks always reuse that session.
//!
//! Visual tone: warm white canvas plus a light gray sidebar; deep green only for key actions and emphasis;
//! dividers are always hairlines, whitespace instead of stacked cards, reading column fixed at 760px and centered.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::{DateTime, Datelike, Local};
use gpui_kit::assets::IconName as Lucide;
use gpui_kit::component::input::{InputEvent, Textarea, TextareaState};
use gpui_kit::component::scroll::ScrollableElement as _;
use gpui_kit::component::text::TextView;
use gpui_kit::component::{h_flex, v_flex, Icon, Root, Sizable as _};
use gpui_kit::*;
use relay_core::{
    ChatMessageRecord, ContextUsage, MessageRole, RuntimeEvent, RuntimeEventKind, Session,
    SessionId, TaskId, TaskState, ToolMeta, Workspace, WorkspaceId,
};
use relay_runtime::TaskRequest;

use crate::app::DesktopContext;

/// The session window is single-instance too: reopening only raises the existing one.
static CONVERSATION_WINDOW: Mutex<Option<WindowHandle<Root>>> = Mutex::new(None);

// ---- Palette ----------------------------------------------------------------
// Accent: low-saturation deep green, only on key actions, active states and the brand dot.
const ACCENT: u32 = 0x1F7A5C;
const ACCENT_DEEP: u32 = 0x165C44;
const ACCENT_SOFT: u32 = 0xE8F3ED;
const ACCENT_LINE: u32 = 0xC9E2D6;
// Neutrals: warm grays, four steps from body text to faint.
const INK: u32 = 0x1A1A18;
const BODY: u32 = 0x43433E;
const MUTED: u32 = 0x767670;
const FAINT: u32 = 0xA9A9A2;
// Surfaces: white main area, light gray sidebar, tinted container, hover.
const CANVAS: u32 = 0xFFFFFF;
const SIDEBAR_BG: u32 = 0xF6F6F4;
const SINK: u32 = 0xF7F7F5;
const HOVER: u32 = 0xEBEBE8;
// Lines: regular dividers and hairlines.
const BORDER: u32 = 0xE8E8E3;
const HAIRLINE: u32 = 0xF0F0EC;
/// User messages use a light warm gray bubble.
const USER_BUBBLE: u32 = 0xF2F2EF;
// Terminal output: dark gray-green background.
const TERMINAL_BG: u32 = 0x232A27;
const TERMINAL_TEXT: u32 = 0xD9E3DD;
// Semantic colors.
const ERROR_BG: u32 = 0xFBF4F2;
const ERROR_LINE: u32 = 0xF1DCD8;
const ERROR_TEXT: u32 = 0xA8402F;
const WARN_TEXT: u32 = 0x9A6A12;
const WARN_BG: u32 = 0xFAF3E2;

/// Reading width: the stream and the composer share one center line.
const READING_WIDTH: f32 = 760.;

/// Sessions shown per workspace before the "show more" row appears; each click adds another page.
const SESSION_PAGE: usize = 5;

struct WorkspaceNode {
    workspace: Workspace,
    sessions: Vec<Session>,
}

/// One record in the stream: history replay and streaming share the same model.
#[derive(Debug, Clone)]
struct ChatMessage {
    /// `messages` primary key; 0 means not persisted yet (optimistic message / date divider).
    id: i64,
    role: MessageRole,
    /// Display time (HH:MM); empty for date dividers.
    time: String,
    /// Used to compute the duration of a turn.
    at: DateTime<Local>,
    /// Status of the task behind a user message; empty for other roles.
    state: Option<TaskState>,
    body: String,
    /// Execution result of a tool message; empty for other roles.
    tool: Option<ToolMeta>,
}

impl ChatMessage {
    fn divider(label: String, at: DateTime<Local>) -> Self {
        Self {
            id: 0,
            role: MessageRole::Notice,
            time: String::new(),
            at,
            state: None,
            body: label,
            tool: None,
        }
    }

    fn user(body: String, at: DateTime<Local>) -> Self {
        Self {
            id: 0,
            role: MessageRole::User,
            time: at.format("%H:%M").to_string(),
            at,
            state: None,
            body,
            tool: None,
        }
    }

    fn error(body: String, at: DateTime<Local>) -> Self {
        Self {
            id: 0,
            role: MessageRole::Error,
            time: String::new(),
            at,
            state: None,
            body,
            tool: None,
        }
    }

    fn from_record(record: ChatMessageRecord, state: Option<TaskState>) -> Self {
        let at = record.created_at.with_timezone(&Local);
        Self {
            id: record.id,
            role: record.role,
            time: at.format("%H:%M").to_string(),
            at,
            state,
            body: record.content,
            tool: record.meta,
        }
    }

    /// A delta arrived but the block is not local yet (e.g. the UI just opened).
    fn streamed(id: i64, role: MessageRole, delta: String) -> Self {
        let at = Local::now();
        Self {
            id,
            role,
            time: at.format("%H:%M").to_string(),
            at,
            state: None,
            body: delta,
            tool: None,
        }
    }

    fn is_divider(&self) -> bool {
        self.role == MessageRole::Notice && self.time.is_empty() && self.id == 0
    }
}

pub struct ConversationView {
    ctx: Arc<DesktopContext>,
    workspaces: Vec<WorkspaceNode>,
    /// Expanded workspaces; collapsed by default so the tree stays a compact list.
    expanded: HashSet<WorkspaceId>,
    /// Per-workspace cap on visible sessions; absent means the first page (`SESSION_PAGE`).
    session_limits: HashMap<WorkspaceId, usize>,
    selected: Option<(WorkspaceId, SessionId)>,
    messages: Vec<ChatMessage>,
    /// Task -> session map, used to attribute live events to the open session.
    task_sessions: HashMap<TaskId, SessionId>,
    scroll: ScrollHandle,
    pending_scroll: bool,
    focus_handle: FocusHandle,
    /// Bottom composer: continues the current session.
    composer: Entity<TextareaState>,
    /// User messages already sent while the runtime creates the task (shown optimistically).
    optimistic: Vec<ChatMessage>,
    sending: bool,
    composer_error: Option<String>,
    /// Manually toggled turn traces (key: turn id, value: expanded).
    trace_open: HashMap<String, bool>,
    /// Context usage of the latest task in this session; drives the window share badge.
    usage: Option<ContextUsage>,
    /// Task running in this session: when set the send button becomes Stop.
    running_task: Option<TaskId>,
    /// Expanded tool groups / thinking rows inside them / tool output.
    expanded_tools: HashSet<String>,
    expanded_thinking: HashSet<String>,
    expanded_outputs: HashSet<i64>,
}

impl ConversationView {
    fn new(window: &mut Window, cx: &mut Context<Self>, ctx: Arc<DesktopContext>) -> Self {
        let focus_handle = cx.focus_handle();
        focus_handle.focus(window, cx);
        let composer = cx.new(|cx| {
            TextareaState::new(window, cx)
                .auto_grow(1, 5)
                .submit_on_enter(true)
                .placeholder("Tell Relay what to do next…")
        });
        let mut view = Self {
            ctx,
            workspaces: Vec::new(),
            expanded: HashSet::new(),
            session_limits: HashMap::new(),
            selected: None,
            messages: Vec::new(),
            task_sessions: HashMap::new(),
            scroll: ScrollHandle::new(),
            pending_scroll: false,
            focus_handle,
            composer,
            optimistic: Vec::new(),
            sending: false,
            composer_error: None,
            trace_open: HashMap::new(),
            usage: None,
            running_task: None,
            expanded_tools: HashSet::new(),
            expanded_thinking: HashSet::new(),
            expanded_outputs: HashSet::new(),
        };
        cx.subscribe_in(
            &view.composer,
            window,
            |this: &mut Self, _state, event: &InputEvent, window, cx| {
                if let InputEvent::PressEnter { shift, .. } = event {
                    if !*shift {
                        this.send(window, cx);
                    }
                }
            },
        )
        .detach();
        let composer_focus = view.composer.read(cx).focus_handle(cx);
        composer_focus.focus(window, cx);
        view.reload_tree();
        view.ensure_selection(cx);
        view.watch_runtime(cx);
        view
    }

    /// Refresh the tree and the open session on state changes so the UI grows with progress.
    fn watch_runtime(&mut self, cx: &mut Context<Self>) {
        let mut events = self.ctx.runtime.subscribe();
        cx.spawn(async move |this, cx| loop {
            match events.recv().await {
                Ok(event) => {
                    let alive = this
                        .update(cx, |this, cx| {
                            if this.apply_event(&event) {
                                cx.notify();
                            }
                        })
                        .is_ok();
                    if !alive {
                        break;
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(_) => break,
            }
        })
        .detach();
    }

    fn reload_tree(&mut self) {
        let mut nodes = Vec::new();
        for workspace in self.ctx.runtime.workspaces().unwrap_or_default() {
            let sessions = self
                .ctx
                .runtime
                .sessions(workspace.id, 50)
                .unwrap_or_default();
            nodes.push(WorkspaceNode {
                workspace,
                sessions,
            });
        }
        self.workspaces = nodes;

        // The selected session may be gone (e.g. the data directory was reset).
        if let Some((workspace_id, session_id)) = self.selected {
            let alive = self.workspaces.iter().any(|node| {
                node.workspace.id == workspace_id
                    && node.sessions.iter().any(|session| session.id == session_id)
            });
            if !alive {
                self.selected = None;
                self.messages.clear();
            }
        }
    }

    /// With no session selected, open the most recently active one.
    fn ensure_selection(&mut self, cx: &mut Context<Self>) {
        if self.selected.is_some() {
            return;
        }
        let latest = self
            .workspaces
            .iter()
            .flat_map(|node| {
                node.sessions
                    .iter()
                    .map(move |session| (node.workspace.id, session.id, session.last_active_at))
            })
            .max_by_key(|(_, _, at)| *at);
        if let Some((workspace_id, session_id, _)) = latest {
            self.open_session(workspace_id, session_id, cx);
        }
    }

    fn open_session(
        &mut self,
        workspace_id: WorkspaceId,
        session_id: SessionId,
        cx: &mut Context<Self>,
    ) {
        if self.selected == Some((workspace_id, session_id)) {
            return;
        }
        self.selected = Some((workspace_id, session_id));
        self.task_sessions.clear();
        self.trace_open.clear();
        self.expanded_tools.clear();
        self.expanded_thinking.clear();
        self.expanded_outputs.clear();
        self.messages = self.load_messages(session_id);
        self.pending_scroll = true;
        cx.notify();
    }

    /// Dev / screenshot use: jump straight to the session in RELAY_OPEN_SESSION.
    fn open_by_id(&mut self, raw: &str, cx: &mut Context<Self>) {
        let Ok(session_id) = raw.trim().parse::<SessionId>() else {
            return;
        };
        let workspace_id = self.workspaces.iter().find_map(|node| {
            node.sessions
                .iter()
                .find(|session| session.id == session_id)
                .map(|_| node.workspace.id)
        });
        if let Some(workspace_id) = workspace_id {
            self.open_session(workspace_id, session_id, cx);
        }
    }

    /// Send a follow-up message: always the current session, executed directly.
    fn send(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        // No new sends while running: the button is Stop and Enter must not bypass it.
        if self.sending || self.running_task.is_some() {
            return;
        }
        let Some((workspace_id, session_id)) = self.selected else {
            return;
        };
        let text = self.composer.read(cx).value().trim().to_string();
        if text.is_empty() {
            return;
        }
        self.composer
            .update(cx, |state, cx| state.set_value("", window, cx));
        self.sending = true;
        self.composer_error = None;
        self.usage = None;
        self.optimistic
            .push(ChatMessage::user(text.clone(), Local::now()));
        self.pending_scroll = true;
        cx.notify();

        tracing::info!(%session_id, chars = text.chars().count(), "session follow-up: submitting task");
        let runtime = self.ctx.runtime.clone();
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.ctx.tokio.spawn(async move {
            let request = TaskRequest {
                instruction: text,
                supplement: None,
                pinned_workspace: Some(workspace_id),
                pinned_session: Some(session_id),
                // The session window is the conversation: Enter confirms, skipping understanding and verification.
                auto_confirm: true,
                direct: true,
                attachments: Vec::new(),
            };
            let _ = tx.send(runtime.submit(request).await);
        });

        cx.spawn(async move |this, cx| {
            let result = rx.await;
            let _ = this.update(cx, |this, cx| {
                this.sending = false;
                match result {
                    Ok(Ok(task_id)) => {
                        tracing::info!(%task_id, %session_id, "session follow-up: task created");
                        // The task is persisted, so the user message now comes from the stored record.
                        this.optimistic.clear();
                        this.task_sessions.insert(task_id, session_id);
                        this.running_task = Some(task_id);
                        if this.selected == Some((workspace_id, session_id)) {
                            this.reload_messages();
                        }
                        this.reload_tree();
                        this.pending_scroll = true;
                    }
                    Ok(Err(err)) => {
                        tracing::warn!(%err, %session_id, "session follow-up: submit failed");
                        this.optimistic.clear();
                        this.composer_error = Some(format!("Send failed: {err}"));
                    }
                    Err(_) => {
                        this.optimistic.clear();
                        this.composer_error = Some("Send failed: runtime closed".into());
                    }
                }
                cx.notify();
            });
        })
        .detach();
    }

    fn is_current(&mut self, task_id: TaskId) -> bool {
        let Some((_, session_id)) = self.selected else {
            return false;
        };
        if let Some(known) = self.task_sessions.get(&task_id) {
            return *known == session_id;
        }
        // Tasks that existed before the window opened: look them up once and cache ownership.
        if let Ok(Some(task)) = self.ctx.runtime.task(task_id) {
            self.task_sessions.insert(task_id, task.session_id);
            return task.session_id == session_id;
        }
        false
    }

    /// Map runtime events into the current session stream; returns whether a repaint is needed.
    ///
    /// Messages are already persisted in order by the Runtime; this only merges deltas:
    /// stream deltas append in place by message_id, tool steps take the latest row by id, end events reconcile the turn.
    fn apply_event(&mut self, event: &RuntimeEvent) -> bool {
        let task_id = event.task_id;
        match &event.kind {
            RuntimeEventKind::Routed { session_id, .. } => {
                self.task_sessions.insert(task_id, *session_id);
                self.reload_tree();
                let current = self.is_current(task_id);
                if current {
                    self.running_task = Some(task_id);
                }
                current
            }
            RuntimeEventKind::AssistantDelta {
                delta, message_id, ..
            } => {
                if !self.is_current(task_id) {
                    return false;
                }
                self.append_stream(*message_id, MessageRole::Assistant, delta);
                self.pending_scroll = true;
                true
            }
            RuntimeEventKind::ThinkingDelta {
                delta, message_id, ..
            } => {
                if !self.is_current(task_id) {
                    return false;
                }
                self.append_stream(*message_id, MessageRole::Thinking, delta);
                self.pending_scroll = true;
                true
            }
            RuntimeEventKind::StepStarted { message_id, .. }
            | RuntimeEventKind::StepFinished { message_id, .. } => {
                if !self.is_current(task_id) {
                    return false;
                }
                if let Some(id) = message_id {
                    self.sync_record(*id);
                }
                // Refresh context usage after each step so the badge updates before the task ends.
                self.sync_usage(task_id);
                self.pending_scroll = true;
                true
            }
            RuntimeEventKind::StateChanged { to, .. } => {
                if !self.is_current(task_id) {
                    return false;
                }
                // Show Stop only while really running (understanding/executing/delivering), not while awaiting confirmation or paused.
                if to.is_active() {
                    self.running_task = Some(task_id);
                } else if self.running_task == Some(task_id) {
                    self.running_task = None;
                }
                // State changes and conclusions are persisted: reconcile the turn so deltas do not drift from the store.
                self.reload_messages();
                true
            }
            RuntimeEventKind::PlanReady { .. }
            | RuntimeEventKind::Completed { .. }
            | RuntimeEventKind::Failed { .. }
            | RuntimeEventKind::Cancelled => {
                if !self.is_current(task_id) {
                    return false;
                }
                if self.running_task == Some(task_id) {
                    self.running_task = None;
                }
                // State changes and conclusions are persisted: reconcile the turn so deltas do not drift from the store.
                self.reload_messages();
                true
            }
            _ => false,
        }
    }

    /// Stop the running task in this session: write Cancelled at once and tell the pipeline to exit.
    fn stop(&mut self, cx: &mut Context<Self>) {
        let Some(task_id) = self.running_task else {
            return;
        };
        match self.ctx.runtime.cancel(task_id) {
            Ok(()) => {
                tracing::info!(%task_id, "user stopped the run");
                self.running_task = None;
                self.sending = false;
                self.reload_messages();
                self.pending_scroll = true;
            }
            Err(err) => {
                tracing::warn!(%err, %task_id, "failed to stop the task");
                self.composer_error = Some(format!("Stop failed: {err}"));
            }
        }
        cx.notify();
    }

    /// Refresh the badge from the latest usage; keep the old value when the model returned no usage.
    fn sync_usage(&mut self, task_id: TaskId) {
        if let Ok(Some(task)) = self.ctx.runtime.task(task_id) {
            if !task.usage.is_empty() {
                self.usage = Some(task.usage);
            }
        }
    }

    /// Append stream deltas to the matching block; create it on arrival if missing.
    fn append_stream(&mut self, id: Option<i64>, role: MessageRole, delta: &str) {
        let id = id.unwrap_or(0);
        if let Some(message) = self
            .messages
            .iter_mut()
            .find(|message| message.id == id && message.role == role)
        {
            message.body.push_str(delta);
            return;
        }
        self.messages
            .push(ChatMessage::streamed(id, role, delta.to_string()));
    }

    /// Replace the local message with the stored row (tool command and output are complete only there).
    fn sync_record(&mut self, id: i64) {
        let Ok(record) = self.ctx.runtime.message(id) else {
            return;
        };
        let Some(record) = record else {
            return;
        };
        let message = ChatMessage::from_record(record, None);
        if let Some(existing) = self.messages.iter_mut().find(|item| item.id == id) {
            *existing = message;
        } else {
            self.messages.push(message);
        }
    }

    fn reload_messages(&mut self) {
        let Some((_, session_id)) = self.selected else {
            return;
        };
        let before = self.messages.len();
        self.messages = self.load_messages(session_id);
        if self.messages.len() > before {
            self.pending_scroll = true;
        }
    }

    fn load_messages(&mut self, session_id: SessionId) -> Vec<ChatMessage> {
        let records = match self.ctx.runtime.messages(session_id) {
            Ok(records) => records,
            Err(err) => {
                return vec![ChatMessage::error(
                    format!("Failed to load the session: {err}"),
                    Local::now(),
                )]
            }
        };

        let mut states: HashMap<TaskId, TaskState> = HashMap::new();
        self.usage = None;
        self.running_task = None;
        for view in self
            .ctx
            .runtime
            .tasks_for_session(session_id, 50)
            .unwrap_or_default()
        {
            self.task_sessions.insert(view.task.id, session_id);
            states.insert(view.task.id, view.task.state);
            // A task already running before the window opened: restore it as Running, with the button set to stop.
            if self.running_task.is_none() && view.task.state.is_active() {
                self.running_task = Some(view.task.id);
            }
            if self.usage.is_none() && !view.task.usage.is_empty() {
                self.usage = Some(view.task.usage);
            }
        }

        let mut messages = Vec::new();
        let mut last_day = String::new();
        for record in records {
            // The plan is internal orchestration; the stream keeps only user, thinking, tools and replies.
            if record.role == MessageRole::Plan {
                continue;
            }
            let at = record.created_at.with_timezone(&Local);
            let day = at.format("%Y-%m-%d").to_string();
            if day != last_day {
                messages.push(ChatMessage::divider(day_label(&at), at));
                last_day = day;
            }
            // The status badge sits on the user message, the entry of a turn, so conclusions map to a turn at a glance.
            let state = if record.role == MessageRole::User {
                states.get(&record.task_id).copied()
            } else {
                None
            };
            messages.push(ChatMessage::from_record(record, state));
        }
        messages
    }

    fn current(&self) -> Option<(&Workspace, &Session)> {
        let (workspace_id, session_id) = self.selected?;
        self.workspaces.iter().find_map(|node| {
            if node.workspace.id != workspace_id {
                return None;
            }
            node.sessions
                .iter()
                .find(|session| session.id == session_id)
                .map(|session| (&node.workspace, session))
        })
    }

    fn toggle_workspace(&mut self, workspace_id: WorkspaceId) {
        if self.expanded.remove(&workspace_id) {
            // Collapsing parks the workspace back on the first page, so re-opening shows 5 again.
            self.session_limits.remove(&workspace_id);
        } else {
            self.expanded.insert(workspace_id);
        }
    }

    /// Light sidebar: workspace tree only; the active session is a white row with a thin border, others tint on hover.
    fn sidebar(&self, cx: &mut Context<Self>) -> AnyElement {
        let mut tree = v_flex().w_full().gap_1();
        if self.workspaces.is_empty() {
            tree = tree.child(
                v_flex()
                    .gap_2()
                    .p_4()
                    .rounded_xl()
                    .border_1()
                    .border_color(rgb(BORDER))
                    .bg(rgb(CANVAS))
                    .child(
                        glyph(Lucide::MessageSquare)
                            .with_size(px(16.))
                            .text_color(rgb(FAINT)),
                    )
                    .child(
                        div()
                            .text_sm()
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(rgb(BODY))
                            .child("No workspaces yet"),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(MUTED))
                            .line_height(rems(1.5))
                            .child(format!(
                                "Hand Relay its first job with {} and a workspace appears automatically.",
                                self.ctx.config.hotkey
                            )),
                    ),
            );
        } else {
            for node in &self.workspaces {
                let workspace_id = node.workspace.id;
                let expanded = self.expanded.contains(&workspace_id);
                tree = tree.child(
                    h_flex()
                        .id(format!("ws-{workspace_id}"))
                        .w_full()
                        .gap_2()
                        .items_center()
                        .px_2()
                        .py_2()
                        .rounded_lg()
                        .cursor_pointer()
                        .hover(|style| style.bg(rgb(HOVER)))
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.toggle_workspace(workspace_id);
                            cx.notify();
                        }))
                        .child(
                            Icon::new(if expanded {
                                Lucide::ChevronDown
                            } else {
                                Lucide::ChevronRight
                            })
                            .with_size(px(12.))
                            .text_color(rgb(FAINT)),
                        )
                        .child(
                            glyph(Lucide::FolderOpen)
                                .with_size(px(14.))
                                .text_color(rgb(MUTED)),
                        )
                        .child(
                            div()
                                .flex_1()
                                .min_w(px(0.))
                                .overflow_hidden()
                                .truncate()
                                .text_sm()
                                .font_weight(FontWeight::MEDIUM)
                                .text_color(rgb(INK))
                                .child(node.workspace.name.clone()),
                        )
                        .child(
                            div()
                                .text_xs()
                                // Highlight the count when this workspace holds the open session, so a
                                // collapsed row still tells you where you are.
                                .text_color(rgb(
                                    if node.sessions.iter().any(|session| {
                                        self.selected == Some((workspace_id, session.id))
                                    }) {
                                        ACCENT
                                    } else {
                                        FAINT
                                    },
                                ))
                                .child(node.sessions.len().to_string()),
                        ),
                );

                if !expanded {
                    continue;
                }
                if node.sessions.is_empty() {
                    tree = tree.child(
                        div()
                            .pl(px(28.))
                            .py_1()
                            .text_xs()
                            .text_color(rgb(FAINT))
                            .child("No sessions yet"),
                    );
                    continue;
                }
                let limit = self
                    .session_limits
                    .get(&workspace_id)
                    .copied()
                    .unwrap_or(SESSION_PAGE);
                let mut sessions = v_flex().w_full().gap_0p5().pl_3().pb_1();
                for session in node.sessions.iter().take(limit) {
                    let session_id = session.id;
                    let active = self.selected == Some((workspace_id, session_id));
                    let stamp = session_stamp(&session.last_active_at);
                    let mut row = h_flex()
                        .id(format!("ss-{session_id}"))
                        .w_full()
                        .gap_2()
                        .items_center()
                        .px_2()
                        .py_2()
                        .rounded_lg()
                        .cursor_pointer()
                        .hover(move |style| style.bg(rgb(if active { CANVAS } else { HOVER })))
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.open_session(workspace_id, session_id, cx);
                        }))
                        .child(
                            glyph(Lucide::MessageSquare)
                                .with_size(px(12.))
                                .text_color(rgb(if active { ACCENT } else { FAINT })),
                        )
                        .child(
                            div()
                                .flex_1()
                                .min_w(px(0.))
                                .overflow_hidden()
                                .truncate()
                                .text_size(px(13.))
                                .text_color(rgb(if active { ACCENT_DEEP } else { BODY }))
                                .child(session.title.clone()),
                        )
                        .child(
                            div()
                                .flex_shrink_0()
                                .text_xs()
                                .text_color(rgb(if active { ACCENT } else { FAINT }))
                                .child(stamp),
                        );
                    if active {
                        row = row.bg(rgb(CANVAS)).border_1().border_color(rgb(BORDER));
                    }
                    sessions = sessions.child(row);
                }
                let hidden = node.sessions.len().saturating_sub(limit);
                if hidden > 0 {
                    let step = hidden.min(SESSION_PAGE);
                    let left = hidden - step;
                    let mut more = h_flex()
                        .id(format!("more-{workspace_id}"))
                        .w_full()
                        .gap_1()
                        .items_center()
                        .pl_2()
                        .pr_2()
                        .py_1()
                        .rounded_lg()
                        .cursor_pointer()
                        .hover(|style| style.bg(rgb(HOVER)))
                        .on_click(cx.listener(move |this, _, _, cx| {
                            let shown = this
                                .session_limits
                                .get(&workspace_id)
                                .copied()
                                .unwrap_or(SESSION_PAGE);
                            this.session_limits
                                .insert(workspace_id, shown + SESSION_PAGE);
                            cx.notify();
                        }))
                        .child(
                            glyph(Lucide::ChevronDown)
                                .with_size(px(12.))
                                .text_color(rgb(FAINT)),
                        )
                        .child(
                            div()
                                .text_xs()
                                .text_color(rgb(MUTED))
                                .child(format!("Show {step} more")),
                        );
                    if left > 0 {
                        more = more.child(
                            div()
                                .flex_1()
                                .text_xs()
                                .text_color(rgb(FAINT))
                                .child(format!("{left} left")),
                        );
                    }
                    sessions = sessions.child(more);
                }
                tree = tree.child(sessions);
            }
        }

        v_flex()
            .w(px(264.))
            .h_full()
            .flex_shrink_0()
            .bg(rgb(SIDEBAR_BG))
            .border_r_1()
            .border_color(rgb(BORDER))
            .child(
                div()
                    .flex_1()
                    .overflow_y_scrollbar()
                    .px_3()
                    .pt_4()
                    .pb_6()
                    .child(tree),
            )
            .into_any_element()
    }

    /// Top bar: 64px tall, session title and breadcrumb on the left; actions live in the native titlebar.
    fn header(&self) -> AnyElement {
        let (workspace, session) = match self.current() {
            Some(pair) => pair,
            None => {
                return h_flex()
                    .h(px(64.))
                    .flex_shrink_0()
                    .px_6()
                    .items_center()
                    .bg(rgb(CANVAS))
                    .border_b_1()
                    .border_color(rgb(BORDER))
                    .child(
                        h_flex()
                            .gap_3()
                            .items_center()
                            .child(
                                div()
                                    .size(px(32.))
                                    .rounded_lg()
                                    .bg(rgb(SINK))
                                    .border_1()
                                    .border_color(rgb(BORDER))
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .child(
                                        glyph(Lucide::MessageSquare)
                                            .with_size(px(14.))
                                            .text_color(rgb(FAINT)),
                                    ),
                            )
                            .child(
                                v_flex()
                                    .gap_0p5()
                                    .child(
                                        div()
                                            .text_sm()
                                            .font_weight(FontWeight::SEMIBOLD)
                                            .text_color(rgb(INK))
                                            .child("Relay · Sessions"),
                                    )
                                    .child(
                                        div()
                                            .text_xs()
                                            .text_color(rgb(MUTED))
                                            .child("Pick a session from a workspace on the left"),
                                    ),
                            ),
                    )
                    .into_any_element();
            }
        };

        let rounds = self
            .messages
            .iter()
            .filter(|message| message.role == MessageRole::User)
            .count();

        let mut title_row = h_flex()
            .gap_2()
            .items_center()
            .min_w(px(0.))
            .child(
                div()
                    .min_w(px(0.))
                    .overflow_hidden()
                    .truncate()
                    .text_lg()
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(rgb(INK))
                    .child(session.title.clone()),
            )
            .child(
                div()
                    .flex_shrink_0()
                    .px_2()
                    .py_0p5()
                    .rounded_full()
                    .bg(rgb(SINK))
                    .border_1()
                    .border_color(rgb(BORDER))
                    .text_xs()
                    .text_color(rgb(MUTED))
                    .child(format!("{rounds} turns")),
            );
        if let Some(usage) = self.usage {
            let window = self.ctx.runtime.context_window();
            // The badge shows current context occupancy: only main requests of run turns count,
            // understanding / summary / verification helpers are excluded (much smaller, they would understate the share).
            let used = usage.occupancy();
            if window > 0 && used > 0 {
                let tight = used as f32 / window as f32 >= self.ctx.runtime.context_compact_ratio();
                let mut label = format!("Context {} / {}", fmt_tokens(used), fmt_tokens(window));
                let cache = usage.cache_ratio();
                if cache > 0.0 {
                    label.push_str(&format!(" · cached {:.0}%", cache * 100.0));
                }
                title_row = title_row.child(
                    div()
                        .flex_shrink_0()
                        .px_2()
                        .py_0p5()
                        .rounded_full()
                        .bg(rgb(if tight { WARN_BG } else { SINK }))
                        .border_1()
                        .border_color(rgb(BORDER))
                        .text_xs()
                        .text_color(rgb(if tight { WARN_TEXT } else { MUTED }))
                        .child(label),
                );
                if usage.compactions > 0 {
                    title_row = title_row.child(
                        div()
                            .flex_shrink_0()
                            .px_2()
                            .py_0p5()
                            .rounded_full()
                            .bg(rgb(SINK))
                            .border_1()
                            .border_color(rgb(BORDER))
                            .text_xs()
                            .text_color(rgb(MUTED))
                            .child(format!("{} compactions", usage.compactions)),
                    );
                }
            }
        }
        if self.sending || self.running_task.is_some() {
            title_row = title_row.child(
                h_flex()
                    .flex_shrink_0()
                    .gap_1()
                    .items_center()
                    .px_2()
                    .py_0p5()
                    .rounded_full()
                    .bg(rgb(ACCENT_SOFT))
                    .child(div().size(px(5.)).rounded_full().bg(rgb(ACCENT)))
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(ACCENT_DEEP))
                            .child("Running"),
                    ),
            );
        }

        h_flex()
            .h(px(64.))
            .flex_shrink_0()
            .px_6()
            .items_center()
            .bg(rgb(CANVAS))
            .border_b_1()
            .border_color(rgb(BORDER))
            .child(
                v_flex()
                    .flex_1()
                    .min_w(px(0.))
                    .gap_1()
                    .child(title_row)
                    .child(
                        h_flex()
                            .gap_2()
                            .items_center()
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(rgb(FAINT))
                                    .child(glyph(Lucide::FolderOpen).with_size(px(12.))),
                            )
                            .child(
                                div()
                                    .text_xs()
                                    .font_weight(FontWeight::MEDIUM)
                                    .text_color(rgb(MUTED))
                                    .child(workspace.name.clone()),
                            )
                            .child(div().text_xs().text_color(rgb(FAINT)).child("·"))
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(rgb(FAINT))
                                    .truncate()
                                    .child(workspace.root_dir.display().to_string()),
                            ),
                    ),
            )
            .into_any_element()
    }

    /// Message area: centered in the 760px reading column; the empty state is a centered guide.
    fn message_area(&self, cx: &mut Context<Self>) -> AnyElement {
        let blank = self.messages.is_empty() && self.optimistic.is_empty() && !self.sending;
        if blank {
            let empty =
                v_flex()
                    .size_full()
                    .items_center()
                    .justify_center()
                    .gap_3()
                    .child(
                        div()
                            .size(px(64.))
                            .rounded_full()
                            .bg(rgb(ACCENT_SOFT))
                            .flex()
                            .items_center()
                            .justify_center()
                            .child(
                                glyph(Lucide::MessageSquareText)
                                    .with_size(px(26.))
                                    .text_color(rgb(ACCENT)),
                            ),
                    )
                    .child(
                        div()
                            .text_xl()
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(rgb(INK))
                            .child(if self.selected.is_some() {
                                "Start a conversation"
                            } else {
                                "Select a session"
                            }),
                    )
                    .child(div().text_sm().text_color(rgb(MUTED)).child(
                        if self.selected.is_some() {
                            "Tell Relay what you want done; Enter starts right away."
                        } else {
                            "Open a conversation in a workspace on the left."
                        },
                    ));
            return div()
                .flex_1()
                .min_w(px(0.))
                .overflow_y_scrollbar()
                .child(empty)
                .into_any_element();
        }

        let all: Vec<&ChatMessage> = self.messages.iter().chain(self.optimistic.iter()).collect();
        let flow = build_flow(&all);
        let mut list = v_flex()
            .w_full()
            .max_w(px(READING_WIDTH))
            .min_w(px(0.))
            .gap_8()
            .pt_8()
            .pb_10();
        let mut user: Option<&ChatMessage> = None;
        for (key, item) in flow.iter().enumerate() {
            list = list.child(match item {
                FlowItem::Divider(label) => divider_view(label),
                FlowItem::User(index) => {
                    user = Some(all[*index]);
                    self.user_view(all[*index], key, cx)
                }
                FlowItem::Turn(indices) => self.turn_view(&all, indices, user, key, cx),
            });
        }

        div()
            .id("conversation-messages")
            .flex_1()
            .min_w(px(0.))
            .track_scroll(&self.scroll)
            .overflow_y_scroll()
            .bg(rgb(CANVAS))
            .child(v_flex().w_full().px_8().items_center().child(list))
            .into_any_element()
    }

    /// Composer: floating rounded card, same width and center as the reading column; Enter sends, Shift + Enter breaks the line.
    fn composer_bar(&self, cx: &mut Context<Self>) -> AnyElement {
        if self.selected.is_none() {
            return h_flex()
                .flex_shrink_0()
                .w_full()
                .px_8()
                .py_4()
                .gap_2()
                .items_center()
                .justify_center()
                .bg(rgb(CANVAS))
                .child(
                    glyph(Lucide::PanelRight)
                        .with_size(px(14.))
                        .text_color(rgb(FAINT)),
                )
                .child(
                    div()
                        .text_xs()
                        .text_color(rgb(MUTED))
                        .child("Pick a session on the left to continue the conversation here"),
                )
                .into_any_element();
        }

        let running = self.running_task.is_some();
        let mut panel = v_flex()
            .w_full()
            .max_w(px(READING_WIDTH))
            .min_w(px(0.))
            .gap_2();
        if let Some(error) = &self.composer_error {
            panel = panel.child(
                h_flex()
                    .w_full()
                    .gap_2()
                    .items_center()
                    .px_1()
                    .child(
                        glyph(Lucide::CircleAlert)
                            .with_size(px(12.))
                            .text_color(rgb(ERROR_TEXT)),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(ERROR_TEXT))
                            .child(error.clone()),
                    ),
            );
        }
        panel = panel.child(
            v_flex()
                .w_full()
                .min_w(px(0.))
                .rounded_2xl()
                .border_1()
                .border_color(rgb(BORDER))
                .bg(rgb(CANVAS))
                .shadow_md()
                .child(
                    div().px_5().pt_4().pb_2().child(
                        Textarea::new(&self.composer)
                            .appearance(false)
                            .into_any_element(),
                    ),
                )
                .child(
                    h_flex()
                        .w_full()
                        .justify_between()
                        .items_center()
                        .gap_3()
                        .px_4()
                        .pt_1()
                        .pb_3()
                        .child(
                            h_flex()
                                .gap_2()
                                .items_center()
                                .child(div().text_xs().text_color(rgb(FAINT)).child(if running {
                                    "Running · click the button to stop"
                                } else {
                                    "Enter to send · Shift + Enter for a new line"
                                }))
                                .child(div().size(px(3.)).rounded_full().bg(rgb(FAINT)))
                                .child(
                                    h_flex()
                                        .gap_1()
                                        .items_center()
                                        .child(
                                            glyph(Lucide::Zap)
                                                .with_size(px(11.))
                                                .text_color(rgb(ACCENT)),
                                        )
                                        .child(
                                            div()
                                                .text_xs()
                                                .text_color(rgb(MUTED))
                                                .child("Direct run"),
                                        ),
                                ),
                        )
                        .child(
                            div()
                                .id("conversation-send")
                                .size(px(32.))
                                .rounded_full()
                                .flex()
                                .flex_shrink_0()
                                .items_center()
                                .justify_center()
                                .cursor_pointer()
                                .bg(rgb(if running {
                                    ACCENT_DEEP
                                } else if self.sending {
                                    0xE7E7E3
                                } else {
                                    ACCENT
                                }))
                                .hover(|style| style.opacity(0.9))
                                .on_click(cx.listener(|this, _, window, cx| {
                                    if this.running_task.is_some() {
                                        this.stop(cx);
                                    } else {
                                        this.send(window, cx);
                                    }
                                }))
                                .child(if running {
                                    // Stop: a solid square, distinct from the send arrow.
                                    glyph(Lucide::Square)
                                        .with_size(px(11.))
                                        .text_color(rgb(CANVAS))
                                        .into_any_element()
                                } else {
                                    glyph(Lucide::ArrowUp)
                                        .with_size(px(15.))
                                        .text_color(rgb(if self.sending { FAINT } else { CANVAS }))
                                        .into_any_element()
                                }),
                        ),
                ),
        );

        v_flex()
            .flex_shrink_0()
            .w_full()
            .px_8()
            .pt_2()
            .pb_5()
            .items_center()
            .bg(rgb(CANVAS))
            .child(panel)
            .into_any_element()
    }
}

/// One visual unit in the stream: a date divider, a user message or a full assistant turn.
enum FlowItem {
    Divider(String),
    User(usize),
    Turn(Vec<usize>),
}

/// A turn is every assistant message between two user messages; tools and thinking merge inside it in order.
fn build_flow(all: &[&ChatMessage]) -> Vec<FlowItem> {
    let mut flow = Vec::new();
    let mut turn: Vec<usize> = Vec::new();
    for (index, message) in all.iter().enumerate() {
        if message.is_divider() {
            if !turn.is_empty() {
                flow.push(FlowItem::Turn(std::mem::take(&mut turn)));
            }
            flow.push(FlowItem::Divider(message.body.clone()));
        } else if message.role == MessageRole::User {
            if !turn.is_empty() {
                flow.push(FlowItem::Turn(std::mem::take(&mut turn)));
            }
            flow.push(FlowItem::User(index));
        } else {
            turn.push(index);
        }
    }
    if !turn.is_empty() {
        flow.push(FlowItem::Turn(turn));
    }
    flow
}

impl ConversationView {
    /// User message: right-aligned warm gray bubble; meta (time, copy / edit) fades in on hover.
    fn user_view(&self, message: &ChatMessage, key: usize, cx: &mut Context<Self>) -> AnyElement {
        let copy_body = message.body.clone();
        let edit_body = message.body.clone();
        let meta = h_flex()
            .w_full()
            .justify_end()
            .items_center()
            .gap_1()
            .child(
                div()
                    .px_1()
                    .text_xs()
                    .text_color(rgb(FAINT))
                    .child(message.time.clone()),
            )
            .child(
                div()
                    .id(format!("message-copy-{key}"))
                    .size(px(26.))
                    .rounded_md()
                    .flex()
                    .items_center()
                    .justify_center()
                    .cursor_pointer()
                    .hover(|style| style.bg(rgb(HOVER)))
                    .child(
                        glyph(Lucide::Copy)
                            .with_size(px(12.))
                            .text_color(rgb(FAINT)),
                    )
                    .on_click(cx.listener(move |_, _, _, _| {
                        let _ = relay_platform::clipboard::write_text(&copy_body);
                    })),
            )
            .child(
                div()
                    .id(format!("message-edit-{key}"))
                    .size(px(26.))
                    .rounded_md()
                    .flex()
                    .items_center()
                    .justify_center()
                    .cursor_pointer()
                    .hover(|style| style.bg(rgb(HOVER)))
                    .child(
                        glyph(Lucide::Pencil)
                            .with_size(px(12.))
                            .text_color(rgb(FAINT)),
                    )
                    .on_click(cx.listener(move |this, _, window, cx| {
                        this.composer.update(cx, |state, cx| {
                            state.set_value(edit_body.clone(), window, cx);
                        });
                        let focus = this.composer.read(cx).focus_handle(cx);
                        focus.focus(window, cx);
                    })),
            );
        // Hover group: entering or leaving the whole user message fades time and action buttons together.
        let group = format!("user-row-{key}");
        v_flex()
            .w_full()
            .items_end()
            .gap_1()
            .group(group.clone())
            .child(
                v_flex()
                    .max_w(px(620.))
                    .min_w(px(0.))
                    .px_5()
                    .py_3()
                    .rounded_2xl()
                    .rounded_br_sm()
                    .bg(rgb(USER_BUBBLE))
                    .text_base()
                    .text_color(rgb(INK))
                    .line_height(rems(1.65))
                    .child(
                        TextView::markdown(
                            ElementId::Name(format!("user-md-{key}").into()),
                            message.body.clone(),
                        )
                        .selectable(true)
                        .scrollable(false),
                    ),
            )
            .child(
                meta.opacity(0.0)
                    .group_hover(group, |style| style.opacity(1.0)),
            )
            .into_any_element()
    }

    /// Assistant turn: one identity and duration row on top, the trace (thinking + tools) collapses, the final reply always shows.
    fn turn_view(
        &self,
        all: &[&ChatMessage],
        indices: &[usize],
        user: Option<&ChatMessage>,
        turn_key: usize,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let key = format!("turn-{turn_key}");
        // The turn's answer is its last assistant block; earlier assistant blocks are process notes
        // streamed between tool calls and belong to the trace, not to the reply.
        let reply = indices
            .iter()
            .rposition(|index| all[*index].role == MessageRole::Assistant);
        // The duration row toggles the run trace (thinking, tools, process notes); the final reply always shows.
        let has_trace =
            indices
                .iter()
                .enumerate()
                .any(|(position, index)| match all[*index].role {
                    MessageRole::Tool | MessageRole::Thinking => true,
                    MessageRole::Assistant => Some(position) != reply,
                    _ => false,
                });
        let running = user
            .and_then(|message| message.state)
            .map(|state| !state.is_terminal())
            .unwrap_or(false);
        // Once the final reply arrives the trace collapses; a manual toggle wins afterwards.
        let finished = indices
            .iter()
            .any(|index| all[*index].role == MessageRole::Assistant);
        let collapsed = has_trace
            && self
                .trace_open
                .get(&key)
                .map(|open| !*open)
                .unwrap_or(finished && !running);
        let earliest = indices.iter().map(|index| all[*index].at).min();
        let latest = indices.iter().map(|index| all[*index].at).max();
        let seconds = match (earliest, latest) {
            (Some(earliest), Some(latest)) => {
                if running {
                    Local::now()
                        .signed_duration_since(earliest)
                        .num_seconds()
                        .max(0)
                } else {
                    latest.signed_duration_since(earliest).num_seconds().max(0)
                }
            }
            _ => 0,
        };
        let label = if running {
            format!("Working · {}", elapsed_label(seconds))
        } else if seconds >= 1 {
            format!("Took {}", elapsed_label(seconds))
        } else {
            String::new()
        };

        // The head keeps one row, "took ...": the text itself is the toggle (up to expand, down to collapse);
        // plain text when there is no trace; the row is omitted when the duration is under 1 second.
        let has_meta = has_trace || !label.is_empty();
        let mut meta = h_flex().w_full().gap_3().items_center();
        if has_trace {
            let toggle = key.clone();
            let text = if label.is_empty() {
                "Trace".to_string()
            } else {
                label.clone()
            };
            meta = meta.child(
                h_flex()
                    .id(format!("turn-trace-{turn_key}"))
                    .gap_1()
                    .items_center()
                    .py_1()
                    .cursor_pointer()
                    .hover(|style| style.opacity(0.7))
                    .child(div().text_xs().text_color(rgb(MUTED)).child(text))
                    .child(
                        Icon::new(if collapsed {
                            Lucide::ChevronDown
                        } else {
                            Lucide::ChevronUp
                        })
                        .with_size(px(12.))
                        .text_color(rgb(FAINT)),
                    )
                    .on_click(cx.listener(move |this, _, _, cx| {
                        // Collapsed opens, expanded closes, and the choice is remembered over the default.
                        this.trace_open.insert(toggle.clone(), collapsed);
                        cx.notify();
                    })),
            );
        } else if !label.is_empty() {
            meta = meta.child(div().text_xs().text_color(rgb(FAINT)).child(label));
        }

        let mut column = v_flex().w_full().min_w(px(0.)).gap_3();
        if has_meta {
            column = column.child(meta);
        }

        let mut body = v_flex().w_full().min_w(px(0.)).gap_4();
        let mut index = 0;
        while index < indices.len() {
            let message = all[indices[index]];
            // With the trace collapsed, thinking, tools and process notes are skipped; only the final reply renders.
            if collapsed && matches!(message.role, MessageRole::Tool | MessageRole::Thinking) {
                index += 1;
                continue;
            }
            if collapsed && message.role == MessageRole::Assistant && reply != Some(index) {
                index += 1;
                continue;
            }
            match message.role {
                MessageRole::Tool | MessageRole::Thinking => {
                    // Thinking and tools belong to one run trace: they merge into the same tool group in order.
                    let start = index;
                    while index < indices.len()
                        && matches!(
                            all[indices[index]].role,
                            MessageRole::Tool | MessageRole::Thinking
                        )
                    {
                        index += 1;
                    }
                    body = body.child(self.tool_group(all, &indices[start..index], turn_key, cx));
                }
                MessageRole::Assistant => {
                    body = body.child(assistant_view(message, &format!("{turn_key}-{index}")));
                    index += 1;
                }
                MessageRole::Error => {
                    body = body.child(error_view(message));
                    index += 1;
                }
                MessageRole::Notice => {
                    if message.is_divider() {
                        body = body.child(divider_view(&message.body));
                    }
                    index += 1;
                }
                _ => {
                    index += 1;
                }
            }
        }
        column = column.child(body);
        column.into_any_element()
    }

    /// Consecutive tool calls collapse into a group: one summary row plus indented tool rows, not a card per command.
    fn tool_group(
        &self,
        all: &[&ChatMessage],
        indices: &[usize],
        turn_key: usize,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let key = format!("tools-{turn_key}-{}", indices[0]);
        // Tool groups start collapsed: the set holds expanded groups, empty for a new session.
        let expanded = self.expanded_tools.contains(&key);
        let summary = tool_summary(all, indices);
        // With only thinking and no tool calls use the brain icon so thinking is not read as commands.
        let only_thinking = indices
            .iter()
            .all(|index| all[*index].role == MessageRole::Thinking);
        let toggle = key.clone();
        // Tool groups get no fill or border: a light header row, expanded steps separated by indentation.
        let mut group = v_flex()
            .w_full()
            .min_w(px(0.))
            .rounded_xl()
            .overflow_hidden();
        group = group.child(
            h_flex()
                .id(format!("tool-group-{key}"))
                .w_full()
                .gap_2()
                .items_center()
                .px_3()
                .py_2()
                .cursor_pointer()
                .hover(|style| style.bg(rgb(HOVER)))
                .child(
                    glyph(if only_thinking {
                        Lucide::Brain
                    } else {
                        Lucide::Terminal
                    })
                    .with_size(px(13.))
                    .text_color(rgb(FAINT)),
                )
                .child(
                    div()
                        .flex_1()
                        .min_w(px(0.))
                        .overflow_hidden()
                        .truncate()
                        .text_xs()
                        .text_color(rgb(BODY))
                        .child(summary),
                )
                .child(
                    div()
                        .flex_shrink_0()
                        .text_xs()
                        .text_color(rgb(FAINT))
                        .child(format!("{} steps", indices.len())),
                )
                .child(
                    Icon::new(if expanded {
                        Lucide::ChevronDown
                    } else {
                        Lucide::ChevronRight
                    })
                    .with_size(px(12.))
                    .text_color(rgb(FAINT)),
                )
                .on_click(cx.listener(move |this, _, _, cx| {
                    if !this.expanded_tools.remove(&toggle) {
                        this.expanded_tools.insert(toggle.clone());
                    }
                    cx.notify();
                })),
        );
        if expanded {
            let mut rows = v_flex()
                .w_full()
                .min_w(px(0.))
                .gap_0p5()
                .px_2()
                .pb_2()
                .pt_1();
            // Thinking rows and tool rows share the group, keeping the model's real order.
            for index in indices {
                let message = all[*index];
                rows = rows.child(if message.role == MessageRole::Thinking {
                    self.thinking_row(message, *index, turn_key, cx)
                } else {
                    self.tool_row(message, cx)
                });
            }
            // Cap the height and scroll inside when there are many steps, so a long tool list does not push the reply away.
            group = group.child(
                div()
                    .w_full()
                    .min_w(px(0.))
                    .max_h(px(320.))
                    .border_t_1()
                    .border_color(rgb(HAIRLINE))
                    .overflow_y_scrollbar()
                    .id(format!("tool-rows-{key}"))
                    .child(rows),
            );
        }
        group.into_any_element()
    }

    /// Single tool row: icon plus a short action (commands keep their text); click to expand the output tail.
    fn tool_row(&self, message: &ChatMessage, cx: &mut Context<Self>) -> AnyElement {
        let id = message.id;
        let expanded = self.expanded_outputs.contains(&id);
        let (kind, label) = tool_label(&message.body);
        let meta = message.tool.as_ref();
        let has_output = meta
            .map(|meta| !meta.output.trim().is_empty())
            .unwrap_or(false);
        let mut row = h_flex()
            .id(format!("tool-{id}"))
            .w_full()
            .min_w(px(0.))
            .gap_2()
            .items_center()
            .px_2()
            .py_1()
            .rounded_md()
            .cursor_pointer()
            .hover(|style| style.bg(rgb(HOVER)))
            .child(
                Icon::new(tool_icon(kind))
                    .with_size(px(12.))
                    .text_color(rgb(FAINT)),
            )
            .child(div().flex_1().min_w(px(0.)).overflow_hidden().child(label))
            .on_click(cx.listener(move |this, _, _, cx| {
                if !has_output {
                    return;
                }
                if !this.expanded_outputs.remove(&id) {
                    this.expanded_outputs.insert(id);
                }
                cx.notify();
            }));
        if has_output {
            row = row.child(
                Icon::new(if expanded {
                    Lucide::ChevronDown
                } else {
                    Lucide::ChevronRight
                })
                .with_size(px(12.))
                .text_color(rgb(FAINT)),
            );
        }
        if let Some(meta) = meta {
            if !meta.done {
                row = row.child(status_chip("Running", WARN_TEXT, WARN_BG));
            } else if meta.timed_out {
                row = row.child(status_chip("Timed out", ERROR_TEXT, ERROR_BG));
            } else if let Some(code) = meta.exit_code {
                if code != 0 {
                    row = row.child(status_chip(&format!("exit {code}"), ERROR_TEXT, ERROR_BG));
                }
            }
        }

        let mut column = v_flex().w_full().min_w(px(0.)).gap_1().child(row);
        if expanded {
            if let Some(meta) = meta {
                if !meta.output.trim().is_empty() {
                    column = column.child(output_block(&meta.output));
                }
            }
        }
        column.into_any_element()
    }

    /// Thinking is a row in the tool group: collapsed by default, expanding shows the text below it.
    fn thinking_row(
        &self,
        message: &ChatMessage,
        index: usize,
        turn_key: usize,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let key = format!("think-{turn_key}-{index}");
        let expanded = self.expanded_thinking.contains(&key);
        let chars = message.body.chars().count();
        let toggle = key.clone();
        let mut column = v_flex().w_full().min_w(px(0.)).gap_1().child(
            h_flex()
                .id(format!("thinking-{key}"))
                .w_full()
                .min_w(px(0.))
                .gap_2()
                .items_center()
                .px_2()
                .py_1()
                .rounded_md()
                .cursor_pointer()
                .hover(|style| style.bg(rgb(HOVER)))
                .child(
                    glyph(Lucide::Brain)
                        .with_size(px(12.))
                        .text_color(rgb(FAINT)),
                )
                .child(div().text_xs().text_color(rgb(MUTED)).child("Thinking"))
                .child(
                    div()
                        .text_xs()
                        .text_color(rgb(FAINT))
                        .child(format!("· {chars} chars")),
                )
                .child(div().flex_1())
                .child(
                    Icon::new(if expanded {
                        Lucide::ChevronDown
                    } else {
                        Lucide::ChevronRight
                    })
                    .with_size(px(12.))
                    .text_color(rgb(FAINT)),
                )
                .on_click(cx.listener(move |this, _, _, cx| {
                    if !this.expanded_thinking.remove(&toggle) {
                        this.expanded_thinking.insert(toggle.clone());
                    }
                    cx.notify();
                })),
        );
        if expanded {
            // Thinking text opens against the left rule and still respects the group scroll height.
            column = column.child(
                div()
                    .w_full()
                    .min_w(px(0.))
                    .pl_3()
                    .pb_1()
                    .border_l_1()
                    .border_color(rgb(ACCENT_LINE))
                    .child(text_lines(&message.body, MUTED, false)),
            );
        }
        column.into_any_element()
    }
}

/// All lucide icons live in the assets directory; the built-in Icon::new knows only a subset.
fn glyph(name: Lucide) -> Icon {
    Icon::default().path(name.path())
}

/// Top bar icon button: 32px rounded square with a tint on hover.
fn elapsed_label(seconds: i64) -> String {
    if seconds < 60 {
        format!("{seconds}s")
    } else {
        format!("{}m {}s", seconds / 60, seconds % 60)
    }
}

/// Small status chip on a tool row.
fn status_chip(label: &str, fg: u32, bg: u32) -> AnyElement {
    div()
        .flex_shrink_0()
        .px_2()
        .py_0p5()
        .rounded_full()
        .bg(rgb(bg))
        .text_xs()
        .text_color(rgb(fg))
        .child(label.to_string())
        .into_any_element()
}

/// Relative time shown for a session in the sidebar.
fn session_stamp(at: &DateTime<chrono::Utc>) -> String {
    let local = at.with_timezone(&Local);
    let today = Local::now().date_naive();
    let day = local.date_naive();
    if day == today {
        local.format("%H:%M").to_string()
    } else if Some(day) == today.pred_opt() {
        "Yesterday".into()
    } else if day.year() == today.year() {
        day.format("%m/%d").to_string()
    } else {
        day.format("%y/%m/%d").to_string()
    }
}

/// Date divider: a centered short label with hairlines on both sides.
fn divider_view(label: &str) -> AnyElement {
    h_flex()
        .w_full()
        .items_center()
        .gap_4()
        .py_2()
        .child(div().flex_1().h(px(1.)).bg(rgb(HAIRLINE)))
        .child(
            div()
                .flex_shrink_0()
                .text_xs()
                .text_color(rgb(FAINT))
                .child(label.to_string()),
        )
        .child(div().flex_1().h(px(1.)).bg(rgb(HAIRLINE)))
        .into_any_element()
}

/// Assistant reply: Markdown (inline code, bold, lists, code blocks) without avatar or bubble.
fn assistant_view(message: &ChatMessage, key: &str) -> AnyElement {
    v_flex()
        .w_full()
        .min_w(px(0.))
        .text_base()
        .text_color(rgb(INK))
        .line_height(rems(1.75))
        .child(
            TextView::markdown(
                ElementId::Name(format!("md-{key}").into()),
                message.body.clone(),
            )
            .selectable(true)
            .scrollable(false),
        )
        .into_any_element()
}

fn error_view(message: &ChatMessage) -> AnyElement {
    h_flex()
        .w_full()
        .min_w(px(0.))
        .items_start()
        .gap_2()
        .px_3()
        .py_2()
        .rounded_lg()
        .bg(rgb(ERROR_BG))
        .border_1()
        .border_color(rgb(ERROR_LINE))
        .child(
            glyph(Lucide::CircleAlert)
                .with_size(px(14.))
                .text_color(rgb(ERROR_TEXT)),
        )
        .child(
            v_flex()
                .flex_1()
                .min_w(px(0.))
                .child(text_lines(&message.body, ERROR_TEXT, false)),
        )
        .into_any_element()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ToolKind {
    Read,
    List,
    Search,
    Image,
    Call,
    Command,
}

fn tool_icon(kind: ToolKind) -> Lucide {
    match kind {
        ToolKind::Read => Lucide::FileText,
        ToolKind::List => Lucide::FolderOpen,
        ToolKind::Search => Lucide::Search,
        ToolKind::Image => Lucide::Image,
        ToolKind::Call => Lucide::Bot,
        ToolKind::Command => Lucide::Terminal,
    }
}

/// A tool message body looks like `call read_file {"path":"src/lib.rs"}` or a raw bash command.
fn call_parts(script: &str) -> Option<(&str, &str)> {
    let rest = script.trim_start().strip_prefix("call ")?;
    let rest = rest.lines().next().unwrap_or(rest).trim();
    Some(match rest.split_once(' ') {
        Some((name, args)) => (name, args),
        None => (rest, ""),
    })
}

fn tool_kind(script: &str) -> ToolKind {
    match call_parts(script).map(|(name, _)| name) {
        Some("read_file") => ToolKind::Read,
        Some("list_files") => ToolKind::List,
        Some("read_image") => ToolKind::Image,
        Some("search") => ToolKind::Search,
        Some(_) => ToolKind::Call,
        None => ToolKind::Command,
    }
}

fn tool_label(script: &str) -> (ToolKind, AnyElement) {
    let Some((name, args)) = call_parts(script) else {
        let line = script.lines().next().unwrap_or("").trim();
        return (
            ToolKind::Command,
            div()
                .font_family("Menlo")
                .text_xs()
                .text_color(rgb(BODY))
                .truncate()
                .child(shorten(line, 96))
                .into_any_element(),
        );
    };

    let view = match name {
        "read_file" => tool_text(
            "Read",
            json_field(args, "path").unwrap_or_else(|| "file".into()),
        ),
        "list_files" => tool_text(
            "Listed",
            format!(
                "files in {}",
                json_field(args, "path").unwrap_or_else(|| "folder".into())
            ),
        ),
        "read_image" => tool_text(
            "Viewed",
            json_field(args, "path").unwrap_or_else(|| "image".into()),
        ),
        "search" => tool_text(
            "Searched",
            json_field(args, "query").unwrap_or_else(|| "content".into()),
        ),
        "submit_result" => div()
            .text_xs()
            .text_color(rgb(BODY))
            .child("Submitted result")
            .into_any_element(),
        _ => tool_text("Called", format!("{name} {}", shorten(args, 48))),
    };
    (tool_kind(script), view)
}

fn tool_text(verb: &str, target: String) -> AnyElement {
    h_flex()
        .gap_1()
        .min_w(px(0.))
        .child(
            div()
                .text_xs()
                .text_color(rgb(MUTED))
                .child(verb.to_string()),
        )
        .child(
            div()
                .font_family("Menlo")
                .text_xs()
                .text_color(rgb(BODY))
                .truncate()
                .child(target),
        )
        .into_any_element()
}

fn tool_summary(all: &[&ChatMessage], indices: &[usize]) -> String {
    let (mut reads, mut lists, mut searches, mut images, mut calls, mut commands, mut thoughts) =
        (0, 0, 0, 0, 0, 0, 0);
    for index in indices {
        let message = all[*index];
        if message.role == MessageRole::Thinking {
            thoughts += 1;
            continue;
        }
        match tool_kind(&message.body) {
            ToolKind::Read => reads += 1,
            ToolKind::List => lists += 1,
            ToolKind::Search => searches += 1,
            ToolKind::Image => images += 1,
            ToolKind::Call => calls += 1,
            ToolKind::Command => commands += 1,
        }
    }
    let mut parts = Vec::new();
    if thoughts > 0 {
        parts.push(format!("{thoughts} thoughts"));
    }
    if reads > 0 {
        parts.push(format!("{reads} files read"));
    }
    if lists > 0 {
        parts.push(format!("{lists} folders listed"));
    }
    if searches > 0 {
        parts.push(format!("{searches} searches"));
    }
    if images > 0 {
        parts.push(format!("{images} images viewed"));
    }
    if commands > 0 {
        parts.push(format!("{commands} commands run"));
    }
    if calls > 0 {
        parts.push(format!("{calls} tool calls"));
    }
    if parts.is_empty() {
        "Run log".into()
    } else {
        parts.join(" · ")
    }
}

/// Tool output tail: dark terminal block, the last 30 lines at most.
fn output_block(output: &str) -> AnyElement {
    let tail: Vec<&str> = output.lines().rev().take(30).collect();
    let text = tail.into_iter().rev().collect::<Vec<_>>().join("\n");
    v_flex()
        .w_full()
        .min_w(px(0.))
        .pl_2()
        .pb_1()
        .child(
            div()
                .w_full()
                .min_w(px(0.))
                .px_3()
                .py_2()
                .rounded_lg()
                .bg(rgb(TERMINAL_BG))
                .overflow_hidden()
                .child(
                    div()
                        .font_family("Menlo")
                        .text_size(px(12.))
                        .line_height(rems(1.55))
                        .text_color(rgb(TERMINAL_TEXT))
                        .whitespace_normal()
                        .child(text),
                ),
        )
        .into_any_element()
}

/// Read a string field from the JSON snippet; fall back to the default text when args are truncated.
fn json_field(args: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let start = args.find(&needle)? + needle.len();
    let rest = &args[start..];
    let colon = rest.find(':')?;
    let rest = &rest[colon + 1..];
    let quote = rest.find('"')?;
    let tail = &rest[quote + 1..];
    let end = tail.find('"')?;
    Some(tail[..end].to_string())
}

/// Token count abbreviation: `980` / `12.4k` / `128k`.
fn fmt_tokens(value: u32) -> String {
    if value < 1_000 {
        value.to_string()
    } else if value < 10_000 {
        format!("{:.1}k", value as f32 / 1_000.0)
    } else {
        format!("{}k", value / 1_000)
    }
}

fn shorten(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        text.to_string()
    } else {
        format!("{}…", text.chars().take(max).collect::<String>())
    }
}

fn text_lines(body: &str, color: u32, mono: bool) -> AnyElement {
    let mut column = v_flex().w_full().min_w(px(0.)).gap_1();
    let mut wrote = false;
    for line in body.lines() {
        if line.trim().is_empty() {
            column = column.child(div().h(px(6.)));
            continue;
        }
        wrote = true;
        let row = div()
            .w_full()
            .min_w(px(0.))
            .whitespace_normal()
            .text_color(rgb(color))
            .child(line.to_string());
        let row = if mono {
            row.font_family("Menlo").text_xs()
        } else {
            row.text_sm().line_height(rems(1.6))
        };
        column = column.child(row);
    }
    if !wrote {
        column = column.child(div().text_sm().text_color(rgb(color)).child("(empty)"));
    }
    column.into_any_element()
}

fn day_label(at: &DateTime<Local>) -> String {
    let today = Local::now().date_naive();
    let day = at.date_naive();
    if day == today {
        "Today".into()
    } else if Some(day) == today.pred_opt() {
        "Yesterday".into()
    } else if day.year() == today.year() {
        day.format("%b %-d").to_string()
    } else {
        day.format("%b %-d, %Y").to_string()
    }
}

impl Render for ConversationView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        if self.pending_scroll {
            self.pending_scroll = false;
            // Wait at least two frames for layout to settle: streaming text changes height,
            // scrolling at once stops halfway, and the scroll height can still be 0 on first open.
            let scroll = self.scroll.clone();
            cx.spawn(async move |this, cx| {
                cx.background_executor()
                    .timer(Duration::from_millis(120))
                    .await;
                // Set a large negative offset directly; layout clamps it to the real limit;
                // scroll_to_bottom is only a flag and does nothing without a repaint.
                scroll.set_offset(point(px(0.), px(-1_000_000.)));
                let _ = this.update(cx, |_, cx| cx.notify());
            })
            .detach();
        }
        let _ = cx;
        h_flex()
            .track_focus(&self.focus_handle)
            .size_full()
            .bg(rgb(CANVAS))
            .child(self.sidebar(cx))
            .child(
                v_flex()
                    .flex_1()
                    .min_w(px(0.))
                    .h_full()
                    .child(self.header())
                    .child(self.message_area(cx))
                    .child(self.composer_bar(cx)),
            )
    }
}

pub fn open_conversation_window(cx: &mut App, ctx: Arc<DesktopContext>) {
    let mut slot = CONVERSATION_WINDOW.lock().unwrap();
    if let Some(handle) = slot.as_ref() {
        if handle
            .update(cx, |_, window, _| window.activate_window())
            .is_ok()
        {
            cx.activate(true);
            return;
        }
        *slot = None;
    }

    let options = WindowOptions {
        window_bounds: Some(WindowBounds::centered(size(px(1320.), px(840.)), cx)),
        titlebar: Some(TitlebarOptions {
            title: Some("Relay · Sessions".into()),
            appears_transparent: false,
            traffic_light_position: None,
        }),
        ..Default::default()
    };
    let env_session = std::env::var("RELAY_OPEN_SESSION").ok();
    match cx.open_window(options, move |window, cx| {
        let view = cx.new(|cx| ConversationView::new(window, cx, ctx));
        if let Some(raw) = env_session.as_deref() {
            view.update(cx, |view, cx| view.open_by_id(raw, cx));
        }
        cx.new(|cx| Root::new(view, window, cx))
    }) {
        Ok(handle) => {
            let _ = handle.update(cx, |_, window, _| window.activate_window());
            cx.activate(true);
            *slot = Some(handle);
        }
        Err(err) => tracing::error!(%err, "failed to create the session window"),
    }
}
