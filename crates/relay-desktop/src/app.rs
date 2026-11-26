//! GPUI desktop shell: minimal hotkey-summoned panel + confirmation card + management center + session window.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::input::{Input, InputEvent, InputState, Textarea, TextareaState};
use gpui_kit::component::scroll::ScrollableElement as _;
use gpui_kit::component::speech::{SpeechEvent, SpeechIndicator, SpeechState, SpeechStatus};
use gpui_kit::component::Icon;
use gpui_kit::component::{h_flex, v_flex, Root, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::FontWeight;
use gpui_kit::*;
use relay_core::{Plan, RuntimeEvent, RuntimeEventKind, TaskId, TaskState};
use relay_openai::AsrClient;
use relay_runtime::{Confirmation, Runtime, TaskRequest};
use tokio::sync::broadcast;

use crate::config::AppConfig;
use crate::voice::{DeepSeekAsr, TRANSCRIBE_TIMEOUT};

actions!(relay_panel, [ClosePanel]);

static PANEL_OPEN: AtomicBool = AtomicBool::new(false);

/// The live panel window. `PANEL_OPEN` on its own can outlive the window, and a stale `true`
/// swallows every later summon — the hotkey then looks dead.
static PANEL_WINDOW: Mutex<Option<WindowHandle<Root>>> = Mutex::new(None);

/// Raise the panel if its window still exists; report false when the window is gone.
///
/// Membership in `App::windows()` is the real test: a handle whose window was removed can still
/// accept an update, which would keep `PANEL_OPEN` stuck at `true` and swallow every summon.
fn panel_window_alive(cx: &mut App) -> bool {
    let mut slot = PANEL_WINDOW.lock().unwrap();
    let Some(handle) = slot.as_ref() else {
        return false;
    };
    let id = handle.window_id();
    if !cx.windows().iter().any(|window| window.window_id() == id) {
        *slot = None;
        return false;
    }
    let _ = handle.update(cx, |_, window, _| window.activate_window());
    true
}

/// The management center is a single-instance window: repeat clicks only raise the existing one.
static MANAGEMENT_WINDOW: Mutex<Option<WindowHandle<Root>>> = Mutex::new(None);

/// One screenshot attached to the compose box: the cropped, annotated image on disk.
#[derive(Debug, Clone)]
pub struct SnipAttachment {
    pub path: PathBuf,
    pub width: u32,
    pub height: u32,
}

/// What survives a panel close/reopen: the draft instruction, attached screenshots and the last notice.
///
/// The compose panel is destroyed while the screenshot editor is open, so the draft lives in the
/// desktop context instead of the panel entity.
#[derive(Debug, Clone, Default)]
pub struct PanelDraft {
    pub instruction: String,
    pub attachments: Vec<SnipAttachment>,
    pub notice: Option<String>,
}

pub struct DesktopContext {
    pub runtime: Runtime,
    pub tokio: tokio::runtime::Handle,
    pub config: AppConfig,
    pub demo: bool,
    pub hotkey_error: Option<String>,
    pub draft: Mutex<PanelDraft>,
    /// Where the user last put the annotation toolbar, so the next capture starts there instead of
    /// covering the same part of the screen again.
    pub snip_toolbar: Mutex<Option<(f32, f32)>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Compose,
    Submitting,
    Confirming,
    Running,
    Paused,
    Done,
    Cancelled,
    Failed,
}

/// Fixed width; height hugs the content per state so Ready does not leave a big blank area.
const PANEL_WIDTH: f32 = 720.;

/// Thumbnails rendered in the compose row; the rest are counted in a "+N" note.
const MAX_TILES: usize = 3;

/// Window height per state (padding included; overflowing content scrolls inside).
fn panel_height(mode: Mode) -> f32 {
    match mode {
        Mode::Compose => 272.,
        Mode::Submitting | Mode::Running => 330.,
        Mode::Confirming => 560.,
        Mode::Paused => 250.,
        Mode::Done | Mode::Cancelled | Mode::Failed => 300.,
    }
}

/// Thumbnail cache entry for one attached screenshot: decoded once, off the UI thread.
enum Thumb {
    Loading,
    Ready(Arc<RenderImage>),
    Failed,
}

/// Small paintable preview of a screenshot attachment.
///
/// GPUI samples textures as BGRA — the same swap the full-size capture path performs — otherwise
/// every thumbnail would come out with red and blue traded.
fn attachment_thumbnail(path: &std::path::Path) -> Option<Arc<RenderImage>> {
    let preview = image::open(path).ok()?.thumbnail(168, 120);
    let (width, height) = (preview.width(), preview.height());
    let mut rgba = preview.into_rgba8().into_raw();
    for pixel in rgba.chunks_exact_mut(4) {
        pixel.swap(0, 2);
    }
    let pixels = image::RgbaImage::from_raw(width, height, rgba)?;
    Some(Arc::new(RenderImage::new(vec![image::Frame::new(pixels)])))
}

struct Panel {
    ctx: Arc<DesktopContext>,
    input: Entity<InputState>,
    /// Editable fields on the confirmation card: title, summary, steps.
    title_input: Entity<InputState>,
    summary_input: Entity<TextareaState>,
    steps_input: Entity<TextareaState>,
    /// Feedback input for "Let AI revise".
    revise_input: Entity<InputState>,
    /// After a plan update the inputs must be synced (done in render, where `window` is available).
    sync_inputs: bool,
    input_focus: FocusHandle,
    focus_handle: FocusHandle,
    focus_input: bool,
    mode: Mode,
    task_id: Option<TaskId>,
    plan: Option<Plan>,
    state_label: String,
    context_line: String,
    permissions: String,
    references: Vec<String>,
    route_line: String,
    /// Tool trail from understanding (last 8 entries) so the user sees what it is checking.
    trail: Vec<String>,
    log: Vec<String>,
    summary: String,
    error: Option<String>,
    /// Hold-to-talk: the microphone, the level meter and the transcript of the current press.
    voice: Entity<SpeechState>,
    /// Why the last press produced no text: no key, no permission, or the service refused.
    voice_notice: Option<String>,
    /// Screenshots attached to this instruction (cropped, annotated desktop captures).
    attachments: Vec<SnipAttachment>,
    /// Decoded previews for `attachments`, keyed by file path.
    thumbs: HashMap<PathBuf, Thumb>,
    /// Window height currently applied; only adjusted on state changes to avoid resizing every frame.
    panel_height: f32,
}

impl Panel {
    fn new(
        window: &mut Window,
        cx: &mut Context<Self>,
        ctx: Arc<DesktopContext>,
        resume_confirmation: bool,
    ) -> Self {
        // Restore the draft: the panel is recreated after the screenshot editor closes.
        let draft = ctx
            .draft
            .lock()
            .map(|draft| draft.clone())
            .unwrap_or_default();
        let input = cx.new(|cx| {
            InputState::new(window, cx)
                .placeholder("Hand it over: files, forms, code… (optional, I read your screen)")
        });
        if !draft.instruction.trim().is_empty() {
            input.update(cx, |input, cx| {
                input.set_value(draft.instruction.clone(), window, cx)
            });
        }
        let title_input =
            cx.new(|cx| InputState::new(window, cx).placeholder("What to deliver, in one line"));
        let summary_input = cx.new(|cx| TextareaState::new(window, cx).auto_grow(2, 6));
        let steps_input = cx.new(|cx| TextareaState::new(window, cx).auto_grow(3, 10));
        let revise_input = cx.new(|cx| {
            InputState::new(window, cx).placeholder(
                "Not right? Say what to change, e.g. checklist only, leave files as-is",
            )
        });
        cx.subscribe_in(
            &revise_input,
            window,
            |this, _input, event: &InputEvent, window, cx| {
                if matches!(event, InputEvent::PressEnter { .. }) {
                    this.revise(window, cx);
                }
            },
        )
        .detach();
        let focus_handle = cx.focus_handle();
        let input_focus = input.read(cx).focus_handle(cx);
        let _ = &input_focus;
        cx.subscribe_in(
            &input,
            window,
            |this, _input, event: &InputEvent, window, cx| {
                if matches!(event, InputEvent::PressEnter { .. }) {
                    this.submit(window, cx);
                }
            },
        )
        .detach();
        input_focus.focus(window, cx);

        // Dictation reuses the model key and base URL: the ASR endpoint sits next to the model API.
        let asr = ctx
            .config
            .openai_config()
            .map(|openai| AsrClient::new(openai.api_key, openai.base_url));
        let voice = cx.new(|cx| {
            SpeechState::new(cx)
                .recognizer(DeepSeekAsr::new(asr, ctx.tokio.clone()))
                .system_fallback(false)
                .stop_timeout(TRANSCRIBE_TIMEOUT)
        });
        cx.subscribe_in(
            &voice,
            window,
            |this, _voice, event: &SpeechEvent, window, cx| this.on_speech(event, window, cx),
        )
        .detach();

        let mut understanding = ctx.runtime.subscribe_understanding();
        cx.spawn(async move |this, cx| loop {
            match understanding.recv().await {
                Ok(event) => {
                    if this
                        .update(cx, |this, cx| {
                            this.trail.push(event.summary.clone());
                            if this.trail.len() > 8 {
                                this.trail.remove(0);
                            }
                            cx.notify();
                        })
                        .is_err()
                    {
                        break;
                    }
                }
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(_) => break,
            }
        })
        .detach();

        let mut events = ctx.runtime.subscribe();
        cx.spawn(async move |this, cx| loop {
            match events.recv().await {
                Ok(event) => {
                    if this
                        .update(cx, |this, cx| {
                            this.on_event(event, cx);
                            cx.notify();
                        })
                        .is_err()
                    {
                        break;
                    }
                }
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(_) => break,
            }
        })
        .detach();

        let mut panel = Self {
            ctx,
            input,
            title_input,
            summary_input,
            steps_input,
            revise_input,
            sync_inputs: false,
            input_focus,
            focus_handle,
            focus_input: false,
            mode: Mode::Compose,
            task_id: None,
            plan: None,
            state_label: "Ready".into(),
            context_line: String::new(),
            permissions: String::new(),
            references: Vec::new(),
            route_line: String::new(),
            trail: Vec::new(),
            log: Vec::new(),
            summary: String::new(),
            error: None,
            voice,
            voice_notice: None,
            attachments: draft.attachments,
            thumbs: HashMap::new(),
            panel_height: 0.,
        };
        // The panel hides while understanding/running; when it is summoned back for confirmation,
        // restore the confirmation card, otherwise the user only sees an empty input.
        if resume_confirmation {
            if let Ok(Some(task)) = panel.ctx.runtime.awaiting_confirmation() {
                panel.task_id = Some(task.id);
                if let Some(plan) = task.plan {
                    panel.summary = plan.title.clone();
                    panel.plan = Some(plan);
                    panel.sync_inputs = true;
                }
                panel.context_line = task.context.summary_line();
                panel.state_label = "Awaiting confirmation".into();
                panel.mode = Mode::Confirming;
                panel.focus_input = true;
            }
        }
        panel
    }

    fn close(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        // A press belongs to the panel that opened it: closing mid-recording discards it.
        self.voice.update(cx, |speech, cx| speech.cancel(cx));
        PANEL_OPEN.store(false, Ordering::SeqCst);
        if let Ok(mut slot) = PANEL_WINDOW.lock() {
            *slot = None;
        }
        window.remove_window();
    }

    fn submit(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.mode == Mode::Confirming {
            self.approve(window, cx);
            return;
        }

        if self.mode != Mode::Compose {
            return;
        }
        // Empty input is valid too: the understanding agent then infers the goal from the screen.
        let instruction = self.input.read(cx).value().trim().to_string();
        let inferred = instruction.is_empty();

        self.input
            .update(cx, |input, cx| input.set_value("", window, cx));
        self.mode = Mode::Submitting;
        self.state_label = if inferred {
            "Read screen and infer".into()
        } else {
            "Understanding".into()
        };
        self.error = None;
        self.trail.clear();
        cx.notify();

        let ctx = self.ctx.clone();
        let (tx, rx) = tokio::sync::oneshot::channel();
        let handle = ctx.tokio.clone();
        let runtime = ctx.runtime.clone();

        let mut request = TaskRequest::new(instruction);
        request.attachments = self
            .attachments
            .iter()
            .map(|attachment| attachment.path.clone())
            .collect();
        // The draft is consumed by this task; the next panel starts clean.
        if let Ok(mut draft) = self.ctx.draft.lock() {
            *draft = PanelDraft::default();
        }

        // Hide the panel before capturing: otherwise the capture (front window, screenshot) grabs Relay itself.
        self.close(window, cx);

        handle.spawn(async move {
            // Wait until the WindowServer has really hidden the panel before reading the front window and screenshot.
            tokio::time::sleep(std::time::Duration::from_millis(120)).await;
            let result = runtime.submit(request).await;
            let _ = tx.send(result);
        });

        cx.spawn(async move |this, cx| {
            let result = rx.await;
            let _ = this.update(cx, |this, cx| {
                match result {
                    Ok(Ok(task_id)) => {
                        this.task_id = Some(task_id);
                        if this.mode == Mode::Submitting {
                            this.mode = Mode::Running;
                            this.state_label = "Understanding".into();
                        }
                    }
                    Ok(Err(err)) => {
                        this.mode = Mode::Failed;
                        this.error = Some(err.to_string());
                    }
                    Err(_) => {
                        this.mode = Mode::Failed;
                        this.error = Some("Runtime closed".into());
                    }
                }
                cx.notify();
            });
        })
        .detach();
    }

    fn on_event(&mut self, event: RuntimeEvent, cx: &mut Context<Self>) {
        if self.task_id.is_none() {
            self.task_id = Some(event.task_id);
        }
        if self.task_id != Some(event.task_id) {
            return;
        }
        match &event.kind {
            RuntimeEventKind::Routed {
                workspace_name,
                session_title,
                confidence,
                ..
            } => {
                self.route_line = format!(
                    "{workspace_name} / {session_title} · {:.0}%",
                    confidence * 100.0
                );
            }
            RuntimeEventKind::PlanReady { plan } => {
                self.plan = Some(plan.clone());
                self.summary = plan.summary.clone();
                self.state_label = "Awaiting confirmation".into();
                self.sync_inputs = true;
            }
            RuntimeEventKind::StateChanged { to, .. } => {
                self.state_label = to.label().to_string();
                if *to == TaskState::AwaitingConfirmation {
                    self.mode = Mode::Confirming;
                    self.focus_input = true;
                    self.refresh_plan(cx);
                } else if *to == TaskState::Executing {
                    self.mode = Mode::Running;
                } else if *to == TaskState::Paused {
                    self.mode = Mode::Paused;
                }
            }
            RuntimeEventKind::ConfirmationRequested {
                title,
                context_line,
                permissions,
                references,
            } => {
                self.mode = Mode::Confirming;
                self.focus_input = true;
                self.context_line = context_line.clone();
                self.permissions = permissions.clone();
                self.references = references.clone();
                self.summary = title.clone();
                self.refresh_plan(cx);
            }
            RuntimeEventKind::StepStarted { turn, index, .. } => {
                self.mode = Mode::Running;
                self.log
                    .push(format!("Round {turn} · step {index} started"));
                self.trim_log();
            }
            RuntimeEventKind::StepFinished {
                turn,
                index,
                exit_code,
                phase,
                ..
            } => {
                self.log.push(format!(
                    "Round {turn} · step {index} finished · exit={exit_code:?} · {phase}"
                ));
                self.trim_log();
            }
            RuntimeEventKind::Notice { message } => {
                self.log.push(message.clone());
                self.trim_log();
            }
            RuntimeEventKind::Completed { report, .. } => {
                self.mode = Mode::Done;
                self.summary = report.summary.clone();
                self.state_label = "Completed".into();
            }
            RuntimeEventKind::Failed { error } => {
                self.mode = Mode::Failed;
                self.error = Some(error.clone());
                self.state_label = "Failed".into();
            }
            RuntimeEventKind::Cancelled => {
                self.mode = Mode::Cancelled;
                self.state_label = "Cancelled".into();
            }
            _ => {}
        }
    }

    fn refresh_plan(&mut self, _cx: &mut Context<Self>) {
        if let Some(id) = self.task_id {
            if let Ok(Some(task)) = self.ctx.runtime.task(id) {
                self.plan = task.plan;
                if let Some(plan) = &self.plan {
                    self.summary = plan.summary.clone();
                }
                self.context_line = task.context.summary_line();
            }
        }
    }

    /// Push the plan into the confirmation card inputs; synced on every plan change so user edits survive.
    fn sync_plan_inputs(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(plan) = self.plan.clone() else {
            return;
        };
        let steps = plan.steps.join("\n");
        let (title, summary) = (plan.title, plan.summary);
        self.title_input
            .update(cx, |state, cx| state.set_value(title, window, cx));
        self.summary_input
            .update(cx, |state, cx| state.set_value(summary, window, cx));
        self.steps_input
            .update(cx, |state, cx| state.set_value(steps, window, cx));
        self.sync_inputs = false;
    }

    /// Approve: write the edited title/summary/steps back into the plan, then let it run.
    fn approve(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(task_id) = self.task_id else {
            return;
        };
        let Some(mut plan) = self.plan.clone() else {
            return;
        };
        plan.title = self.title_input.read(cx).value().trim().to_string();
        plan.summary = self.summary_input.read(cx).value().trim().to_string();
        plan.steps = self
            .steps_input
            .read(cx)
            .value()
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(str::to_string)
            .collect();
        if let Err(err) = self.ctx.runtime.update_plan(task_id, plan) {
            self.error = Some(err.to_string());
            cx.notify();
            return;
        }
        if let Err(err) = self.ctx.runtime.confirm(task_id, Confirmation::Approve) {
            self.error = Some(err.to_string());
            cx.notify();
            return;
        }
        self.error = None;
        self.close(window, cx);
    }

    /// Let the AI understand again from the user's sentence, then return to the confirmation card.
    fn revise(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(task_id) = self.task_id else {
            return;
        };
        let feedback = self.revise_input.read(cx).value().trim().to_string();
        if feedback.is_empty() {
            self.error =
                Some("Say what to change first, e.g. checklist only, leave files as-is".into());
            cx.notify();
            return;
        }
        if let Err(err) = self
            .ctx
            .runtime
            .confirm(task_id, Confirmation::Revise(feedback))
        {
            self.error = Some(err.to_string());
            cx.notify();
            return;
        }
        self.revise_input
            .update(cx, |input, cx| input.set_value("", window, cx));
        self.error = None;
        self.mode = Mode::Submitting;
        self.state_label = "Revising from your feedback".into();
        self.trail.clear();
        cx.notify();
    }

    fn trim_log(&mut self) {
        if self.log.len() > 6 {
            let drain = self.log.len() - 6;
            self.log.drain(0..drain);
        }
    }

    fn cancel(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        if let Some(id) = self.task_id {
            let _ = self.ctx.runtime.cancel(id);
        }
        self.mode = Mode::Cancelled;
        self.state_label = "Cancelled".into();
        cx.notify();
    }

    fn open_artifact(&mut self, _window: &mut Window, _cx: &mut Context<Self>) {
        if let Some(id) = self.task_id {
            if let Ok(artifacts) = self.ctx.runtime.artifacts(id) {
                if let Some(first) = artifacts.iter().find(|a| a.path.is_file()) {
                    let _ = open::that_detached(&first.path);
                }
            }
        }
    }

    fn header(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let mut right = h_flex().gap_2().items_center();
        if !self.route_line.is_empty() {
            right = right.child(
                div()
                    .text_xs()
                    .text_color(rgb(0x6B7280))
                    .child(self.route_line.clone()),
            );
        }
        right = right.child(
            div()
                .px_2()
                .py_1()
                .rounded_full()
                .bg(rgb(0xF1F3F5))
                .text_xs()
                .text_color(rgb(0x4B5563))
                .child(self.state_label.clone()),
        );
        // In flight the footer only offers Cancel, which kills the task; this one just hides the panel.
        if matches!(
            self.mode,
            Mode::Submitting | Mode::Confirming | Mode::Running | Mode::Paused
        ) {
            right = right.child(
                Button::new("panel-close")
                    .icon(Icon::new(IconName::Minimize))
                    .tooltip("Hide the panel; the task keeps running")
                    .accessibility_label("Hide panel")
                    .compact()
                    .ghost()
                    .on_click(cx.listener(|this, _, window, cx| this.close(window, cx))),
            );
        }
        h_flex()
            .items_center()
            .justify_between()
            .w_full()
            .child(
                div()
                    .text_xl()
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(rgb(0x111827))
                    .child("Hand off the work. Keep your flow."),
            )
            .child(right)
    }

    fn submitting_body(&self) -> AnyElement {
        let mut list = v_flex().gap_1().w_full();
        for line in self.trail.iter().rev().take(6).rev() {
            list = list.child(
                div()
                    .text_xs()
                    .text_color(rgb(0x6B7280))
                    .child(format!("· {line}")),
            );
        }
        v_flex()
            .gap_3()
            .w_full()
            .child(
                h_flex()
                    .gap_2()
                    .items_center()
                    .child(div().size(px(8.)).rounded_full().bg(rgb(0x3B82F6)))
                    .child(
                        div()
                            .text_xl()
                            .font_weight(FontWeight::SEMIBOLD)
                            .child("Understanding…"),
                    ),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(rgb(0x4B5563))
                    .child("Reading the environment and history before the plan."),
            )
            .child(list)
            .into_any_element()
    }

    fn compose_body(&mut self, cx: &mut Context<Self>) -> AnyElement {
        self.load_pending_thumbnails(cx);

        let notice = self
            .ctx
            .draft
            .lock()
            .ok()
            .and_then(|draft| draft.notice.clone());

        // One row: screenshots, the instruction, and capture. An attached capture then reads as part
        // of the sentence it belongs to instead of drifting into a second toolbar row.
        let mut row = h_flex()
            .gap_2()
            .items_center()
            .w_full()
            .min_h(px(76.))
            .rounded_xl()
            .border_1()
            .border_color(rgb(0xE3E6EA))
            .bg(rgb(0xFFFFFF))
            .pl_3()
            .pr_2()
            .py_2();
        let attachments = self.attachments.clone();
        for (index, attachment) in attachments.iter().take(MAX_TILES).enumerate() {
            row = row.child(self.attachment_tile(index, attachment, cx));
        }
        if attachments.len() > MAX_TILES {
            row = row.child(
                div()
                    .flex_shrink_0()
                    .text_xs()
                    .text_color(rgb(0x6B7280))
                    .child(format!("+{}", attachments.len() - MAX_TILES)),
            );
        }
        let row = row
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .h(px(44.))
                    .flex()
                    .items_center()
                    .child(
                        Input::new(&self.input)
                            .appearance(false)
                            .cleanable(true)
                            .text_size(px(15.)),
                    ),
            )
            .child(self.voice_button(cx))
            .child(
                Button::new("attach-screenshot")
                    .icon(Icon::new(IconName::Camera))
                    .tooltip("Screenshot: capture the screen, pick a region, draw on it")
                    .accessibility_label("Attach a screenshot")
                    .ghost()
                    .on_click(cx.listener(|this, _, window, cx| this.start_snip(window, cx))),
            );

        let voice_notice = self.voice_notice.clone();
        let mut body = v_flex().gap_3().w_full().child(row);
        if let Some(notice) = notice {
            body = body.child(div().text_xs().text_color(rgb(0xB45309)).child(notice));
        }
        if let Some(notice) = voice_notice {
            body = body.child(div().text_xs().text_color(rgb(0xB45309)).child(notice));
        }
        let hint = match self.voice.read(cx).status() {
            SpeechStatus::Connecting | SpeechStatus::Recording => "Listening… release to insert",
            SpeechStatus::Stopping => "Transcribing…",
            SpeechStatus::Idle if self.voice.read(cx).is_available(cx) => {
                "Hold the microphone to dictate · Enter to understand · Esc to close"
            }
            SpeechStatus::Idle => "Enter to understand · Esc to close",
        };
        body.child(
            h_flex()
                .justify_between()
                .w_full()
                .text_xs()
                .text_color(rgb(0x9AA0A6))
                .child(self.context_line.clone())
                .child(hint),
        )
        .into_any_element()
    }

    /// Hold-to-talk: pressed it captures, released it inserts the transcript at the caret.
    fn voice_button(&self, cx: &mut Context<Self>) -> AnyElement {
        let status = self.voice.read(cx).status();
        let listening = status.is_capturing();
        let glyph: AnyElement = if listening {
            div()
                .size(px(18.))
                .child(SpeechIndicator::new(&self.voice))
                .into_any_element()
        } else {
            Icon::new(IconName::Mic)
                .with_size(px(18.))
                .text_color(rgb(if status == SpeechStatus::Stopping {
                    0x9AA0A6
                } else {
                    0x4B5563
                }))
                .into_any_element()
        };

        let mut button = div()
            .id("voice-input")
            .flex_shrink_0()
            .size(px(30.))
            .flex()
            .items_center()
            .justify_center()
            .rounded_lg()
            .when(listening, |this| {
                this.bg(rgb(0xFDE8E8)).text_color(rgb(0xDC2626))
            })
            .when(!listening, |this| {
                this.hover(|style| style.bg(rgb(0xF2F3F5)))
            });
        // Without a key there is nothing to transcribe with: the button stays visible but inert.
        if self.voice.read(cx).is_available(cx) {
            button = button
                .cursor(CursorStyle::PointingHand)
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(|this, _: &MouseDownEvent, _, cx| this.start_voice(cx)),
                )
                .on_mouse_up(
                    MouseButton::Left,
                    cx.listener(|this, _: &MouseUpEvent, _, cx| this.stop_voice(cx)),
                )
                // Releasing away from the button still ends the press; otherwise it would run on.
                .on_mouse_up_out(
                    MouseButton::Left,
                    cx.listener(|this, _: &MouseUpEvent, _, cx| this.stop_voice(cx)),
                );
        }
        button.child(glyph).into_any_element()
    }

    /// One attachment in the compose row: the capture itself, with a remove target on top.
    fn attachment_tile(
        &self,
        index: usize,
        attachment: &SnipAttachment,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let mut tile = div()
            .relative()
            .flex_shrink_0()
            .w(px(84.))
            .h(px(60.))
            .rounded_lg()
            .overflow_hidden()
            .border_1()
            .border_color(rgb(0xD7DBE0))
            .bg(rgb(0xF3F4F6));
        tile = match self.thumbs.get(&attachment.path) {
            Some(Thumb::Ready(image)) => tile.child(
                img(image.clone())
                    .absolute()
                    .left(px(0.))
                    .top(px(0.))
                    .size_full()
                    .object_fit(ObjectFit::Cover),
            ),
            _ => tile.child(
                v_flex()
                    .size_full()
                    .items_center()
                    .justify_center()
                    .gap_1()
                    .child(
                        Icon::new(IconName::Image)
                            .with_size(px(16.))
                            .text_color(rgb(0x9AA0A6)),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0x9AA0A6))
                            .child(format!("{}×{}", attachment.width, attachment.height)),
                    ),
            ),
        };
        let path = attachment.path.clone();
        tile.child(
            div()
                .id(("detach-attachment", index))
                .absolute()
                .top(px(3.))
                .right(px(3.))
                .size(px(20.))
                .rounded_full()
                .bg(rgb(0xFFFFFF))
                .border_1()
                .border_color(rgb(0xD7DBE0))
                .flex()
                .items_center()
                .justify_center()
                .cursor_pointer()
                .hover(|style| style.bg(rgb(0xFEE2E2)))
                .on_click(cx.listener(move |this, _, _, cx| {
                    this.attachments.retain(|item| item.path != path);
                    this.thumbs.remove(&path);
                    if let Ok(mut draft) = this.ctx.draft.lock() {
                        draft.attachments = this.attachments.clone();
                    }
                    cx.notify();
                }))
                .child(
                    Icon::new(IconName::X)
                        .with_size(px(11.))
                        .text_color(rgb(0x6B7280)),
                ),
        )
        .into_any_element()
    }

    /// Decode attachments that have no preview yet. A full-size capture takes long enough to stall a
    /// frame, so the work happens on the background executor and lands as an update.
    fn load_pending_thumbnails(&mut self, cx: &mut Context<Self>) {
        let pending: Vec<PathBuf> = self
            .attachments
            .iter()
            .map(|attachment| attachment.path.clone())
            .filter(|path| !self.thumbs.contains_key(path))
            .collect();
        for path in pending {
            self.thumbs.insert(path.clone(), Thumb::Loading);
            cx.spawn(async move |panel, cx| {
                let source = path.clone();
                let decoded = cx
                    .background_executor()
                    .spawn(async move { attachment_thumbnail(&source) })
                    .await;
                let _ = panel.update(cx, |panel, cx| {
                    panel
                        .thumbs
                        .insert(path, decoded.map_or(Thumb::Failed, Thumb::Ready));
                    cx.notify();
                });
            })
            .detach();
        }
    }

    /// Close the panel, capture the desktop, then hand it to the annotation window.
    /// A press on the microphone: capture until the button is released.
    fn start_voice(&mut self, cx: &mut Context<Self>) {
        self.voice_notice = None;
        self.voice.update(cx, |speech, cx| {
            if !speech.status().is_active() {
                speech.start(cx);
            }
        });
        cx.notify();
    }

    /// Letting go ends the press; the recognizer then uploads it and answers with the transcript.
    fn stop_voice(&mut self, cx: &mut Context<Self>) {
        self.voice.update(cx, |speech, cx| {
            if speech.status().is_capturing() {
                speech.stop(cx);
            }
        });
        cx.notify();
    }

    /// The transcript is inserted at the caret, so dictation extends whatever is already typed.
    fn on_speech(&mut self, event: &SpeechEvent, window: &mut Window, cx: &mut Context<Self>) {
        match event {
            SpeechEvent::Final(text) => {
                let text = text.replace(['\n', '\r'], " ").trim().to_string();
                if !text.is_empty() {
                    self.input
                        .update(cx, |input, cx| input.insert(text, window, cx));
                    self.input_focus.focus(window, cx);
                }
            }
            SpeechEvent::Error(error) => self.voice_notice = Some(error.to_string()),
            SpeechEvent::Started | SpeechEvent::Partial(_) | SpeechEvent::Cancelled => {}
        }
        cx.notify();
    }

    fn start_snip(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let instruction = self.input.read(cx).value().to_string();
        if let Ok(mut draft) = self.ctx.draft.lock() {
            draft.instruction = instruction;
            draft.attachments = self.attachments.clone();
            draft.notice = None;
        }
        // Hide the panel first: the capture must not contain Relay itself.
        self.close(window, cx);

        let ctx = self.ctx.clone();
        cx.spawn(async move |_this, cx| {
            let started = std::time::Instant::now();
            // One frame is enough for the WindowServer to order the panel out; the old 160 ms wait
            // was pure latency on every capture.
            cx.background_executor()
                .timer(std::time::Duration::from_millis(60))
                .await;
            let capture = cx
                .background_executor()
                .spawn(async move { crate::snip::capture_desktop() })
                .await;
            tracing::debug!(ms = started.elapsed().as_millis(), "snip: desktop captured");
            cx.update(|cx| match capture {
                Ok(image) => crate::snip::open_snip(cx, ctx.clone(), image),
                Err(err) => {
                    if let Ok(mut draft) = ctx.draft.lock() {
                        draft.notice = Some(format!("Screenshot failed: {err}"));
                    }
                    open_panel_after_snip(cx, ctx.clone());
                }
            });
        })
        .detach();
    }

    fn confirm_body(&self, cx: &mut Context<Self>) -> AnyElement {
        let Some(plan) = &self.plan else {
            return div().child("Preparing the plan…").into_any_element();
        };
        let mut body = v_flex()
            .gap_3()
            .w_full()
            .child(
                h_flex()
                    .items_center()
                    .justify_between()
                    .w_full()
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0x9AA0A6))
                            .child("Understanding · edit anything"),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0x9AA0A6))
                            .child("Edit, then hit Approve"),
                    ),
            )
            .child(field_box(
                "Title",
                Input::new(&self.title_input)
                    .appearance(false)
                    .into_any_element(),
            ))
            .child(field_box(
                "Goal",
                Textarea::new(&self.summary_input)
                    .appearance(false)
                    .into_any_element(),
            ))
            .child(field_box(
                "Steps (one per line)",
                Textarea::new(&self.steps_input)
                    .appearance(false)
                    .into_any_element(),
            ))
            .child(
                h_flex()
                    .gap_2()
                    .text_xs()
                    .text_color(rgb(0x6B7280))
                    .child(self.context_line.clone())
                    .child("·")
                    .child(plan.delivery.label()),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x8A6D3B))
                    .child(format!("Permissions: {}", self.permissions)),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x6B7280))
                    .child(self.references_line()),
            )
            .child(
                h_flex()
                    .gap_2()
                    .items_end()
                    .w_full()
                    .child(
                        div().flex_1().child(field_box(
                            "Not right? Say what to change and the AI re-reads it",
                            Input::new(&self.revise_input)
                                .appearance(false)
                                .into_any_element(),
                        )),
                    )
                    .child(
                        Button::new("revise")
                            .icon(Icon::new(IconName::WandSparkles))
                            .tooltip("Let the AI revise the plan (Enter)")
                            .accessibility_label("Revise the plan")
                            .compact()
                            .ghost()
                            .on_click(cx.listener(|this, _, window, cx| this.revise(window, cx))),
                    ),
            );
        if let Some(error) = &self.error {
            body = body.child(
                div()
                    .text_xs()
                    .text_color(rgb(0xB91C1C))
                    .child(error.clone()),
            );
        }
        body.into_any_element()
    }

    fn references_line(&self) -> String {
        if self.references.is_empty() {
            "References: no files in the workspace yet".into()
        } else {
            format!("References: {}", self.references.join(", "))
        }
    }

    fn paused_body(&self) -> AnyElement {
        v_flex()
            .gap_3()
            .w_full()
            .child(
                h_flex()
                    .gap_2()
                    .items_center()
                    .child(div().size(px(8.)).rounded_full().bg(rgb(0x2563EB)))
                    .child(
                        div()
                            .text_lg()
                            .font_weight(FontWeight::SEMIBOLD)
                            .child("Task paused"),
                    ),
            )
            .child(
                div().text_sm().text_color(rgb(0x4B5563)).child(
                    self.log
                        .last()
                        .cloned()
                        .unwrap_or_else(|| "Waiting on your call.".into()),
                ),
            )
            .child(div().text_xs().text_color(rgb(0x9AA0A6)).child(
                "Resume keeps the plan and continues from the next round; Cancel ends the task.",
            ))
            .into_any_element()
    }

    fn resume(&mut self, cx: &mut Context<Self>) {
        let Some(task_id) = self.task_id else {
            return;
        };
        let runtime = self.ctx.runtime.clone();
        self.mode = Mode::Running;
        self.state_label = "Running".into();
        cx.spawn(async move |this, cx| {
            let result = runtime.resume(task_id).await;
            let _ = this.update(cx, |this, cx| {
                if let Err(err) = result {
                    this.mode = Mode::Failed;
                    this.error = Some(err.to_string());
                }
                cx.notify();
            });
        })
        .detach();
    }

    fn running_body(&self) -> AnyElement {
        let mut log = v_flex().gap_1().w_full();
        for line in &self.log {
            log = log.child(
                div()
                    .text_xs()
                    .text_color(rgb(0x6B7280))
                    .child(line.clone()),
            );
        }
        v_flex()
            .gap_3()
            .w_full()
            .child(
                h_flex()
                    .gap_2()
                    .items_center()
                    .child(div().size(px(8.)).rounded_full().bg(rgb(0xD97706)))
                    .child(
                        div()
                            .text_lg()
                            .font_weight(FontWeight::SEMIBOLD)
                            .child(self.summary.clone()),
                    ),
            )
            .child(log)
            .child(
                div().text_xs().text_color(rgb(0x9AA0A6)).child(
                    "Runs in the background and delivers when done; carry on with your work.",
                ),
            )
            .into_any_element()
    }

    fn done_body(&self) -> AnyElement {
        v_flex()
            .gap_3()
            .w_full()
            .child(
                h_flex()
                    .gap_2()
                    .items_center()
                    .child(div().size(px(8.)).rounded_full().bg(rgb(0x1F7A5C)))
                    .child(
                        div()
                            .text_xl()
                            .font_weight(FontWeight::SEMIBOLD)
                            .child("Task completed"),
                    ),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(rgb(0x4B5563))
                    .child(self.summary.clone()),
            )
            .into_any_element()
    }

    fn failed_body(&self, cancelled: bool) -> AnyElement {
        v_flex()
            .gap_3()
            .w_full()
            .child(
                div()
                    .text_xl()
                    .font_weight(FontWeight::SEMIBOLD)
                    .child(if cancelled {
                        "Task cancelled"
                    } else {
                        "Task not completed"
                    }),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(rgb(0x6B7280))
                    .child(self.error.clone().unwrap_or_else(|| "Stopped.".into())),
            )
            .into_any_element()
    }
}

/// One editable field on the confirmation card: small caption plus bordered input.
fn field_box(label: &str, input: AnyElement) -> AnyElement {
    v_flex()
        .gap_1()
        .w_full()
        .child(
            div()
                .text_xs()
                .text_color(rgb(0x9AA0A6))
                .child(label.to_string()),
        )
        .child(
            div()
                .rounded_xl()
                .border_1()
                .border_color(rgb(0xE3E6EA))
                .bg(rgb(0xFFFFFF))
                .px_3()
                .py_2()
                .child(input),
        )
        .into_any_element()
}

impl Render for Panel {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let height = panel_height(self.mode);
        if (height - self.panel_height).abs() > 0.5 {
            self.panel_height = height;
            window.resize(size(px(PANEL_WIDTH), px(height)));
        }
        if self.focus_input {
            window.focus(&self.input_focus, cx);
            self.focus_input = false;
        }
        if self.sync_inputs {
            self.sync_plan_inputs(window, cx);
        }

        let body: AnyElement = match self.mode {
            Mode::Compose => self.compose_body(cx),
            Mode::Submitting => self.submitting_body(),
            Mode::Confirming => self.confirm_body(cx),
            Mode::Running => self.running_body(),
            Mode::Paused => self.paused_body(),
            Mode::Done => self.done_body(),
            Mode::Failed => self.failed_body(false),
            Mode::Cancelled => self.failed_body(true),
        };

        let mut footer = h_flex().justify_between().items_center().w_full();
        let mut footer_left = h_flex()
            .gap_2()
            .text_xs()
            .text_color(rgb(0x9AA0A6))
            .child(format!("{} to summon", self.ctx.config.hotkey));
        if self.ctx.hotkey_error.is_some() {
            footer_left = footer_left.child("·").child(
                div()
                    .text_color(rgb(0xB45309))
                    .child("Hotkey inactive, open from the tray"),
            );
        }
        if self.ctx.demo {
            footer_left = footer_left.child("·").child("Demo model");
        }
        footer = footer.child(footer_left);
        let mut footer_right = h_flex().gap_2();
        if self.mode == Mode::Compose {
            // One action plus an exit; the session list and management center stay on the tray menu.
            footer_right = footer_right
                .child(
                    Button::new("close-panel")
                        .icon(Icon::new(IconName::X))
                        .tooltip("Close (Esc)")
                        .accessibility_label("Close")
                        .compact()
                        .ghost()
                        .on_click(cx.listener(|this, _, window, cx| this.close(window, cx))),
                )
                .child(
                    Button::new("start")
                        .icon(Icon::new(IconName::Sparkles))
                        .tooltip("Understand the screen and draft a plan (Enter)")
                        .accessibility_label("Understand")
                        .compact()
                        .primary()
                        .on_click(cx.listener(|this, _, window, cx| this.submit(window, cx))),
                );
        } else if self.mode == Mode::Confirming {
            footer_right = footer_right
                .child(
                    Button::new("cancel-task")
                        .icon(Icon::new(IconName::X))
                        .tooltip("Discard this plan and go back to the instruction")
                        .accessibility_label("Cancel")
                        .compact()
                        .ghost()
                        .on_click(cx.listener(|this, _, window, cx| this.cancel(window, cx))),
                )
                .child(
                    Button::new("approve-task")
                        .icon(Icon::new(IconName::Check))
                        .tooltip("Approve the plan and start working (Enter)")
                        .accessibility_label("Approve")
                        .compact()
                        .primary()
                        .on_click(cx.listener(|this, _, window, cx| this.submit(window, cx))),
                );
        } else if self.mode == Mode::Running || self.mode == Mode::Paused {
            footer_right = footer_right.child(
                Button::new("stop")
                    .icon(Icon::new(IconName::CircleStop))
                    .tooltip("Stop the task")
                    .accessibility_label("Stop")
                    .compact()
                    .ghost()
                    .on_click(cx.listener(|this, _, window, cx| this.cancel(window, cx))),
            );
            if self.mode == Mode::Paused {
                footer_right = footer_right.child(
                    Button::new("resume")
                        .icon(Icon::new(IconName::Play))
                        .tooltip("Resume the paused task")
                        .accessibility_label("Resume")
                        .compact()
                        .primary()
                        .on_click(cx.listener(|this, _, _, cx| this.resume(cx))),
                );
            }
        } else if self.mode == Mode::Done {
            footer_right = footer_right
                .child(
                    Button::new("open-artifact")
                        .icon(Icon::new(IconName::FolderOpen))
                        .tooltip("Open the delivered artifact")
                        .accessibility_label("Open artifact")
                        .compact()
                        .ghost()
                        .on_click(
                            cx.listener(|this, _, window, cx| this.open_artifact(window, cx)),
                        ),
                )
                .child(
                    Button::new("finish")
                        .icon(Icon::new(IconName::Check))
                        .tooltip("Done (Esc)")
                        .accessibility_label("Done")
                        .compact()
                        .primary()
                        .on_click(cx.listener(|this, _, window, cx| this.close(window, cx))),
                );
        } else {
            footer_right = footer_right.child(
                Button::new("finish")
                    .icon(Icon::new(IconName::X))
                    .tooltip("Close (Esc)")
                    .accessibility_label("Close")
                    .compact()
                    .ghost()
                    .on_click(cx.listener(|this, _, window, cx| this.close(window, cx))),
            );
        }
        footer = footer.child(footer_right);

        div()
            .id("relay-panel")
            .key_context("RelayPanel")
            .track_focus(&self.focus_handle)
            .on_action(cx.listener(|this, _: &ClosePanel, window, cx| this.close(window, cx)))
            .flex()
            .flex_col()
            .justify_between()
            .size_full()
            .gap_4()
            .bg(rgb(0xFBFBFC))
            .border_1()
            .border_color(rgb(0xE5E7EB))
            .rounded_xl()
            .p_5()
            .child(self.header(cx))
            .child(div().flex_1().min_h_0().overflow_y_scrollbar().child(body))
            .child(footer)
    }
}

pub fn open_panel(cx: &mut App, ctx: Arc<DesktopContext>) {
    open_panel_inner(cx, ctx, false, true);
}

/// Back from the screenshot editor: the front window was already remembered before the capture, so
/// skip that lookup — enumerating every window costs more than rebuilding the panel.
pub fn open_panel_after_snip(cx: &mut App, ctx: Arc<DesktopContext>) {
    open_panel_inner(cx, ctx, false, false);
}

/// `resume_confirmation`: whether to restore an "awaiting confirmation" task as the confirmation card.
/// True only when the runtime explicitly summons the panel, so stale tasks are not resurfaced.
pub fn open_panel_with(cx: &mut App, ctx: Arc<DesktopContext>, resume_confirmation: bool) {
    open_panel_inner(cx, ctx, resume_confirmation, true);
}

fn open_panel_inner(
    cx: &mut App,
    ctx: Arc<DesktopContext>,
    resume_confirmation: bool,
    remember_front: bool,
) {
    // The screenshot editor owns the screen; the panel comes back when it closes.
    if crate::snip::editor_open(cx) {
        tracing::debug!("panel summon ignored: the screenshot editor is open");
        return;
    }
    if !panel_window_alive(cx) {
        // Nothing on screen: the flag outlived its window, so start clean instead of no-opping.
        PANEL_OPEN.store(false, Ordering::SeqCst);
    }
    if PANEL_OPEN.swap(true, Ordering::SeqCst) {
        cx.activate(true);
        return;
    }
    // The panel steals focus, so remember the window the user is watching as a fallback for understanding.
    if remember_front {
        relay_platform::remember_user_window();
    }
    cx.bind_keys([KeyBinding::new("escape", ClosePanel, Some("RelayPanel"))]);
    let options = WindowOptions {
        window_bounds: Some(WindowBounds::centered(
            size(
                px(PANEL_WIDTH),
                px(if resume_confirmation {
                    panel_height(Mode::Confirming)
                } else {
                    panel_height(Mode::Compose)
                }),
            ),
            cx,
        )),
        titlebar: None,
        // Floating, not PopUp: `PopUp` maps to an NSWindow with `NonactivatingPanel`, so the
        // window takes keys without activating the app, and macOS then refuses to run the input
        // method against it — Chinese/Japanese typing loses its candidate window (and IMK logs
        // "error messaging the mach port for IMKCFRunLoopWakeUpReliable"). Floating is still an
        // NSPanel above normal windows, but the app becomes active, so IME works.
        kind: WindowKind::Floating,
        is_movable: true,
        is_resizable: false,
        focus: true,
        ..Default::default()
    };
    let result = cx.open_window(options, move |window, cx| {
        let panel = cx.new(|cx| Panel::new(window, cx, ctx, resume_confirmation));
        window.activate_window();
        cx.new(|cx| Root::new(panel, window, cx))
    });
    match result {
        Ok(handle) => {
            if let Ok(mut slot) = PANEL_WINDOW.lock() {
                *slot = Some(handle);
            }
            cx.activate(true);
        }
        Err(err) => {
            tracing::error!(%err, "failed to open the panel window");
            PANEL_OPEN.store(false, Ordering::SeqCst);
        }
    }
}

struct Management {
    ctx: Arc<DesktopContext>,
    task_rows: Vec<(TaskId, String)>,
    session_rows: Vec<String>,
    settings: Vec<String>,
    selected: Option<TaskId>,
    detail: Vec<String>,
    focus_handle: FocusHandle,
}

impl Management {
    fn new(window: &mut Window, cx: &mut Context<Self>, ctx: Arc<DesktopContext>) -> Self {
        let focus_handle = cx.focus_handle();
        focus_handle.focus(window, cx);
        let mut view = Self {
            ctx,
            task_rows: Vec::new(),
            session_rows: Vec::new(),
            settings: Vec::new(),
            selected: None,
            detail: Vec::new(),
            focus_handle,
        };
        view.reload();
        view
    }

    fn reload(&mut self) {
        self.task_rows = self
            .ctx
            .runtime
            .recent_tasks(40)
            .unwrap_or_default()
            .into_iter()
            .map(|view| {
                (
                    view.task.id,
                    format!(
                        "{}  ·  {}  ·  {} / {}",
                        view.task.state.label(),
                        view.task.title,
                        view.workspace_name,
                        view.session_title
                    ),
                )
            })
            .collect();

        let mut sessions = Vec::new();
        for workspace in self.ctx.runtime.workspaces().unwrap_or_default() {
            for session in self
                .ctx
                .runtime
                .sessions(workspace.id, 5)
                .unwrap_or_default()
            {
                sessions.push(format!(
                    "{} / {}  ·  recent activity {}",
                    workspace.name,
                    session.title,
                    session.last_active_at.format("%m-%d %H:%M")
                ));
            }
        }
        self.session_rows = sessions;

        let config = &self.ctx.config;
        self.settings = vec![
            format!("Model: {} ({})", config.model, config.base_url),
            format!("API Key: {}", config.api_key_source()),
            format!("Hotkey: {}", config.hotkey),
            format!(
                "Sandbox: {} · Confirm before run: {} · Auto-open artifacts: {}",
                if config.sandbox { "On" } else { "Off" },
                if config.require_confirmation {
                    "Required"
                } else {
                    "Skip"
                },
                if config.auto_open { "Yes" } else { "No" }
            ),
            format!("Data directory: {}", AppConfig::home().display()),
            format!("Config file: {}", AppConfig::config_path().display()),
        ];
    }

    fn task_section(&self, cx: &mut Context<Self>) -> AnyElement {
        let mut list = v_flex().gap_1().w_full();
        if self.task_rows.is_empty() {
            list = list.child(
                div()
                    .text_sm()
                    .text_color(rgb(0x9AA0A6))
                    .child("No tasks yet. Use the hotkey to hand Relay its first job."),
            );
        } else {
            for (id, label) in &self.task_rows {
                let id = *id;
                let selected = self.selected == Some(id);
                list = list.child(
                    div()
                        .id(format!("task-{id}"))
                        .px_3()
                        .py_2()
                        .rounded_md()
                        .bg(rgb(if selected { 0xECF6F1 } else { 0xFFFFFF }))
                        .border_1()
                        .border_color(rgb(if selected { 0xBBDCCB } else { 0xEEF0F2 }))
                        .text_sm()
                        .text_color(rgb(0x374151))
                        .child(label.clone())
                        .on_click(cx.listener(move |this, _, _, cx| this.select(id, cx))),
                );
            }
        }
        v_flex()
            .gap_2()
            .w_full()
            .child(
                div()
                    .text_sm()
                    .font_weight(FontWeight::SEMIBOLD)
                    .child("Task queue and run history"),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x9AA0A6))
                    .child("Latest 40; click one to see its run log"),
            )
            .child(list)
            .into_any_element()
    }

    fn detail_section(&self) -> AnyElement {
        let mut list = v_flex().gap_1().w_full();
        for line in &self.detail {
            list = list.child(
                div()
                    .text_xs()
                    .text_color(rgb(0x4B5563))
                    .child(line.clone()),
            );
        }
        v_flex()
            .gap_2()
            .w_full()
            .child(
                div()
                    .text_sm()
                    .font_weight(FontWeight::SEMIBOLD)
                    .child("Run log"),
            )
            .child(
                div()
                    .px_3()
                    .py_2()
                    .rounded_md()
                    .bg(rgb(0xFFFFFF))
                    .border_1()
                    .border_color(rgb(0xEEF0F2))
                    .child(list),
            )
            .into_any_element()
    }

    fn section(&self, title: &str, subtitle: &str, rows: &[String], empty: &str) -> AnyElement {
        let mut list = v_flex().gap_1().w_full();
        if rows.is_empty() {
            list = list.child(
                div()
                    .text_sm()
                    .text_color(rgb(0x9AA0A6))
                    .child(empty.to_string()),
            );
        } else {
            for row in rows {
                list = list.child(
                    div()
                        .px_3()
                        .py_2()
                        .rounded_md()
                        .bg(rgb(0xFFFFFF))
                        .border_1()
                        .border_color(rgb(0xEEF0F2))
                        .text_sm()
                        .text_color(rgb(0x374151))
                        .child(row.clone()),
                );
            }
        }
        v_flex()
            .gap_2()
            .w_full()
            .child(
                div()
                    .text_sm()
                    .font_weight(FontWeight::SEMIBOLD)
                    .child(title.to_string()),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x9AA0A6))
                    .child(subtitle.to_string()),
            )
            .child(list)
            .into_any_element()
    }

    fn select(&mut self, task_id: TaskId, cx: &mut Context<Self>) {
        self.selected = Some(task_id);
        self.detail = self
            .ctx
            .runtime
            .task_log(task_id)
            .unwrap_or_else(|err| vec![format!("Failed to read the log: {err}")]);
        cx.notify();
    }

    fn close(window: &mut Window) {
        window.remove_window();
    }
}

impl Render for Management {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let mut body = v_flex().gap_5().w_full().child(self.task_section(cx));
        if self.selected.is_some() {
            body = body.child(self.detail_section());
        }
        let body = body
            .child(self.section(
                "Workspaces and sessions",
                "Latest 5 sessions per workspace",
                &self.session_rows,
                "No workspaces yet; one is created when a task is routed.",
            ))
            .child(self.section(
                "Settings",
                "Read-only; edit config.toml and restart Relay to apply",
                &self.settings,
                "Failed to read the config.",
            ));

        div()
            .track_focus(&self.focus_handle)
            .flex()
            .flex_col()
            .gap_4()
            .size_full()
            .p_6()
            .bg(rgb(0xF7F8FA))
            .child(
                h_flex()
                    .justify_between()
                    .items_center()
                    .w_full()
                    .child(
                        div()
                            .text_xl()
                            .font_weight(FontWeight::SEMIBOLD)
                            .child("Relay management center"),
                    )
                    .child(
                        h_flex()
                            .gap_2()
                            .child(
                                Button::new("conversation")
                                    .icon(Icon::new(IconName::MessageSquarePlus))
                                    .tooltip("Open the session window")
                                    .accessibility_label("Sessions")
                                    .compact()
                                    .ghost()
                                    .on_click(cx.listener(|this, _, _, cx| {
                                        crate::conversation::open_conversation_window(
                                            cx,
                                            this.ctx.clone(),
                                        );
                                    })),
                            )
                            .child(
                                Button::new("open-config")
                                    .icon(Icon::new(IconName::FolderOpen))
                                    .tooltip("Open the Relay data folder")
                                    .accessibility_label("Open config folder")
                                    .compact()
                                    .ghost()
                                    .on_click(|_, _, _| {
                                        let _ = open::that_detached(AppConfig::home());
                                    }),
                            )
                            .child(
                                Button::new("reload")
                                    .icon(Icon::new(IconName::RefreshCw))
                                    .tooltip("Refresh")
                                    .accessibility_label("Refresh")
                                    .compact()
                                    .ghost()
                                    .on_click(cx.listener(|this, _, _, cx| {
                                        this.reload();
                                        cx.notify();
                                    })),
                            )
                            .child(
                                Button::new("close")
                                    .icon(Icon::new(IconName::X))
                                    .tooltip("Close")
                                    .accessibility_label("Close")
                                    .compact()
                                    .ghost()
                                    .on_click(|_, window, _| Management::close(window)),
                            ),
                    ),
            )
            .child(div().flex_1().overflow_y_scrollbar().child(body))
    }
}

pub fn open_management_window(cx: &mut App, ctx: Arc<DesktopContext>) {
    let mut slot = MANAGEMENT_WINDOW.lock().unwrap();
    // If already open, raise it instead of stacking a second or third window.
    if let Some(handle) = slot.as_ref() {
        if handle
            .update(cx, |_, window, _| window.activate_window())
            .is_ok()
        {
            cx.activate(true);
            return;
        }
        // A stale handle means the user closed the window.
        *slot = None;
    }

    let options = WindowOptions {
        window_bounds: Some(WindowBounds::centered(size(px(860.), px(560.)), cx)),
        titlebar: Some(TitlebarOptions {
            title: Some("Relay".into()),
            appears_transparent: false,
            traffic_light_position: None,
        }),
        ..Default::default()
    };
    match cx.open_window(options, move |window, cx| {
        let view = cx.new(|cx| Management::new(window, cx, ctx));
        cx.new(|cx| Root::new(view, window, cx))
    }) {
        Ok(handle) => {
            // The app runs as an Accessory (no Dock icon), so new windows are not raised automatically;
            // without explicit activation, hiding the panel looks like the center failed to open.
            let _ = handle.update(cx, |_, window, _| window.activate_window());
            cx.activate(true);
            *slot = Some(handle);
        }
        Err(err) => tracing::error!(%err, "failed to create the management center window"),
    }
}

/// Start the resident desktop client: hotkey -> minimal panel -> background run.
pub fn run(config: AppConfig) -> anyhow::Result<()> {
    // One shell per data directory. A second launch would add a duplicate menu-bar icon, fight the
    // first process for the global hotkey and open a second writer on the SQLite file, so it only
    // wakes the running instance and exits.
    let (wake_tx, mut wake_rx) = tokio::sync::mpsc::unbounded_channel();
    let _guard = match crate::instance::claim(wake_tx)? {
        crate::instance::Launch::Primary(guard) => guard,
        crate::instance::Launch::AlreadyRunning => {
            let note = if cfg!(unix) {
                "relay: already running — raised the existing panel"
            } else {
                "relay: already running — use the shortcut to reach it"
            };
            tracing::info!("another Relay instance is running; not starting a second one");
            eprintln!("{note}");
            return Ok(());
        }
    };

    let tokio_rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let handle = tokio_rt.handle().clone();
    let (runtime, demo) = crate::build_runtime(&config, false)?;
    if demo {
        tracing::warn!("no model API key configured, using the demo model");
    }
    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        home = %crate::config::AppConfig::home().display(),
        model = %config.model,
        demo,
        hotkey = %config.hotkey,
        "Relay started"
    );

    let (hotkey_tx, mut hotkey_rx) = tokio::sync::mpsc::unbounded_channel();
    let (tray_tx, mut tray_rx) = tokio::sync::mpsc::unbounded_channel();

    gpui_kit::application()
        .with_assets(gpui_kit::assets::AllAssets)
        .run(move |cx| {
            gpui_kit::init(cx);
            // Resident tray: no Dock icon, no auto popup; summoned only by hotkey or tray.
            cx.set_activation_policy(ActivationPolicy::Accessory);
            cx.bind_keys([KeyBinding::new("escape", ClosePanel, Some("RelayPanel"))]);

            let hotkey_error = match crate::hotkey::register(&config.hotkey, hotkey_tx) {
                Ok(()) => None,
                Err(err) => {
                    tracing::error!(%err, "global hotkey registration failed");
                    Some(err)
                }
            };

            let ctx = Arc::new(DesktopContext {
                runtime,
                tokio: handle.clone(),
                config: config.clone(),
                demo,
                hotkey_error,
                draft: Mutex::new(PanelDraft::default()),
                snip_toolbar: Mutex::new(None),
            });

            // Dev / screenshot use: open a given window at startup to check the UI in a separate RELAY_HOME.
            if std::env::var("RELAY_OPEN_WINDOW").as_deref() == Ok("conversation") {
                crate::conversation::open_conversation_window(cx, ctx.clone());
            }
            if std::env::var("RELAY_OPEN_WINDOW").as_deref() == Ok("snip") {
                match crate::snip::capture_desktop() {
                    Ok(image) => crate::snip::open_snip(cx, ctx.clone(), image),
                    Err(err) => tracing::warn!(%err, "dev: could not capture for the snip window"),
                }
            }

            let ctx_for_hotkey = ctx.clone();
            cx.spawn(async move |cx| {
                while hotkey_rx.recv().await.is_some() {
                    tracing::debug!("hotkey pressed");
                    let ctx = ctx_for_hotkey.clone();
                    cx.update(|cx| open_panel(cx, ctx));
                }
            })
            .detach();

            // Relaunching `relay` while the shell runs connects to the instance socket; that
            // wake-up opens the panel instead of starting a second shell.
            let ctx_for_launch = ctx.clone();
            cx.spawn(async move |cx| {
                while wake_rx.recv().await.is_some() {
                    tracing::debug!("second launch arrived; raising the panel");
                    let ctx = ctx_for_launch.clone();
                    cx.update(|cx| open_panel(cx, ctx));
                }
            })
            .detach();

            // The panel hides while understanding/running; it is summoned back when confirmation is needed.
            let ctx_for_confirm = ctx.clone();
            cx.spawn(async move |cx| {
                let mut events = ctx_for_confirm.runtime.subscribe();
                while let Ok(event) = events.recv().await {
                    if matches!(event.kind, RuntimeEventKind::ConfirmationRequested { .. }) {
                        let ctx = ctx_for_confirm.clone();
                        cx.update(move |cx| open_panel_with(cx, ctx, true));
                    }
                }
            })
            .detach();

            let ctx_for_tray = ctx.clone();
            cx.spawn(async move |cx| {
                while let Some(action) = tray_rx.recv().await {
                    let ctx = ctx_for_tray.clone();
                    cx.update(move |cx| match action {
                        crate::tray::TrayAction::OpenPanel => open_panel(cx, ctx),
                        crate::tray::TrayAction::OpenConversation => {
                            crate::conversation::open_conversation_window(cx, ctx)
                        }
                        crate::tray::TrayAction::OpenManagement => open_management_window(cx, ctx),
                        crate::tray::TrayAction::Quit => cx.quit(),
                    });
                }
            })
            .detach();

            if let Err(err) = crate::tray::install(cx, &ctx.runtime, tray_tx) {
                // Keep a usable entry point when the tray is unavailable so the app is not invisible.
                tracing::warn!(%err, "failed to create the menu bar icon, opening the panel directly");
                open_panel(cx, ctx);
            }
        });

    Ok(())
}
