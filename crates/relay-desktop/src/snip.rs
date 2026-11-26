//! Screenshot annotation window.
//!
//! The panel closes first, then this window opens on the primary display showing the desktop
//! screenshot the user just triggered. It offers region selection plus rectangle / ellipse /
//! arrow / text markup, and returns to the panel on attach or cancel.
//!
//! The desktop pixels are handed to the window in memory: encoding a Retina-sized PNG and decoding
//! it again costs more than a second per capture, and the editor never needs the file.
//!
//! The attachment is produced by capturing the editor with its chrome hidden: the bitmap then
//! contains the screenshot plus every annotation exactly as painted, including text in any
//! language the system font covers.

use std::cell::Cell;
use std::path::Path as FsPath;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::input::{Input, InputEvent, InputState};
use gpui_kit::component::Icon;
use gpui_kit::component::{h_flex, Disableable as _, Root};
use gpui_kit::*;

use crate::app::{open_panel, open_panel_after_snip, DesktopContext, SnipAttachment};

actions!(
    relay_snip,
    [
        CancelSnip,
        ConfirmSnip,
        SelectTool,
        RectTool,
        EllipseTool,
        ArrowTool,
        TextTool,
        UndoSnip
    ]
);

pub(crate) static SNIP_OPEN: AtomicBool = AtomicBool::new(false);
static SNIP_WINDOW: Mutex<Option<WindowHandle<Root>>> = Mutex::new(None);

/// Whether the editor really owns the screen right now.
///
/// The flag alone is not enough: if the window goes away without `close_to_panel` (a crashed
/// frame, the system dropping it), a stale `true` would make every later summon a silent no-op,
/// which looks exactly like "the hotkey stopped working".
pub fn editor_open(cx: &mut App) -> bool {
    if !SNIP_OPEN.load(Ordering::SeqCst) {
        return false;
    }
    let mut slot = SNIP_WINDOW.lock().unwrap();
    if let Some(handle) = slot.as_ref() {
        let id = handle.window_id();
        if cx.windows().iter().any(|window| window.window_id() == id) {
            return true;
        }
    }
    *slot = None;
    SNIP_OPEN.store(false, Ordering::SeqCst);
    tracing::warn!("screenshot editor window was gone; releasing the screen for the panel");
    false
}

const TOOLBAR_HEIGHT: f32 = 44.;
const HINT_WIDTH: f32 = 620.;
const TEXT_BOX_WIDTH: f32 = 260.;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tool {
    Select,
    Rect,
    Ellipse,
    Arrow,
    Text,
}

#[derive(Debug, Clone, Copy)]
struct Rect {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

impl Rect {
    fn from_points(a: (f32, f32), b: (f32, f32)) -> Self {
        Self {
            x: a.0.min(b.0),
            y: a.1.min(b.1),
            w: (a.0 - b.0).abs(),
            h: (a.1 - b.1).abs(),
        }
    }
}

#[derive(Debug, Clone)]
enum Shape {
    Rect(Rect),
    Ellipse(Rect),
    Arrow { from: (f32, f32), to: (f32, f32) },
    Text { at: (f32, f32), text: String },
}

impl Shape {
    fn from_points(tool: Tool, a: (f32, f32), b: (f32, f32)) -> Option<Self> {
        match tool {
            Tool::Rect => Some(Shape::Rect(Rect::from_points(a, b))),
            Tool::Ellipse => Some(Shape::Ellipse(Rect::from_points(a, b))),
            Tool::Arrow => Some(Shape::Arrow { from: a, to: b }),
            _ => None,
        }
    }

    fn is_degenerate(&self) -> bool {
        match self {
            Shape::Rect(rect) | Shape::Ellipse(rect) => rect.w < 4. || rect.h < 4.,
            Shape::Arrow { from, to } => (from.0 - to.0).abs() < 6. && (from.1 - to.1).abs() < 6.,
            Shape::Text { .. } => false,
        }
    }
}

/// Everything one paint pass needs, cloned per frame so the canvas closure stays `'static`.
#[derive(Clone)]
struct PaintState {
    selection: Option<Rect>,
    shapes: Vec<Shape>,
    draft: Option<Shape>,
    /// Dim outside the selection and draw the selection chrome; off while exporting.
    chrome: bool,
}

/// One undo step: the markup and the region the user had picked before a change.
#[derive(Debug, Clone)]
struct Step {
    shapes: Vec<Shape>,
    selection: Option<Rect>,
}

pub struct Snip {
    ctx: Arc<DesktopContext>,
    /// The desktop capture, ready to paint: no file, no decode.
    image: Arc<RenderImage>,
    /// Logical bounds of the display the screenshot came from; the image is drawn 1:1 inside it.
    display: Option<Bounds<Pixels>>,
    window_size: (f32, f32),
    tool: Tool,
    selection: Option<Rect>,
    shapes: Vec<Shape>,
    drag_from: Option<(f32, f32)>,
    draft: Option<Shape>,
    text_at: Option<(f32, f32)>,
    text_input: Option<Entity<InputState>>,
    /// Focus the note input on the next frame, once its element exists.
    text_focus: bool,
    /// Editor states before each change, newest last: undo restores the previous one.
    undo: Vec<Step>,
    exporting: bool,
    /// Toolbar top-left in window coordinates once the user has moved it; `None` centers it on top.
    toolbar_at: Option<(f32, f32)>,
    /// Pointer offset inside the toolbar while it is being dragged.
    toolbar_drag: Option<(f32, f32)>,
    /// A press was a drag, so releasing it must not also fire the button under the pointer.
    toolbar_moved: bool,
    /// Where the press landed: a couple of pixels of jitter must still count as a click.
    toolbar_press_at: Option<Point<Pixels>>,
    /// Where the toolbar was painted last frame: start position and clamp bounds for dragging.
    toolbar_rect: Rc<Cell<Option<Bounds<Pixels>>>>,
    error: Option<String>,
    focus_handle: FocusHandle,
}

impl Snip {
    fn new(
        window: &mut Window,
        cx: &mut Context<Self>,
        ctx: Arc<DesktopContext>,
        image: Arc<RenderImage>,
        display: Option<Bounds<Pixels>>,
        window_size: (f32, f32),
    ) -> Self {
        let focus_handle = cx.focus_handle();
        window.focus(&focus_handle, cx);
        let toolbar_at = ctx.snip_toolbar.lock().ok().and_then(|slot| *slot);
        Self {
            ctx,
            image,
            display,
            window_size,
            tool: Tool::Select,
            selection: None,
            shapes: Vec::new(),
            drag_from: None,
            draft: None,
            text_at: None,
            text_input: None,
            text_focus: false,
            undo: Vec::new(),
            exporting: false,
            toolbar_at,
            toolbar_drag: None,
            toolbar_moved: false,
            toolbar_press_at: None,
            toolbar_rect: Rc::new(Cell::new(None)),
            error: None,
            focus_handle,
        }
    }

    // ---------------------------------------------------------------- interaction

    /// Snapshot the editor before a change so Undo can return to it.
    fn remember(&mut self) {
        self.undo.push(Step {
            shapes: self.shapes.clone(),
            selection: self.selection,
        });
        if self.undo.len() > 50 {
            self.undo.remove(0);
        }
    }

    fn point_of(position: Point<Pixels>) -> (f32, f32) {
        (position.x.into(), position.y.into())
    }

    fn mouse_down(&mut self, event: &MouseDownEvent, window: &mut Window, cx: &mut Context<Self>) {
        if self.exporting || event.button != MouseButton::Left {
            return;
        }
        let at = Self::point_of(event.position);
        // A click elsewhere finishes the note being typed.
        if self.text_input.is_some() {
            self.commit_text(window, cx);
        }
        match self.tool {
            Tool::Select => {
                self.remember();
                self.drag_from = Some(at);
                self.selection = Some(Rect {
                    x: at.0,
                    y: at.1,
                    w: 0.,
                    h: 0.,
                });
            }
            Tool::Rect | Tool::Ellipse | Tool::Arrow => {
                self.drag_from = Some(at);
                self.draft = Shape::from_points(self.tool, at, at);
            }
            Tool::Text => self.begin_text(at, window, cx),
        }
        cx.notify();
    }

    fn mouse_move(&mut self, event: &MouseMoveEvent, cx: &mut Context<Self>) {
        if self.exporting {
            return;
        }
        // A fast drag can outrun the toolbar, so the window keeps steering it even once the pointer
        // is no longer over the grip.
        if self.toolbar_drag.is_some() {
            self.toolbar_drag_to(event, cx);
            return;
        }
        let Some(from) = self.drag_from else {
            return;
        };
        let at = Self::point_of(event.position);
        if self.tool == Tool::Select {
            self.selection = Some(Rect::from_points(from, at));
        } else if let Some(draft) = Shape::from_points(self.tool, from, at) {
            self.draft = Some(draft);
        }
        cx.notify();
    }

    fn mouse_up(&mut self, _event: &MouseUpEvent, cx: &mut Context<Self>) {
        if self.toolbar_drag.is_some() {
            self.toolbar_release(cx);
            return;
        }
        if let Some(draft) = self.draft.take() {
            if !draft.is_degenerate() {
                self.remember();
                self.shapes.push(draft);
            }
            self.drag_from = None;
            cx.notify();
            return;
        }
        if self.drag_from.take().is_some() {
            if let Some(rect) = self.selection {
                if rect.w < 3. || rect.h < 3. {
                    self.selection = None;
                }
            }
            cx.notify();
        }
    }

    fn begin_text(&mut self, at: (f32, f32), window: &mut Window, cx: &mut Context<Self>) {
        let input =
            cx.new(|cx| InputState::new(window, cx).placeholder("Type a note, Enter to place"));
        cx.subscribe_in(
            &input,
            window,
            |this, _input, event: &InputEvent, window, cx| {
                if matches!(event, InputEvent::PressEnter { .. }) {
                    this.commit_text(window, cx);
                }
            },
        )
        .detach();
        self.text_at = Some(at);
        self.text_input = Some(input);
        self.text_focus = true;
        let _ = window;
    }

    fn commit_text(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(input) = self.text_input.take() else {
            return;
        };
        let Some(at) = self.text_at.take() else {
            return;
        };
        let text = input.read(cx).value().trim().to_string();
        tracing::info!(%text, "snip: commit note");
        if !text.is_empty() {
            self.remember();
            self.shapes.push(Shape::Text { at, text });
        }
        window.focus(&self.focus_handle, cx);
        cx.notify();
    }

    fn pick(&mut self, tool: Tool, cx: &mut Context<Self>) {
        if self.tool == tool {
            return;
        }
        self.tool = tool;
        if tool != Tool::Text {
            self.draft = None;
            self.drag_from = None;
        }
        cx.notify();
    }

    fn undo(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.text_input.is_some() {
            // Mid-note: undo drops the note being typed instead of a placed shape.
            self.text_input = None;
            self.text_at = None;
            window.focus(&self.focus_handle, cx);
            cx.notify();
            return;
        }
        if let Some(step) = self.undo.pop() {
            self.shapes = step.shapes;
            self.selection = step.selection;
            self.draft = None;
            self.drag_from = None;
            cx.notify();
        }
    }

    // ---------------------------------------------------------------- toolbar placement

    /// The toolbar covers the desktop, so it has to get out of the way: press anywhere on it —
    /// a grip, a gap, or a button — and it follows the pointer.
    fn toolbar_press(&mut self, event: &MouseDownEvent, cx: &mut Context<Self>) {
        cx.stop_propagation();
        self.toolbar_moved = false;
        if event.button != MouseButton::Left {
            return;
        }
        let Some(rect) = self.toolbar_rect.get() else {
            return;
        };
        let origin = (f32::from(rect.origin.x), f32::from(rect.origin.y));
        self.toolbar_at = Some(origin);
        self.toolbar_press_at = Some(event.position);
        self.toolbar_drag = Some((
            f32::from(event.position.x) - origin.0,
            f32::from(event.position.y) - origin.1,
        ));
    }

    fn toolbar_drag_to(&mut self, event: &MouseMoveEvent, cx: &mut Context<Self>) {
        let (Some((grab_x, grab_y)), Some(rect)) = (self.toolbar_drag, self.toolbar_rect.get())
        else {
            return;
        };
        if !self.toolbar_moved {
            let moved = self
                .toolbar_press_at
                .map(|start| {
                    (f32::from(event.position.x) - f32::from(start.x)).abs() > 2.
                        || (f32::from(event.position.y) - f32::from(start.y)).abs() > 2.
                })
                .unwrap_or(true);
            if !moved {
                return;
            }
        }
        let size = (f32::from(rect.size.width), f32::from(rect.size.height));
        let x =
            (f32::from(event.position.x) - grab_x).clamp(0., (self.window_size.0 - size.0).max(0.));
        let y =
            (f32::from(event.position.y) - grab_y).clamp(0., (self.window_size.1 - size.1).max(0.));
        if self.toolbar_at != Some((x, y)) {
            self.toolbar_moved = true;
            self.toolbar_at = Some((x, y));
            cx.notify();
        }
    }

    fn toolbar_release(&mut self, cx: &mut Context<Self>) {
        cx.stop_propagation();
        self.toolbar_drag = None;
        self.toolbar_press_at = None;
        if let Some(at) = self.toolbar_at {
            if let Ok(mut slot) = self.ctx.snip_toolbar.lock() {
                *slot = Some(at);
            }
        }
    }

    /// Keep a remembered position on screen: a smaller display must not hide the toolbar.
    fn clamp_toolbar(&self, (x, y): (f32, f32)) -> (f32, f32) {
        let (width, height) = self
            .toolbar_rect
            .get()
            .map(|rect| (f32::from(rect.size.width), f32::from(rect.size.height)))
            .unwrap_or((0., 0.));
        (
            x.clamp(0., (self.window_size.0 - width).max(0.)),
            y.clamp(0., (self.window_size.1 - height).max(0.)),
        )
    }

    // ---------------------------------------------------------------- window flow

    fn close_to_panel(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let started = std::time::Instant::now();
        SNIP_OPEN.store(false, Ordering::SeqCst);
        if let Ok(mut slot) = SNIP_WINDOW.lock() {
            *slot = None;
        }
        leave_full_screen(window);
        window.remove_window();
        open_panel_after_snip(cx, self.ctx.clone());
        tracing::debug!(
            ms = started.elapsed().as_millis(),
            "snip: back on the panel"
        );
    }

    fn cancel(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.text_input.is_some() {
            self.text_input = None;
            self.text_at = None;
            window.focus(&self.focus_handle, cx);
            cx.notify();
            return;
        }
        self.close_to_panel(window, cx);
    }

    fn confirm(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.exporting {
            return;
        }
        // While a note is being typed, Enter places the note first; a second Enter attaches.
        if self.text_input.is_some() {
            self.commit_text(window, cx);
            return;
        }
        let selection = self.selection.unwrap_or(Rect {
            x: 0.,
            y: 0.,
            w: self.window_size.0,
            h: self.window_size.1,
        });
        if selection.w < 2. || selection.h < 2. {
            self.error = Some("Select a region first.".into());
            cx.notify();
            return;
        }

        let win_bounds = window.bounds();
        let display = cx.primary_display().map(|display| display.bounds());
        self.exporting = true;
        self.error = None;
        cx.notify();

        let ctx = self.ctx.clone();
        cx.spawn(async move |this, cx| {
            // Let the chrome-hiding frame reach the screen before capturing.
            cx.background_executor()
                .timer(Duration::from_millis(140))
                .await;
            let dir = crate::config::AppConfig::home().join("context");
            let result = cx
                .background_executor()
                .spawn(async move { export_region(&dir, selection, win_bounds, display) })
                .await;

            match result {
                Ok(attachment) => {
                    let handle = SNIP_WINDOW.lock().ok().and_then(|slot| slot.clone());
                    let _ = cx.update(|cx| {
                        if let Ok(mut draft) = ctx.draft.lock() {
                            draft.attachments.push(attachment);
                            draft.notice = None;
                        }
                        SNIP_OPEN.store(false, Ordering::SeqCst);
                        open_panel(cx, ctx.clone());
                        if let Some(handle) = handle {
                            let _ = handle.update(cx, |_root, window, _cx| {
                                leave_full_screen(window);
                                window.remove_window();
                            });
                        }
                    });
                }
                Err(err) => {
                    let message = format!("Could not capture the screen: {err}");
                    let _ = this.update(cx, |this, cx| {
                        this.exporting = false;
                        this.error = Some(message);
                        cx.notify();
                    });
                }
            }
        })
        .detach();
    }

    // ---------------------------------------------------------------- rendering

    fn paint_state(&self) -> PaintState {
        PaintState {
            selection: self.selection,
            shapes: self.shapes.clone(),
            draft: self.draft.clone(),
            chrome: !self.exporting,
        }
    }

    fn tool_button(
        &self,
        id: &'static str,
        icon: IconName,
        tooltip: &'static str,
        tool: Tool,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let button = Button::new(id)
            .icon(Icon::new(icon))
            .tooltip(tooltip)
            .compact();
        let button = if self.tool == tool {
            button.primary()
        } else {
            button.ghost()
        };
        button
            .on_click(cx.listener(move |this, _, _, cx| {
                if !this.toolbar_moved {
                    this.pick(tool, cx);
                }
            }))
            .into_any_element()
    }

    fn toolbar(&self, cx: &mut Context<Self>) -> AnyElement {
        if self.exporting {
            return div().into_any_element();
        }
        let rect_slot = self.toolbar_rect.clone();
        let row = h_flex()
            .h(px(TOOLBAR_HEIGHT))
            .px_2()
            .items_center()
            .gap_1()
            .rounded_xl()
            .bg(rgba(0xFFFFFFF2))
            .border_1()
            .border_color(rgb(0xE5E7EB))
            .shadow_lg()
            .cursor(CursorStyle::OpenHand)
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, event: &MouseDownEvent, _, cx| this.toolbar_press(event, cx)),
            )
            .on_mouse_move(
                cx.listener(|this, event: &MouseMoveEvent, _, cx| this.toolbar_drag_to(event, cx)),
            )
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|this, _, _, cx| this.toolbar_release(cx)),
            )
            // Report the painted bounds so a drag can start from wherever the toolbar currently is.
            .child(
                canvas(
                    move |bounds, _, _| rect_slot.set(Some(bounds)),
                    |_, _, _, _| (),
                )
                .absolute()
                .left(px(0.))
                .top(px(0.))
                .size_full(),
            )
            .child(
                div()
                    .px_1()
                    .text_color(rgb(0xB6BCC6))
                    .child(IconName::GripVertical),
            )
            .child(self.tool_button(
                "tool-select",
                IconName::MousePointer,
                "Select region (1)",
                Tool::Select,
                cx,
            ))
            .child(self.tool_button(
                "tool-rect",
                IconName::Square,
                "Rectangle (2)",
                Tool::Rect,
                cx,
            ))
            .child(self.tool_button(
                "tool-ellipse",
                IconName::Circle,
                "Circle (3)",
                Tool::Ellipse,
                cx,
            ))
            .child(self.tool_button(
                "tool-arrow",
                IconName::ArrowUpRight,
                "Arrow (4)",
                Tool::Arrow,
                cx,
            ))
            .child(self.tool_button("tool-text", IconName::Type, "Text (5)", Tool::Text, cx))
            .child(
                Button::new("snip-undo")
                    .icon(Icon::new(IconName::Undo2))
                    .tooltip("Undo the last step (Cmd+Z)")
                    .accessibility_label("Undo")
                    .compact()
                    .ghost()
                    .disabled(self.undo.is_empty())
                    .on_click(cx.listener(|this, _, window, cx| {
                        if !this.toolbar_moved {
                            this.undo(window, cx);
                        }
                    })),
            )
            .child(div().w(px(1.)).h(px(20.)).mx_2().bg(rgb(0xE5E7EB)))
            .child(
                Button::new("snip-cancel")
                    .icon(Icon::new(IconName::X))
                    .tooltip("Cancel (Esc)")
                    .compact()
                    .ghost()
                    .on_click(cx.listener(|this, _, window, cx| {
                        if !this.toolbar_moved {
                            this.cancel(window, cx);
                        }
                    })),
            )
            .child(
                Button::new("snip-attach")
                    .icon(Icon::new(IconName::Check))
                    .tooltip("Attach (Enter)")
                    .compact()
                    .primary()
                    .on_click(cx.listener(|this, _, window, cx| {
                        if !this.toolbar_moved {
                            this.confirm(window, cx);
                        }
                    })),
            );

        // Center with layout instead of arithmetic: the toolbar stays reachable even on the first
        // frame, before the real window size is known. Once dragged, it keeps the position the user
        // left it at.
        let placed = self.toolbar_at.map(|at| self.clamp_toolbar(at));
        let holder = match placed {
            Some((x, y)) => div().absolute().left(px(x)).top(px(y)),
            None => div()
                .absolute()
                .top(px(16.))
                .left_0()
                .right_0()
                .flex()
                .justify_center(),
        };
        holder.child(row).into_any_element()
    }

    fn hint(&self) -> AnyElement {
        if self.exporting {
            return div().into_any_element();
        }
        let mut line = match self.selection {
            Some(rect) => format!(
                "{} × {} · drag to adjust · Enter to attach · Esc to cancel",
                rect.w.round() as i32,
                rect.h.round() as i32
            ),
            None => "Drag to select a region · 1-5 switch tools · Enter attaches the whole screen"
                .to_string(),
        };
        if let Some(error) = &self.error {
            line = error.clone();
        }
        let color = if self.error.is_some() {
            rgb(0xFCA5A5)
        } else {
            rgb(0xFFFFFF)
        };
        div()
            .absolute()
            .bottom(px(28.))
            .left_0()
            .right_0()
            .flex()
            .justify_center()
            .child(
                div()
                    .max_w(px(HINT_WIDTH))
                    .px_3()
                    .py_1()
                    .rounded_full()
                    .bg(rgba(0x111827D9))
                    .text_sm()
                    .text_color(color)
                    .child(line),
            )
            .into_any_element()
    }

    fn text_input_box(&self) -> AnyElement {
        let Some(input) = &self.text_input else {
            return div().into_any_element();
        };
        let Some(at) = self.text_at else {
            return div().into_any_element();
        };
        let left = at.0.min(self.window_size.0 - TEXT_BOX_WIDTH - 24.).max(8.);
        let top = at.1.min(self.window_size.1 - 64.).max(8.);
        div()
            .absolute()
            .left(px(left))
            .top(px(top))
            .w(px(TEXT_BOX_WIDTH))
            .px_2()
            .py_1()
            .rounded_md()
            .bg(rgb(0xFFFFFF))
            .border_1()
            .border_color(rgb(0x3B82F6))
            .shadow_md()
            .on_mouse_down(MouseButton::Left, |_, _, cx| cx.stop_propagation())
            .child(Input::new(input).appearance(false))
            .into_any_element()
    }

    fn text_shapes_layer(&self) -> AnyElement {
        let mut layer = div().absolute().left(px(0.)).top(px(0.)).size_full();
        for shape in &self.shapes {
            if let Shape::Text { at, text } = shape {
                layer = layer.child(
                    div()
                        .absolute()
                        .left(px(at.0))
                        .top(px(at.1))
                        .max_w(px(420.))
                        .px_2()
                        .py_1()
                        .rounded_md()
                        .bg(rgba(0x111827E6))
                        .text_sm()
                        .text_color(rgb(0xFFFFFF))
                        .child(text.clone()),
                );
            }
        }
        layer.into_any_element()
    }
}

impl Render for Snip {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // The platform may inset the window (menu bar), so lay out against the real bounds.
        if self.text_focus {
            if let Some(input) = &self.text_input {
                let handle = input.read(cx).focus_handle(cx);
                window.focus(&handle, cx);
                tracing::info!(
                    focused = handle.is_focused(window),
                    "snip: focus note input"
                );
            }
            self.text_focus = false;
        }
        let window_bounds = window.bounds();
        let sized = (
            f32::from(window_bounds.size.width),
            f32::from(window_bounds.size.height),
        );
        if sized != self.window_size {
            tracing::debug!(
                width = sized.0,
                height = sized.1,
                fullscreen = window.is_simple_fullscreen(),
                bounds = ?window_bounds,
                "snip: window sized"
            );
        }
        self.window_size = sized;
        // Draw the screenshot at its natural size: the window is the display minus system insets,
        // so the image is shifted up and the hidden strip simply stays hidden.
        let (image_left, image_top, image_width, image_height) = match self.display {
            Some(display) => (
                f32::from(display.origin.x - window_bounds.origin.x),
                f32::from(display.origin.y - window_bounds.origin.y),
                f32::from(display.size.width),
                f32::from(display.size.height),
            ),
            None => (0., 0., self.window_size.0, self.window_size.1),
        };
        let paint = self.paint_state();
        let dimmed = self.selection.is_some();
        div()
            .id("snip-root")
            .relative()
            .size_full()
            .overflow_hidden()
            .key_context("RelaySnip")
            .track_focus(&self.focus_handle)
            .on_action(cx.listener(|this, _: &CancelSnip, window, cx| this.cancel(window, cx)))
            .on_action(cx.listener(|this, _: &ConfirmSnip, window, cx| this.confirm(window, cx)))
            .on_action(cx.listener(|this, _: &SelectTool, _, cx| this.pick(Tool::Select, cx)))
            .on_action(cx.listener(|this, _: &RectTool, _, cx| this.pick(Tool::Rect, cx)))
            .on_action(cx.listener(|this, _: &EllipseTool, _, cx| this.pick(Tool::Ellipse, cx)))
            .on_action(cx.listener(|this, _: &ArrowTool, _, cx| this.pick(Tool::Arrow, cx)))
            .on_action(cx.listener(|this, _: &TextTool, _, cx| this.pick(Tool::Text, cx)))
            .on_action(cx.listener(|this, _: &UndoSnip, window, cx| this.undo(window, cx)))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, event: &MouseDownEvent, window, cx| {
                    this.mouse_down(event, window, cx)
                }),
            )
            .on_mouse_move(
                cx.listener(|this, event: &MouseMoveEvent, _, cx| this.mouse_move(event, cx)),
            )
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|this, event: &MouseUpEvent, _, cx| this.mouse_up(event, cx)),
            )
            .cursor(CursorStyle::Crosshair)
            .child(
                img(self.image.clone())
                    .absolute()
                    .left(px(image_left))
                    .top(px(image_top))
                    .w(px(image_width))
                    .h(px(image_height))
                    .object_fit(ObjectFit::Fill),
            )
            .child(
                canvas(
                    |_, _, _| (),
                    move |bounds, (), window, _cx| paint.paint(bounds, window),
                )
                .absolute()
                .left(px(0.))
                .top(px(0.))
                .size_full(),
            )
            .child(self.text_shapes_layer())
            .child(self.text_input_box())
            .child(self.toolbar(cx))
            .child(self.hint())
            .child(
                // The mode stays readable at a glance without a sentence: crop icon once a region
                // is picked, full-screen icon while the whole desktop is the target.
                div()
                    .absolute()
                    .left(px(16.))
                    .top(px(14.))
                    .flex()
                    .items_center()
                    .justify_center()
                    .size(px(30.))
                    .rounded_md()
                    .bg(rgba(0x111827A6))
                    .text_color(rgb(0xFFFFFF))
                    .child(if dimmed {
                        IconName::Crop
                    } else {
                        IconName::MonitorUp
                    }),
            )
    }
}

impl PaintState {
    fn paint(&self, bounds: Bounds<Pixels>, window: &mut Window) {
        let ox: f32 = bounds.origin.x.into();
        let oy: f32 = bounds.origin.y.into();
        let width: f32 = bounds.size.width.into();
        let height: f32 = bounds.size.height.into();

        if self.chrome {
            match self.selection {
                None => window.paint_quad(fill(bounds, rgba(0x0B122033))),
                Some(sel) => {
                    let top = sel.y.clamp(0., height);
                    let left = sel.x.clamp(0., width);
                    let right = (sel.x + sel.w).clamp(0., width);
                    let bottom = (sel.y + sel.h).clamp(0., height);
                    let dim = rgba(0x0B122066);
                    let mut band = |x: f32, y: f32, w: f32, h: f32| {
                        if w > 0. && h > 0. {
                            window.paint_quad(fill(
                                Bounds::new(point(px(ox + x), px(oy + y)), size(px(w), px(h))),
                                dim,
                            ));
                        }
                    };
                    band(0., 0., width, top);
                    band(0., bottom, width, height - bottom);
                    band(0., top, left, bottom - top);
                    band(right, top, width - right, bottom - top);

                    if let Some(outer) = stroke_rect(ox, oy, sel, 3.) {
                        window.paint_path(outer, rgb(0xFFFFFF));
                    }
                    if let Some(inner) = stroke_rect(ox, oy, sel, 1.6) {
                        window.paint_path(inner, rgb(0x3B82F6));
                    }
                    for (hx, hy) in [
                        (sel.x, sel.y),
                        (sel.x + sel.w, sel.y),
                        (sel.x, sel.y + sel.h),
                        (sel.x + sel.w, sel.y + sel.h),
                    ] {
                        window.paint_quad(fill(
                            Bounds::new(
                                point(px(ox + hx - 3.), px(oy + hy - 3.)),
                                size(px(6.), px(6.)),
                            ),
                            rgb(0x3B82F6),
                        ));
                    }
                }
            }
        }

        for shape in &self.shapes {
            paint_shape(window, ox, oy, shape);
        }
        if let Some(draft) = &self.draft {
            paint_shape(window, ox, oy, draft);
        }
    }
}

fn paint_shape(window: &mut Window, ox: f32, oy: f32, shape: &Shape) {
    let red = rgb(0xEF4444);
    match shape {
        Shape::Rect(rect) => {
            if let Some(path) = stroke_rect(ox, oy, *rect, 3.) {
                window.paint_path(path, red);
            }
        }
        Shape::Ellipse(rect) => {
            if let Some(path) = stroke_ellipse(ox, oy, *rect, 3.) {
                window.paint_path(path, red);
            }
        }
        Shape::Arrow { from, to } => {
            let (fx, fy) = (ox + from.0, oy + from.1);
            let (tx, ty) = (ox + to.0, oy + to.1);
            let dx = tx - fx;
            let dy = ty - fy;
            let len = (dx * dx + dy * dy).sqrt();
            if len < 1. {
                return;
            }
            let (ux, uy) = (dx / len, dy / len);
            let head = 18_f32.min(len * 0.55).max(8.);
            let half = head * 0.5;
            let base = (tx - ux * head, ty - uy * head);

            let mut shaft = PathBuilder::stroke(px(3.));
            shaft.move_to(point(px(fx), px(fy)));
            shaft.line_to(point(px(base.0), px(base.1)));
            if let Ok(path) = shaft.build() {
                window.paint_path(path, red);
            }

            let mut head_path = PathBuilder::fill();
            head_path.move_to(point(px(tx), px(ty)));
            head_path.line_to(point(px(base.0 - uy * half), px(base.1 + ux * half)));
            head_path.line_to(point(px(base.0 + uy * half), px(base.1 - ux * half)));
            head_path.close();
            if let Ok(path) = head_path.build() {
                window.paint_path(path, red);
            }
        }
        Shape::Text { .. } => {}
    }
}

fn stroke_rect(ox: f32, oy: f32, rect: Rect, width: f32) -> Option<Path<Pixels>> {
    let left = ox + rect.x;
    let top = oy + rect.y;
    let right = left + rect.w;
    let bottom = top + rect.h;
    let mut builder = PathBuilder::stroke(px(width));
    builder.move_to(point(px(left), px(top)));
    builder.line_to(point(px(right), px(top)));
    builder.line_to(point(px(right), px(bottom)));
    builder.line_to(point(px(left), px(bottom)));
    builder.close();
    builder.build().ok()
}

fn stroke_ellipse(ox: f32, oy: f32, rect: Rect, width: f32) -> Option<Path<Pixels>> {
    let cx = ox + rect.x + rect.w / 2.;
    let cy = oy + rect.y + rect.h / 2.;
    let rx = rect.w / 2.;
    let ry = rect.h / 2.;
    let steps = 72;
    let mut builder = PathBuilder::stroke(px(width));
    for step in 0..=steps {
        let angle = step as f32 / steps as f32 * std::f32::consts::TAU;
        let x = cx + rx * angle.cos();
        let y = cy + ry * angle.sin();
        if step == 0 {
            builder.move_to(point(px(x), px(y)));
        } else {
            builder.line_to(point(px(x), px(y)));
        }
    }
    builder.close();
    builder.build().ok()
}

// ---------------------------------------------------------------- capture and export

/// Crop the freshly captured display to the chosen region.
///
/// The editor paints the desktop screenshot over its whole window, so the captured pixels at the
/// selection hold the screenshot plus the annotations exactly as the user saw them.
fn export_region(
    dir: &FsPath,
    selection: Rect,
    window_bounds: Bounds<Pixels>,
    display: Option<Bounds<Pixels>>,
) -> anyhow::Result<SnipAttachment> {
    std::fs::create_dir_all(dir)?;
    // Pixels straight from the display: encoding the full-screen PNG only to decode it again cost
    // more than a second per capture.
    let shot = relay_platform::capture_primary_rgba()?;
    let image_w = shot.width as f32;
    let image_h = shot.height as f32;
    let image = image::RgbaImage::from_raw(shot.width, shot.height, shot.rgba)
        .ok_or_else(|| anyhow::anyhow!("screenshot buffer size mismatch"))?;

    let display_w = display
        .map(|bounds| f32::from(bounds.size.width))
        .filter(|value| *value > 1.)
        .unwrap_or(image_w);
    let display_h = display
        .map(|bounds| f32::from(bounds.size.height))
        .filter(|value| *value > 1.)
        .unwrap_or(image_h);
    let scale_x = image_w / display_w;
    let scale_y = image_h / display_h;

    let origin_x = display
        .map(|bounds| (f32::from(window_bounds.origin.x) - f32::from(bounds.origin.x)) * scale_x)
        .unwrap_or(0.);
    let origin_y = display
        .map(|bounds| (f32::from(window_bounds.origin.y) - f32::from(bounds.origin.y)) * scale_y)
        .unwrap_or(0.);

    let x = (origin_x + selection.x * scale_x)
        .round()
        .clamp(0., image_w - 1.);
    let y = (origin_y + selection.y * scale_y)
        .round()
        .clamp(0., image_h - 1.);
    let w = (selection.w * scale_x).round().max(1.).min(image_w - x);
    let h = (selection.h * scale_y).round().max(1.).min(image_h - y);

    let cropped =
        image::imageops::crop_imm(&image, x as u32, y as u32, w as u32, h as u32).to_image();
    let path = dir.join(format!(
        "snip-{}.png",
        chrono::Local::now().format("%Y%m%d-%H%M%S-%3f")
    ));
    cropped.save(&path)?;
    Ok(SnipAttachment {
        path,
        width: w as u32,
        height: h as u32,
    })
}

/// Grab the primary display as a paintable image; the caller must have hidden the panel first.
pub fn capture_desktop() -> anyhow::Result<Arc<RenderImage>> {
    let started = std::time::Instant::now();
    let shot = relay_platform::capture_primary_rgba()?;
    let (shot_width, shot_height) = (shot.width, shot.height);
    let mut rgba = shot.rgba;
    // GPUI samples image textures as BGRA — its own decoder swaps red and blue before uploading.
    // Handing it plain RGBA shifts every hue (white stays white, but the scene reads yellow), so do
    // the same swap here.
    for pixel in rgba.chunks_exact_mut(4) {
        pixel.swap(0, 2);
    }
    let pixels = image::RgbaImage::from_raw(shot_width, shot_height, rgba)
        .ok_or_else(|| anyhow::anyhow!("screenshot buffer size mismatch"))?;
    let image = Arc::new(RenderImage::new(vec![image::Frame::new(pixels)]));
    tracing::debug!(
        ms = started.elapsed().as_millis(),
        width = shot_width,
        height = shot_height,
        "snip: capture ready to paint"
    );
    Ok(image)
}

/// Grow the editor over the entire display, menu-bar strip included.
///
/// macOS keeps any windowed frame below the menu bar, which would leave a strip of live desktop
/// above the frozen screenshot. A borderless full-screen frame is the one frame it accepts there,
/// so the editor asks for one as soon as its window exists; the window keeps its own chrome-less
/// look, this only changes the frame and hides the menu bar and Dock while capturing.
fn enter_full_screen(window: &mut Window) {
    tracing::debug!(
        already = window.is_simple_fullscreen(),
        bounds = ?window.bounds(),
        "snip: enter full screen"
    );
    if !window.is_simple_fullscreen() {
        window.toggle_simple_fullscreen();
    }
}

/// Give the screen back before the window is destroyed.
///
/// The menu bar and Dock are hidden through app-wide presentation options that only the exit path
/// restores, so leaving has to happen while the window still exists.
fn leave_full_screen(window: &mut Window) {
    if window.is_simple_fullscreen() {
        window.toggle_simple_fullscreen();
    }
}

/// Open the editor for one desktop screenshot; the panel must already be hidden.
pub fn open_snip(cx: &mut App, ctx: Arc<DesktopContext>, image: Arc<RenderImage>) {
    if SNIP_OPEN.swap(true, Ordering::SeqCst) {
        cx.activate(true);
        return;
    }
    // The capture is anchored to the raw display frame, so the image lines up with the screen once
    // the window covers all of it. It is created at the usable desktop instead: a windowed frame
    // spanning the whole screen gets pushed below the menu bar and hangs off the bottom edge, and a
    // frame matching the screen exactly can be parked on a full-screen Space, which made the editor
    // slow to appear. `enter_full_screen` grows it over the menu-bar strip right afterwards.
    let display = cx.primary_display().map(|screen| screen.bounds());
    let bounds =
        display.unwrap_or_else(|| Bounds::new(point(px(0.), px(0.)), size(px(1440.), px(900.))));
    let window_size = (f32::from(bounds.size.width), f32::from(bounds.size.height));
    let display_for_window = display;

    cx.bind_keys([
        KeyBinding::new("escape", CancelSnip, Some("RelaySnip")),
        KeyBinding::new("enter", ConfirmSnip, Some("RelaySnip")),
        KeyBinding::new("1", SelectTool, Some("RelaySnip")),
        KeyBinding::new("2", RectTool, Some("RelaySnip")),
        KeyBinding::new("3", EllipseTool, Some("RelaySnip")),
        KeyBinding::new("4", ArrowTool, Some("RelaySnip")),
        KeyBinding::new("5", TextTool, Some("RelaySnip")),
        KeyBinding::new("cmd-z", UndoSnip, Some("RelaySnip")),
        KeyBinding::new("ctrl-z", UndoSnip, Some("RelaySnip")),
    ]);

    let options = WindowOptions {
        window_bounds: Some(WindowBounds::Windowed(bounds)),
        titlebar: None,
        kind: WindowKind::Floating,
        is_movable: false,
        is_resizable: false,
        focus: true,
        show: true,
        ..Default::default()
    };
    let ctx_for_window = ctx.clone();
    let started = std::time::Instant::now();
    let result = cx.open_window(options, move |window, cx| {
        let snip = cx.new(|cx| {
            Snip::new(
                window,
                cx,
                ctx_for_window,
                image,
                display_for_window,
                window_size,
            )
        });
        window.activate_window();
        cx.new(|cx| Root::new(snip, window, cx))
    });
    match result {
        Ok(handle) => {
            if let Ok(mut slot) = SNIP_WINDOW.lock() {
                *slot = Some(handle);
            }
            let _ = handle.update(cx, |_root, window, _cx| enter_full_screen(window));
            tracing::debug!(ms = started.elapsed().as_millis(), "snip: editor window up");
            cx.activate(true);
        }
        Err(err) => {
            tracing::error!(%err, "failed to open the screenshot window");
            SNIP_OPEN.store(false, Ordering::SeqCst);
            open_panel(cx, ctx);
        }
    }
}
