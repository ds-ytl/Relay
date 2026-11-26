//! Menu bar / system tray resident icon (tray-icon + muda, cross-platform).
//!
//! Design constraint: no main window by default, only a tray resident; the entry points are the tray menu and the global hotkey.
//! This module only handles "icon + menu + turning clicks into [`TrayAction`]" and depends on
//! no UI framework beyond GPUI / tokio, which keeps it testable and easy to swap out.
//!
//! The icon is state-driven: each [`TrayStatus`] has a set of RGBA frames drawn one by one,
//! and callers only need to call [`Tray::tick`] on the main thread on a timer; idle (static) states produce no updates.
//! The title to the right of the icon is drawn natively by the system (`set_title`) and adapts to light or dark menu bars;
//! it is followed by an activity symbol of equal width (e.g. `Thinking ◜`), so the text and width stay fixed while the symbol rotates.

use std::time::{Duration, Instant};

use thiserror::Error;
use tokio::sync::mpsc::UnboundedSender;
use tray_icon::menu::accelerator::Accelerator;
use tray_icon::menu::{Menu, MenuEvent, MenuItem, PredefinedMenuItem};
use tray_icon::{Icon, TrayIcon, TrayIconBuilder};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrayAction {
    OpenPanel,
    OpenConversation,
    OpenManagement,
    Quit,
}

impl TrayAction {
    fn menu_id(self) -> &'static str {
        match self {
            TrayAction::OpenPanel => "relay.open-panel",
            TrayAction::OpenConversation => "relay.open-conversation",
            TrayAction::OpenManagement => "relay.open-management",
            TrayAction::Quit => "relay.quit",
        }
    }

    fn from_menu_id(id: &str) -> Option<TrayAction> {
        match id {
            "relay.open-panel" => Some(TrayAction::OpenPanel),
            "relay.open-conversation" => Some(TrayAction::OpenConversation),
            "relay.open-management" => Some(TrayAction::OpenManagement),
            "relay.quit" => Some(TrayAction::Quit),
            _ => None,
        }
    }
}

#[derive(Debug, Error)]
pub enum TrayError {
    #[error("Failed to create tray menu: {0}")]
    Menu(tray_icon::menu::Error),
    #[error("Failed to create tray icon: {0}")]
    Icon(tray_icon::Error),
}

/// Tray animation state. Callers map their business state here; the tray only cares about "how to animate".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrayStatus {
    /// Idle: static icon.
    Idle,
    /// Understanding the task: three dots breathing in sequence.
    Thinking,
    /// Background execution: a rotating arc.
    Working,
    /// Waiting for confirmation: amber exclamation mark plus an outer pulse.
    Attention,
    /// Paused: two grey bars breathing slowly.
    Paused,
    /// Done: green check stroke, returns to idle after holding.
    Success,
    /// Failed: red cross plus a shake, returns to idle after holding.
    Error,
}

impl TrayStatus {
    /// Short title on the menu bar, drawn natively by the system (adapts to dark / light menu bars).
    pub fn label(self) -> &'static str {
        match self {
            Self::Idle => "Idle",
            Self::Thinking => "Thinking",
            Self::Working => "Working",
            Self::Attention => "Confirm",
            Self::Paused => "Paused",
            Self::Success => "Done",
            Self::Error => "Failed",
        }
    }

    /// "Activity symbol" appended to the title: all symbols of a status have the same width, so animating never shifts the menu bar.
    fn title_symbols(self) -> &'static [&'static str] {
        match self {
            Self::Thinking | Self::Working => &["◜", "◝", "◞", "◟"],
            Self::Attention => &["●", "○"],
            _ => &[],
        }
    }

    fn symbol_interval(self) -> Option<Duration> {
        match self {
            Self::Thinking | Self::Working => Some(Duration::from_millis(220)),
            Self::Attention => Some(Duration::from_millis(700)),
            _ => None,
        }
    }

    /// Menu bar title: `Thinking ◜`. The symbol rotates per frame while the text part stays the same.
    pub fn title(self, symbol: usize) -> String {
        let symbols = self.title_symbols();
        if symbols.is_empty() {
            return self.label().to_string();
        }
        format!("{} {}", self.label(), symbols[symbol % symbols.len()])
    }

    pub fn tooltip(self) -> &'static str {
        match self {
            Self::Idle => "Relay · Idle",
            Self::Thinking => "Relay · Understanding your task",
            Self::Working => "Relay · Running in the background",
            Self::Attention => "Relay · Waiting for your confirmation",
            Self::Paused => "Relay · Paused",
            Self::Success => "Relay · Task completed",
            Self::Error => "Relay · Task failed",
        }
    }

    /// Frames in one animation loop; static states have a single frame.
    fn frames(self) -> usize {
        match self {
            Self::Idle => 1,
            Self::Thinking => 12,
            Self::Working => 24,
            Self::Attention => 12,
            Self::Paused => 6,
            Self::Success => 10,
            Self::Error => 10,
        }
    }

    /// How long one frame stays on screen; `None` means no animation is needed.
    fn frame_duration(self) -> Option<Duration> {
        let ms = match self {
            Self::Idle => return None,
            Self::Thinking => 90,
            Self::Working => 55,
            Self::Attention => 100,
            Self::Paused => 220,
            Self::Success => 70,
            Self::Error => 60,
        };
        Some(Duration::from_millis(ms))
    }

    /// How long a one-shot effect holds after finishing before settling back to [`TrayStatus::Idle`].
    fn hold(self) -> Option<Duration> {
        match self {
            Self::Success => Some(Duration::from_millis(2400)),
            Self::Error => Some(Duration::from_millis(3200)),
            _ => None,
        }
    }

    /// Whether to loop (done / failed play once and stop on the last frame).
    fn looping(self) -> bool {
        !matches!(self, Self::Success | Self::Error)
    }

    fn base_color(self) -> [u8; 3] {
        match self {
            Self::Idle | Self::Thinking | Self::Working => [64, 110, 246],
            Self::Attention => [232, 146, 24],
            Self::Paused => [110, 118, 136],
            Self::Success => [32, 150, 103],
            Self::Error => [214, 64, 69],
        }
    }
}

/// Pure animation state machine: touches no system icon and takes time from the caller, keeping it testable and easy to swap out.
struct Animation {
    status: TrayStatus,
    frame: usize,
    frame_started: Instant,
    status_started: Instant,
}

impl Animation {
    fn new(now: Instant) -> Self {
        Self {
            status: TrayStatus::Idle,
            frame: 0,
            frame_started: now,
            status_started: now,
        }
    }

    fn status(&self) -> TrayStatus {
        self.status
    }

    /// Switch state; returns `true` when the picture needs to be repainted.
    fn set_status(&mut self, status: TrayStatus, now: Instant) -> bool {
        if self.status == status {
            return false;
        }
        self.status = status;
        self.frame = 0;
        self.frame_started = now;
        self.status_started = now;
        true
    }

    /// Whether a timer still needs to advance things: animating, or a one-shot effect still waiting to settle back.
    fn needs_tick(&self) -> bool {
        self.animating() || self.status.hold().is_some()
    }

    /// Advance by time; returns `true` when the picture needs to be repainted.
    fn tick(&mut self, now: Instant) -> bool {
        if let Some(hold) = self.status.hold() {
            if now.saturating_duration_since(self.status_started) >= hold {
                return self.set_status(TrayStatus::Idle, now);
            }
        }
        let Some(frame_duration) = self.status.frame_duration() else {
            return false;
        };
        if !self.animating() || now.saturating_duration_since(self.frame_started) < frame_duration {
            return false;
        }
        self.frame = if self.status.looping() {
            (self.frame + 1) % self.status.frames()
        } else {
            (self.frame + 1).min(self.status.frames() - 1)
        };
        self.frame_started = now;
        true
    }

    fn animating(&self) -> bool {
        self.status.frame_duration().is_some() && self.frame + 1 < self.status.frames()
    }
}

/// Tray handle: must be held for as long as the icon is needed, otherwise the icon gets collected.
pub struct Tray {
    icon: TrayIcon,
    _menu: Menu,
    animation: Animation,
    /// Current frame of the title suffix symbol.
    symbol: usize,
    symbol_started: Instant,
    /// One-line conclusion of the last finished task, shown in the menu bar until the next task starts.
    headline: Option<String>,
}

impl Tray {
    pub fn status(&self) -> TrayStatus {
        self.animation.status()
    }

    /// Switch state: draw the first frame of the new state right away and update the tooltip text.
    pub fn set_status(&mut self, status: TrayStatus) {
        let now = Instant::now();
        if self.animation.set_status(status, now) {
            self.symbol = 0;
            self.symbol_started = now;
            self.apply_labels(status);
            self.redraw();
        }
    }

    /// Show the last task's conclusion next to the icon: the final reply, shortened to one line.
    /// `None` restores the plain status title; running states show their own label either way.
    pub fn set_headline(&mut self, headline: Option<String>) {
        let headline = headline
            .map(|text| compact_headline(&text))
            .filter(|text| !text.is_empty());
        if headline == self.headline {
            return;
        }
        self.headline = headline;
        self.apply_labels(self.animation.status());
    }

    /// The title to the right of the icon (including the activity symbol) and the tooltip both follow the state.
    fn apply_labels(&mut self, status: TrayStatus) {
        let title = compose_title(status, self.headline.as_deref(), self.symbol);
        self.icon.set_title(Some(title));
        let _ = self.icon.set_tooltip(Some(status.tooltip()));
    }

    /// Whether a timer still needs to advance things: icon animation, settle-back wait, or title symbol rotation.
    pub fn needs_tick(&self) -> bool {
        self.animation.needs_tick() || self.animation.status().symbol_interval().is_some()
    }

    /// Advance one frame; called by the main thread when needed.
    pub fn tick(&mut self) {
        let now = Instant::now();
        let previous = self.animation.status();
        if self.animation.tick(now) {
            if self.animation.status() != previous {
                // The one-shot effect timed out and is settling back, so the title and tooltip must recover too.
                self.symbol = 0;
                self.symbol_started = now;
                self.apply_labels(self.animation.status());
            }
            self.redraw();
        }
        self.advance_symbol(now);
    }

    /// Rhythm of the text area: only the suffix symbol changes; the title text and its width stay the same.
    fn advance_symbol(&mut self, now: Instant) {
        let status = self.animation.status();
        let Some(interval) = status.symbol_interval() else {
            return;
        };
        if now.saturating_duration_since(self.symbol_started) < interval {
            return;
        }
        self.symbol = self.symbol.wrapping_add(1);
        self.symbol_started = now;
        self.icon.set_title(Some(status.title(self.symbol)));
    }

    fn redraw(&mut self) {
        let icon = render_icon(self.animation.status, self.animation.frame);
        let _ = self.icon.set_icon(Some(icon));
    }
}

/// Install the tray icon and forward menu events to `sender`.
///
/// Must be called on the GUI main thread (on macOS tray-icon relies on a running NSApplication);
/// later [`Tray::set_status`] / [`Tray::tick`] calls must run on the main thread as well.
pub fn install(sender: UnboundedSender<TrayAction>) -> Result<Tray, TrayError> {
    let menu = Menu::new();
    let open = MenuItem::with_id(
        TrayAction::OpenPanel.menu_id(),
        "Open Panel",
        true,
        "Ctrl+Alt+Space".parse::<Accelerator>().ok(),
    );
    let conversation = MenuItem::with_id(
        TrayAction::OpenConversation.menu_id(),
        "Conversation",
        true,
        None,
    );
    let manage = MenuItem::with_id(
        TrayAction::OpenManagement.menu_id(),
        "Management",
        true,
        None,
    );
    let quit = MenuItem::with_id(TrayAction::Quit.menu_id(), "Quit Relay", true, None);
    let separator = PredefinedMenuItem::separator();
    menu.append_items(&[&open, &conversation, &manage, &separator, &quit])
        .map_err(TrayError::Menu)?;

    let icon = TrayIconBuilder::new()
        .with_menu(Box::new(menu.clone()))
        .with_icon(render_icon(TrayStatus::Idle, 0))
        .with_title(TrayStatus::Idle.title(0))
        .with_tooltip(TrayStatus::Idle.tooltip())
        .with_menu_on_left_click(true)
        .build()
        .map_err(TrayError::Icon)?;

    forward_menu_events(sender);

    Ok(Tray {
        icon,
        _menu: menu,
        animation: Animation::new(Instant::now()),
        symbol: 0,
        symbol_started: Instant::now(),
        headline: None,
    })
}

/// Longest headline that still fits beside the icon in a menu bar, in characters.
const HEADLINE_MAX: usize = 36;

/// The headline replaces the quiet states; work in progress keeps its own label so it stays recognizable.
fn compose_title(status: TrayStatus, headline: Option<&str>, symbol: usize) -> String {
    match headline {
        Some(headline) if matches!(status, TrayStatus::Idle | TrayStatus::Success) => {
            format!("Relay · {headline}")
        }
        _ => status.title(symbol),
    }
}

/// Menu bars have room for one short line: take the first non-empty line, drop markdown markers,
/// and cut with an ellipsis.
fn compact_headline(text: &str) -> String {
    let line = text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("");
    let line = line
        .trim_start_matches(|c: char| c.is_whitespace() || matches!(c, '#' | '>' | '*' | '-'))
        .trim();
    let mut out: String = line.chars().take(HEADLINE_MAX).collect();
    if line.chars().count() > HEADLINE_MAX {
        out.push('…');
    }
    out
}

/// muda menu events travel over a global channel; spawn a thread here to translate them into `TrayAction`.
fn forward_menu_events(sender: UnboundedSender<TrayAction>) {
    std::thread::Builder::new()
        .name("relay-tray-events".into())
        .spawn(move || {
            let receiver = MenuEvent::receiver();
            // recv blocks waiting; the thread exits once the sender is closed.
            while let Ok(event) = receiver.recv() {
                let Some(action) = TrayAction::from_menu_id(event.id().0.as_str()) else {
                    continue;
                };
                if sender.send(action).is_err() {
                    break;
                }
            }
        })
        .ok();
}

// ---------------------------------------------------------------------------
// Icon drawing: 32×32 RGBA, fully procedural, no external image assets.
// ---------------------------------------------------------------------------

/// Actual pixels: the macOS menu bar draws the icon at 18pt, which is exactly 36px on Retina and avoids blurry upscaling.
const ICON_SIZE: u32 = 36;
/// Shapes are drawn in 32×32 logical coordinates and then scaled as a whole to [`ICON_SIZE`].
const LOGICAL: f32 = 32.0;
const SCALE: f32 = ICON_SIZE as f32 / LOGICAL;
const ICON_HALF: f32 = 16.0;
const CORNER: f32 = 9.0;
/// Foreground glyph color: a slightly blue-tinted white that stays legible on light and dark menu bars.
const GLYPH: [u8; 3] = [245, 248, 255];
const TAU: f32 = std::f32::consts::TAU;

struct Canvas {
    size: u32,
    data: Vec<u8>,
}

impl Canvas {
    fn new(size: u32) -> Self {
        Self {
            size,
            data: vec![0; (size * size * 4) as usize],
        }
    }

    /// Composite a color onto the canvas by coverage (straight alpha blending).
    fn paint(&mut self, color: [u8; 3], coverage: impl Fn(f32, f32) -> f32) {
        for y in 0..self.size {
            for x in 0..self.size {
                let lx = (x as f32 + 0.5) / SCALE;
                let ly = (y as f32 + 0.5) / SCALE;
                let alpha = coverage(lx, ly).clamp(0.0, 1.0);
                if alpha <= 0.0 {
                    continue;
                }
                let offset = ((y * self.size + x) * 4) as usize;
                let dst_alpha = self.data[offset + 3] as f32 / 255.0;
                let out_alpha = alpha + dst_alpha * (1.0 - alpha);
                if out_alpha <= 0.0 {
                    continue;
                }
                for channel in 0..3 {
                    let src = color[channel] as f32;
                    let dst = self.data[offset + channel] as f32;
                    let out = (src * alpha + dst * dst_alpha * (1.0 - alpha)) / out_alpha;
                    self.data[offset + channel] = out.round().clamp(0.0, 255.0) as u8;
                }
                self.data[offset + 3] = (out_alpha * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

fn clamp01(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

/// Signed distance field of a rounded rectangle.
fn sd_round_rect(px: f32, py: f32, half: f32, radius: f32) -> f32 {
    let qx = (px - ICON_HALF).abs() - (half - radius);
    let qy = (py - ICON_HALF).abs() - (half - radius);
    let outside = (qx.max(0.0).powi(2) + qy.max(0.0).powi(2)).sqrt();
    outside + qx.max(qy).min(0.0) - radius
}

fn circle(px: f32, py: f32, cx: f32, cy: f32, radius: f32) -> f32 {
    clamp01(radius + 0.5 - ((px - cx).powi(2) + (py - cy).powi(2)).sqrt())
}

fn ring(px: f32, py: f32, cx: f32, cy: f32, radius: f32, half_width: f32) -> f32 {
    let dist = ((px - cx).powi(2) + (py - cy).powi(2)).sqrt();
    clamp01(half_width + 0.5 - (dist - radius).abs())
}

/// A circular arc sweeping from `from` to `to` on a ring.
fn arc(
    px: f32,
    py: f32,
    center: (f32, f32),
    radius: f32,
    half_width: f32,
    from: f32,
    to: f32,
) -> f32 {
    let sweep = to - from;
    if sweep <= 0.0 {
        return 0.0;
    }
    let (cx, cy) = center;
    let ring = ring(px, py, cx, cy, radius, half_width);
    if ring <= 0.0 {
        return 0.0;
    }
    let angle = ((py - cy).atan2(px - cx) - from).rem_euclid(TAU);
    let head = clamp01(angle * radius + 0.5);
    let tail = clamp01((sweep - angle) * radius + 0.5);
    ring * head.min(tail)
}

/// A line segment with round caps.
fn segment(px: f32, py: f32, a: (f32, f32), b: (f32, f32), width: f32) -> f32 {
    let (vx, vy) = (b.0 - a.0, b.1 - a.1);
    let (wx, wy) = (px - a.0, py - a.1);
    let len_sq = vx * vx + vy * vy;
    let t = if len_sq <= f32::EPSILON {
        0.0
    } else {
        ((wx * vx + wy * vy) / len_sq).clamp(0.0, 1.0)
    };
    let (dx, dy) = (wx - vx * t, wy - vy * t);
    clamp01(width * 0.5 + 0.5 - (dx * dx + dy * dy).sqrt())
}

/// Draw a polyline step by step according to `progress` (used for the check / cross stroke animation).
fn polyline(px: f32, py: f32, points: &[(f32, f32)], width: f32, progress: f32) -> f32 {
    if progress <= 0.0 {
        return 0.0;
    }
    let total: f32 = points
        .windows(2)
        .map(|pair| ((pair[1].0 - pair[0].0).powi(2) + (pair[1].1 - pair[0].1).powi(2)).sqrt())
        .sum();
    let mut remaining = total * clamp01(progress);
    let mut coverage: f32 = 0.0;
    for pair in points.windows(2) {
        if remaining <= 0.0 {
            break;
        }
        let len = ((pair[1].0 - pair[0].0).powi(2) + (pair[1].1 - pair[0].1).powi(2)).sqrt();
        let ratio = if len <= f32::EPSILON {
            1.0
        } else {
            (remaining / len).min(1.0)
        };
        let end = (
            pair[0].0 + (pair[1].0 - pair[0].0) * ratio,
            pair[0].1 + (pair[1].1 - pair[0].1) * ratio,
        );
        coverage = coverage.max(segment(px, py, pair[0], end, width));
        remaining -= len;
    }
    coverage
}

fn render_frame(status: TrayStatus, frame: usize) -> Vec<u8> {
    let mut canvas = Canvas::new(ICON_SIZE);
    // Base: rounded square.
    canvas.paint(status.base_color(), |px, py| {
        clamp01(0.5 - sd_round_rect(px, py, ICON_HALF, CORNER))
    });
    let frames = status.frames();
    let frame = frame % frames;
    let progress = frame as f32 / frames as f32;
    match status {
        TrayStatus::Idle => {
            // Same as the old icon: a single white dot in the middle.
            canvas.paint(GLYPH, |px, py| circle(px, py, 16.0, 16.0, 5.5));
        }
        TrayStatus::Thinking => {
            // Three dots breathing in sequence, as if thinking.
            for (index, cx) in [10.5f32, 16.0, 21.5].into_iter().enumerate() {
                let phase = progress * TAU - index as f32 * (TAU / 3.0);
                let pulse = 0.5 + 0.5 * phase.cos();
                let radius = 2.0 + 1.4 * pulse;
                let alpha = 0.55 + 0.45 * pulse;
                canvas.paint(GLYPH, |px, py| circle(px, py, cx, 16.0, radius) * alpha);
            }
        }
        TrayStatus::Working => {
            // Rotating arc + faint tail + center dot.
            let start = -std::f32::consts::FRAC_PI_2 + progress * TAU;
            let sweep = TAU * 0.62;
            canvas.paint(GLYPH, |px, py| {
                arc(px, py, (16.0, 16.0), 7.6, 1.9, start, start + sweep)
            });
            canvas.paint(GLYPH, |px, py| {
                arc(px, py, (16.0, 16.0), 7.6, 1.9, start - 0.55, start) * 0.4
            });
            canvas.paint(GLYPH, |px, py| circle(px, py, 16.0, 16.0, 2.3));
        }
        TrayStatus::Attention => {
            // Outer pulse + slightly bobbing exclamation mark.
            let ring_radius = 11.0 + 2.8 * progress;
            let fade = 1.0 - progress;
            canvas.paint(GLYPH, |px, py| {
                ring(px, py, 16.0, 16.0, ring_radius, 1.1) * fade * 0.6
            });
            let bob = (progress * TAU).sin() * 0.9;
            canvas.paint(GLYPH, |px, py| {
                segment(px, py, (16.0, 9.5 + bob), (16.0, 18.0 + bob), 3.2)
            });
            canvas.paint(GLYPH, |px, py| circle(px, py, 16.0, 22.4 + bob, 1.9));
        }
        TrayStatus::Paused => {
            // Two bars, breathing slowly.
            let pulse = 0.72 + 0.28 * (0.5 + 0.5 * (progress * TAU).cos());
            canvas.paint(GLYPH, |px, py| {
                segment(px, py, (12.6, 10.5), (12.6, 21.5), 3.3) * pulse
            });
            canvas.paint(GLYPH, |px, py| {
                segment(px, py, (19.4, 10.5), (19.4, 21.5), 3.3) * pulse
            });
        }
        TrayStatus::Success => {
            // Check stroke; stops on the complete check when the animation ends.
            let draw = ((frame + 1) as f32 / frames as f32).min(1.0);
            canvas.paint(GLYPH, |px, py| {
                polyline(
                    px,
                    py,
                    &[(9.5, 16.5), (14.0, 21.0), (22.5, 10.5)],
                    3.2,
                    draw,
                )
            });
        }
        TrayStatus::Error => {
            // Cross stroke + a slight shake in the first frames, as if something went wrong.
            const SHAKE: [f32; 6] = [0.0, 1.6, -1.4, 1.0, -0.5, 0.0];
            let dx = SHAKE[frame.min(SHAKE.len() - 1)];
            let draw = ((frame + 1) as f32 / frames as f32).min(1.0);
            canvas.paint(GLYPH, |px, py| {
                polyline(px, py, &[(11.0 + dx, 11.0), (21.0 + dx, 21.0)], 3.0, draw).max(polyline(
                    px,
                    py,
                    &[(21.0 + dx, 11.0), (11.0 + dx, 21.0)],
                    3.0,
                    draw,
                ))
            });
        }
    }
    canvas.data
}

fn render_icon(status: TrayStatus, frame: usize) -> Icon {
    Icon::from_rgba(render_frame(status, frame), ICON_SIZE, ICON_SIZE)
        .expect("Inline icon size is fixed, so construction cannot fail")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn menu_ids_round_trip() {
        for action in [
            TrayAction::OpenPanel,
            TrayAction::OpenConversation,
            TrayAction::OpenManagement,
            TrayAction::Quit,
        ] {
            assert_eq!(TrayAction::from_menu_id(action.menu_id()), Some(action));
        }
        assert_eq!(TrayAction::from_menu_id("unknown"), None);
    }

    #[test]
    fn headline_stays_one_short_line() {
        assert_eq!(compact_headline("# Done\n\nNothing left to do"), "Done");
        assert_eq!(compact_headline("   "), "");
        let long = "a".repeat(HEADLINE_MAX + 20);
        let short = compact_headline(&long);
        assert_eq!(short.chars().count(), HEADLINE_MAX + 1);
        assert!(short.ends_with('…'));
    }

    #[test]
    fn headline_shows_when_quiet_and_status_wins_when_busy() {
        assert_eq!(
            compose_title(TrayStatus::Success, Some("Cleaned 12 files"), 0),
            "Relay · Cleaned 12 files"
        );
        assert_eq!(
            compose_title(TrayStatus::Idle, Some("Cleaned 12 files"), 0),
            "Relay · Cleaned 12 files"
        );
        assert_eq!(
            compose_title(TrayStatus::Working, Some("Cleaned 12 files"), 0),
            TrayStatus::Working.title(0)
        );
        assert_eq!(
            compose_title(TrayStatus::Idle, None, 0),
            TrayStatus::Idle.title(0)
        );
    }

    #[test]
    fn every_status_renders_all_frames() {
        for status in [
            TrayStatus::Idle,
            TrayStatus::Thinking,
            TrayStatus::Working,
            TrayStatus::Attention,
            TrayStatus::Paused,
            TrayStatus::Success,
            TrayStatus::Error,
        ] {
            for frame in 0..status.frames() {
                let rgba = render_frame(status, frame);
                assert_eq!(rgba.len(), (ICON_SIZE * ICON_SIZE * 4) as usize);
                assert!(
                    rgba.chunks(4).any(|px| px[3] > 0),
                    "{status:?} frame is fully transparent"
                );
            }
        }
    }

    #[test]
    fn animated_statuses_actually_move() {
        for status in [
            TrayStatus::Thinking,
            TrayStatus::Working,
            TrayStatus::Attention,
            TrayStatus::Paused,
            TrayStatus::Success,
            TrayStatus::Error,
        ] {
            let first = render_frame(status, 0);
            let middle = render_frame(status, status.frames() / 2);
            assert_ne!(first, middle, "{status:?} animation frames do not change");
        }
    }

    #[test]
    fn idle_is_static_and_transient_states_expire() {
        assert!(TrayStatus::Idle.frame_duration().is_none());
        assert!(TrayStatus::Idle.hold().is_none());
        assert!(TrayStatus::Success.hold().is_some());
        assert!(TrayStatus::Error.hold().is_some());
        assert!(TrayStatus::Thinking.hold().is_none());
    }

    #[test]
    fn animation_advances_frames_and_settles() {
        let start = Instant::now();
        let mut anim = Animation::new(start);
        assert!(!anim.needs_tick(), "the idle state should not spin");

        assert!(anim.set_status(TrayStatus::Working, start));
        assert!(anim.needs_tick());
        assert!(
            !anim.tick(start),
            "frame interval not reached yet, should not repaint"
        );
        assert!(
            anim.tick(start + Duration::from_millis(60)),
            "should advance one frame once due"
        );

        // One-shot effect: plays to the last frame, then settles back to idle after the hold.
        let mut anim = Animation::new(start);
        anim.set_status(TrayStatus::Success, start);
        for step in 1..TrayStatus::Success.frames() {
            let now = start + Duration::from_millis(70 * step as u64);
            assert!(anim.tick(now), "frame {step} should advance");
        }
        assert_eq!(anim.status(), TrayStatus::Success);
        assert!(
            anim.needs_tick(),
            "still needs to settle back during the hold"
        );
        assert!(
            !anim.tick(start + Duration::from_millis(1000)),
            "no further change after the last frame"
        );
        assert!(anim.tick(start + Duration::from_millis(2400)));
        assert_eq!(anim.status(), TrayStatus::Idle);
        assert!(!anim.needs_tick());
    }

    #[test]
    fn animated_titles_keep_a_fixed_width_suffix() {
        for status in [
            TrayStatus::Thinking,
            TrayStatus::Working,
            TrayStatus::Attention,
        ] {
            let symbols = status.title_symbols();
            assert!(
                !symbols.is_empty(),
                "{status:?} should have activity symbols"
            );
            for symbol in symbols {
                assert_eq!(
                    symbol.chars().count(),
                    1,
                    "symbol {symbol} of {status:?} is not a single character"
                );
            }
            let titles: Vec<String> = (0..symbols.len()).map(|i| status.title(i)).collect();
            assert!(
                titles
                    .iter()
                    .all(|t| t.chars().count() == titles[0].chars().count()),
                "{status:?} titles have inconsistent lengths: {titles:?}"
            );
            assert!(
                titles
                    .iter()
                    .collect::<std::collections::HashSet<_>>()
                    .len()
                    > 1,
                "{status:?} titles do not change"
            );
        }
    }

    #[test]
    fn static_states_do_not_animate_their_title() {
        for status in [
            TrayStatus::Idle,
            TrayStatus::Paused,
            TrayStatus::Success,
            TrayStatus::Error,
        ] {
            assert!(
                status.title_symbols().is_empty(),
                "{status:?} should not have activity symbols"
            );
            assert!(status.symbol_interval().is_none());
            assert_eq!(status.title(3), status.label());
        }
    }

    #[test]
    fn every_status_has_a_short_label() {
        for status in [
            TrayStatus::Idle,
            TrayStatus::Thinking,
            TrayStatus::Working,
            TrayStatus::Attention,
            TrayStatus::Paused,
            TrayStatus::Success,
            TrayStatus::Error,
        ] {
            let chars = status.label().chars().count();
            assert!(
                (1..=8).contains(&chars),
                "{status:?} title length is unsuitable: {}",
                status.label()
            );
        }
    }
}
