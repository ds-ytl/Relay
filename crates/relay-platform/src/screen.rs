//! Windows and screenshots (xcap, supports macOS / Windows / Linux).

use std::path::{Path, PathBuf};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ScreenError {
    #[error("No available display")]
    NoMonitor,
    #[error("Screenshot failed: {0}")]
    Capture(String),
    #[error("Failed to save screenshot: {0}")]
    Save(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FrontWindow {
    pub app: String,
    pub title: String,
}

impl FrontWindow {
    pub fn is_empty(&self) -> bool {
        self.app.trim().is_empty() && self.title.trim().is_empty()
    }
}

/// Current front window; falls back to any visible window when it cannot be read.
pub fn front_window() -> Option<FrontWindow> {
    let windows = xcap::Window::all().ok()?;
    let mut fallback: Option<FrontWindow> = None;
    for window in windows {
        let app = window.app_name().unwrap_or_default();
        let title = window.title().unwrap_or_default();
        if app.trim().is_empty() && title.trim().is_empty() {
            continue;
        }
        let candidate = FrontWindow { app, title };
        if window.is_focused().unwrap_or(false) {
            return Some(candidate);
        }
        if fallback.is_none() {
            fallback = Some(candidate);
        }
    }
    fallback
}

/// Relay itself and the system shell: these windows cannot represent "what the user is looking at".
const SHELL_APPS: &[&str] = &[
    "relay",
    "Relay",
    "Dock",
    // Localized shell names are kept as escapes so the filter still works on non-English systems.
    "\u{7a0b}\u{5e8f}\u{575e}",
    "WindowManager",
    "Window Server",
    "SystemUIServer",
    "Control Center",
    "\u{63a7}\u{5236}\u{4e2d}\u{5fc3}",
    "Notification Center",
    "\u{901a}\u{77e5}\u{4e2d}\u{5fc3}",
    "Wallpaper",
    "\u{5899}\u{7eb8}",
    "loginwindow",
];

/// Windows that are too small or carry no title (Dock, desktop widgets, system overlays) are not "the app the user is looking at".
fn usable_window(window: &xcap::Window, app: &str, title: &str) -> bool {
    if is_shell_app(app) || title.trim().is_empty() {
        return false;
    }
    if window.is_minimized().unwrap_or(false) {
        return false;
    }
    let width = window.width().unwrap_or(0);
    let height = window.height().unwrap_or(0);
    width >= 320 && height >= 200
}

/// Whether the app name belongs to Relay itself or the system shell.
pub fn is_shell_app(app: &str) -> bool {
    let app = app.trim();
    !app.is_empty()
        && SHELL_APPS
            .iter()
            .any(|known| known.eq_ignore_ascii_case(app))
}

/// Remember "what the user was looking at" before the panel opens.
///
/// The panel grabs focus as soon as it appears, so asking the system "who is in front?" afterwards only returns relay itself;
/// so record this window before the panel is created and use it as a fallback in the understanding phase.
static LAST_USER_WINDOW: std::sync::Mutex<Option<FrontWindow>> = std::sync::Mutex::new(None);

/// Record the current user window (call before the panel pops up).
pub fn remember_user_window() {
    if let Some(window) = user_front_window() {
        if let Ok(mut slot) = LAST_USER_WINDOW.lock() {
            *slot = Some(window);
        }
    }
}

/// User window remembered before the panel popped up.
pub fn last_user_window() -> Option<FrontWindow> {
    LAST_USER_WINDOW.lock().ok().and_then(|slot| slot.clone())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserWindow {
    pub app: String,
    pub title: String,
    pub path: PathBuf,
}

/// Pick a window "the user is looking at": prefer the focused window, but exclude the Relay panel and the system shell.
///
/// With an empty instruction the Relay panel itself grabs focus, and treating that focused window as the user's location
/// would leave the understanding agent looking at itself (in testing this pushed confidence down to 0.3 and made it ask the user back).
fn pick_user_window() -> Option<(xcap::Window, String, String)> {
    let windows = xcap::Window::all().ok()?;
    let mut fallback: Option<(xcap::Window, String, String)> = None;
    for window in windows {
        let app = window.app_name().unwrap_or_default();
        let title = window.title().unwrap_or_default();
        if !usable_window(&window, &app, &title) {
            continue;
        }
        if window.is_focused().unwrap_or(false) {
            return Some((window, app, title));
        }
        // No focused window (the panel is in front): take the first usable window in z-order.
        if fallback.is_none() {
            fallback = Some((window, app, title));
        }
    }
    fallback
}

/// The front window the user is really looking at (Relay itself excluded). Returns `None` when unavailable.
pub fn user_front_window() -> Option<FrontWindow> {
    pick_user_window().map(|(_, app, title)| FrontWindow { app, title })
}

/// Capture the window the user is looking at; falls back to a full-screen screenshot when no specific window is found.
pub fn capture_user_window_png(dir: &Path) -> Result<UserWindow, ScreenError> {
    std::fs::create_dir_all(dir)?;
    if let Some((window, app, title)) = pick_user_window() {
        if let Ok(image) = window.capture_image() {
            let stamp = chrono::Local::now().format("%Y%m%d-%H%M%S-%3f");
            let path = dir.join(format!("window-{stamp}.png"));
            image
                .save(&path)
                .map_err(|err| ScreenError::Save(err.to_string()))?;
            return Ok(UserWindow { app, title, path });
        }
    }
    let path = capture_primary_png(dir)?;
    Ok(UserWindow {
        app: String::new(),
        title: String::new(),
        path,
    })
}

/// Capture the primary display into `dir` and return the image path.
/// One captured display: RGBA8 pixels, row-major, top-left origin.
#[derive(Debug, Clone)]
pub struct Capture {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Grab the primary display into memory.
///
/// Callers that hand the pixels straight to a renderer skip PNG encoding, which costs well over a
/// second on a Retina display; only paths that really need a file on disk encode one.
pub fn capture_primary_rgba() -> Result<Capture, ScreenError> {
    let started = std::time::Instant::now();
    let monitors = xcap::Monitor::all().map_err(|err| ScreenError::Capture(err.to_string()))?;
    let monitor = monitors
        .iter()
        .find(|monitor| monitor.is_primary().unwrap_or(false))
        .or_else(|| monitors.first())
        .ok_or(ScreenError::NoMonitor)?;
    let image = monitor
        .capture_image()
        .map_err(|err| ScreenError::Capture(err.to_string()))?;
    let (width, height) = image.dimensions();
    tracing::debug!(
        ms = started.elapsed().as_millis(),
        width,
        height,
        "captured the primary display"
    );
    Ok(Capture {
        width,
        height,
        rgba: image.into_raw(),
    })
}

pub fn capture_primary_png(dir: &Path) -> Result<PathBuf, ScreenError> {
    let capture = capture_primary_rgba()?;
    std::fs::create_dir_all(dir)?;
    let image = image::RgbaImage::from_raw(capture.width, capture.height, capture.rgba)
        .ok_or_else(|| ScreenError::Capture("screenshot buffer size mismatch".into()))?;
    let stamp = chrono::Local::now().format("%Y%m%d-%H%M%S");
    let path = dir.join(format!("context-{stamp}.png"));
    image
        .save(&path)
        .map_err(|err| ScreenError::Save(err.to_string()))?;
    Ok(path)
}

/// Capture a given app: the app name is case-insensitive and partially matched; the largest visible window wins.
pub fn capture_app_png(app: &str, dir: &Path) -> Result<UserWindow, ScreenError> {
    let wanted = app.trim().to_lowercase();
    if wanted.is_empty() {
        return Err(ScreenError::Capture("App name is empty".into()));
    }
    std::fs::create_dir_all(dir)?;
    let windows = xcap::Window::all().map_err(|err| ScreenError::Capture(err.to_string()))?;
    let mut best: Option<(xcap::Window, String, String, u64)> = None;
    for window in windows {
        let name = window.app_name().unwrap_or_default();
        let title = window.title().unwrap_or_default();
        if !usable_window(&window, &name, &title) || !name.to_lowercase().contains(&wanted) {
            continue;
        }
        let area = u64::from(window.width().unwrap_or(0)) * u64::from(window.height().unwrap_or(0));
        let bigger = best
            .as_ref()
            .map(|(_, _, _, found)| area > *found)
            .unwrap_or(true);
        if bigger {
            best = Some((window, name, title, area));
        }
    }
    let Some((window, app, title, _)) = best else {
        return Err(ScreenError::Capture(format!(
            "No visible window found for \"{app}\""
        )));
    };
    let image = window
        .capture_image()
        .map_err(|err| ScreenError::Capture(err.to_string()))?;
    let stamp = chrono::Local::now().format("%Y%m%d-%H%M%S-%3f");
    let path = dir.join(format!("app-{stamp}.png"));
    image
        .save(&path)
        .map_err(|err| ScreenError::Save(err.to_string()))?;
    Ok(UserWindow { app, title, path })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_apps_are_recognized() {
        assert!(is_shell_app("relay"));
        assert!(is_shell_app(" Dock "));
        assert!(is_shell_app("Notification Center"));
        assert!(is_shell_app("\u{7a0b}\u{5e8f}\u{575e}"));
        assert!(!is_shell_app("WeChat"));
        assert!(!is_shell_app(""));
    }

    #[test]
    fn lists_front_window_without_panicking() {
        // Returning None is allowed when there is no window or no screen recording permission.
        if let Some(window) = front_window() {
            assert!(!window.is_empty());
        }
    }

    #[tokio::test]
    async fn capture_writes_png_or_reports_permission_error() {
        let dir = tempfile::tempdir().unwrap();
        match capture_primary_png(dir.path()) {
            Ok(path) => {
                assert!(path.extension().is_some_and(|ext| ext == "png"));
                assert!(std::fs::metadata(&path).unwrap().len() > 0);
            }
            Err(err) => {
                // Without screen recording permission / a display, explaining why is enough.
                assert!(!err.to_string().is_empty());
            }
        }
    }

    #[test]
    fn app_capture_requires_a_name() {
        let dir = tempfile::tempdir().unwrap();
        assert!(capture_app_png("  ", dir.path()).is_err());
    }
}
