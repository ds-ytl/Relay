//! Cross-platform desktop capability wrappers.
//!
//! Uses existing crates instead of hand-written platform FFI:
//! - Clipboard: `arboard`
//! - Windows and screenshots: `xcap`
//! - Global hotkeys: `global-hotkey`
//! - Tray: `tray-icon` + `muda`
//! - Foreground input: see `relay-input-tool` (computer tool, enabled by default)

pub mod clipboard;
pub mod hotkey;
pub mod screen;
pub mod shell;
pub mod tray;

pub use clipboard::{read_text as clipboard_text, write_text as set_clipboard_text};
pub use hotkey::{global_hotkey_events, register as register_hotkey, HotkeyHandle, HotkeyParse};
pub use screen::{
    capture_app_png, capture_primary_png, capture_primary_rgba, capture_user_window_png,
    front_window, is_shell_app, last_user_window, remember_user_window, user_front_window, Capture,
    FrontWindow, ScreenError, UserWindow,
};
pub use shell::{spec as shell_spec, ShellSpec};
pub use tray::{install as install_tray, Tray, TrayAction, TrayError, TrayStatus};
