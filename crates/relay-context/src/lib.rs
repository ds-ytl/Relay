//! Context awareness engine: captures the front app, window title, selected files and
//! clipboard only when the user triggers it.
//!
//! No continuous screen recording, no resident listeners; screenshots must be enabled
//! explicitly.

use std::path::PathBuf;

use chrono::Utc;
use relay_core::{ContextSource, EnvironmentSnapshot};
use thiserror::Error;

pub mod manual;
pub mod native;

pub use manual::ManualContextProvider;
pub use native::NativeContextProvider;

#[derive(Debug, Error)]
pub enum ContextError {
    #[error("Failed to run system command: {0}")]
    Command(String),
    #[error("Insufficient permission to capture context: {0}")]
    Permission(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Clone, Default)]
pub struct CaptureOptions {
    pub authorized_dirs: Vec<PathBuf>,
    pub include_clipboard: bool,
    pub include_screenshot: bool,
    /// Directory to save screenshots in (the task directory); no screenshot when empty.
    pub screenshot_dir: Option<PathBuf>,
    pub clipboard_max_chars: usize,
}

pub trait ContextProvider: Send + Sync {
    fn capture(&self, options: &CaptureOptions) -> Result<EnvironmentSnapshot, ContextError>;
    fn name(&self) -> &'static str;
}

/// Default context provider: always the cross-platform native implementation
/// (xcap + arboard); capture failures are not errors, the context is just empty.
pub fn default_provider() -> Box<dyn ContextProvider> {
    Box::new(NativeContextProvider::new())
}

pub(crate) fn finalize(
    mut snapshot: EnvironmentSnapshot,
    options: &CaptureOptions,
) -> EnvironmentSnapshot {
    snapshot.captured_at = Some(Utc::now());
    snapshot.authorized_dirs = options.authorized_dirs.clone();
    if let Some(clip) = snapshot.clipboard_excerpt.take() {
        let clipped = truncate_chars(clip.trim(), options.clipboard_max_chars.max(120));
        if !clipped.is_empty() {
            snapshot.clipboard_excerpt = Some(clipped);
        }
    }
    snapshot
}

pub(crate) fn truncate_chars(value: &str, max: usize) -> String {
    if value.chars().count() <= max {
        return value.to_string();
    }
    let mut out: String = value.chars().take(max).collect();
    out.push_str("… (truncated)");
    out
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
pub(crate) fn run_osascript(script: &str) -> Result<String, ContextError> {
    let output = std::process::Command::new("osascript")
        .args(["-e", script])
        .output()
        .map_err(|e| ContextError::Command(format!("osascript: {e}")))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(ContextError::Command(if stderr.is_empty() {
            "osascript returned a non-zero status".into()
        } else {
            stderr
        }));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

#[allow(dead_code)]
pub(crate) fn empty_snapshot(source: ContextSource) -> EnvironmentSnapshot {
    EnvironmentSnapshot {
        source,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncates_long_clipboard() {
        let text = "a".repeat(500);
        let clipped = truncate_chars(&text, 200);
        assert!(clipped.chars().count() < 220);
        assert!(clipped.ends_with("(truncated)"));
    }

    #[test]
    fn short_text_is_untouched() {
        assert_eq!(truncate_chars("hello", 100), "hello");
    }

    #[test]
    fn finalize_applies_authorized_dirs() {
        let snapshot = EnvironmentSnapshot::default();
        let options = CaptureOptions {
            authorized_dirs: vec!["/tmp".into()],
            clipboard_max_chars: 10,
            ..Default::default()
        };
        let out = finalize(snapshot, &options);
        assert_eq!(out.authorized_dirs, vec![PathBuf::from("/tmp")]);
        assert!(out.captured_at.is_some());
    }
}
