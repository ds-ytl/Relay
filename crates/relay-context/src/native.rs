//! Native context: front window, clipboard and screenshots all go through the
//! cross-platform crates in `relay-platform`.
//!
//! - Front app and window title: `xcap` (macOS / Windows / Linux)
//! - Clipboard: `arboard`
//! - Screenshot: `xcap` captures the primary display
//!
//! Only "files selected in Finder" stays macOS-specific (via AppleScript);
//! other platforms leave it empty, which does not affect task execution.

use std::path::PathBuf;

use relay_core::{ContextSource, EnvironmentSnapshot};

use crate::{finalize, CaptureOptions, ContextError, ContextProvider};

#[cfg(target_os = "macos")]
use crate::run_osascript;

/// Cross-platform native context.
#[derive(Debug, Default)]
pub struct NativeContextProvider;

impl NativeContextProvider {
    pub fn new() -> Self {
        Self
    }

    /// Window titles of apps such as WeChat or Finder may look like
    /// "[Direct chat] Zhang San"; keep the raw text and let the Router match it.
    #[cfg(target_os = "macos")]
    fn selected_files(&self) -> Vec<PathBuf> {
        let script = r#"tell application "Finder"
  try
    set theFiles to selection as alias list
    set out to ""
    repeat with f in theFiles
      set out to out & (POSIX path of f) & linefeed
    end repeat
    return out
  on error
    return ""
  end try
end tell"#;
        run_osascript(script)
            .map(|out| {
                out.lines()
                    .map(str::trim)
                    .filter(|line| !line.is_empty())
                    .map(PathBuf::from)
                    .collect()
            })
            .unwrap_or_default()
    }

    #[cfg(not(target_os = "macos"))]
    fn selected_files(&self) -> Vec<PathBuf> {
        Vec::new()
    }
}

impl ContextProvider for NativeContextProvider {
    fn capture(&self, options: &CaptureOptions) -> Result<EnvironmentSnapshot, ContextError> {
        let front = relay_platform::front_window();
        let (app, title) = match front.as_ref() {
            Some(window) => (window.app.clone(), window.title.clone()),
            None => {
                tracing::warn!(
                    "Cannot read the front window, possibly missing accessibility/screen \
                     recording permission; continuing with empty context"
                );
                (String::new(), String::new())
            }
        };

        let mut snapshot = EnvironmentSnapshot {
            app_name: app,
            window_title: title,
            selected_files: self.selected_files(),
            source: ContextSource::Native,
            ..Default::default()
        };

        if options.include_clipboard {
            snapshot.clipboard_excerpt = relay_platform::clipboard_text()
                .map(|text| text.trim().to_string())
                .filter(|text| !text.is_empty());
        }

        if options.include_screenshot {
            if let Some(dir) = options.screenshot_dir.as_deref() {
                match relay_platform::capture_primary_png(dir) {
                    Ok(path) => snapshot.screenshots.push(path),
                    Err(err) => tracing::warn!(%err, "Screenshot failed, ignoring"),
                }
            }
        }

        Ok(finalize(snapshot, options))
    }

    fn name(&self) -> &'static str {
        "native"
    }
}
