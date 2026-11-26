use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Environment snapshot captured when a hotkey fires.
///
/// Captured only on an explicit user trigger, never by continuous screen recording; `screenshots`
/// is only populated once the user turns screenshots on.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnvironmentSnapshot {
    pub captured_at: Option<DateTime<Utc>>,
    /// Frontmost app name, e.g. "WeChat", "Finder", "Visual Studio Code".
    pub app_name: String,
    /// Frontmost window title or document name.
    pub window_title: String,
    /// The first few characters of the clipboard (truncated), used as a text clue.
    pub clipboard_excerpt: Option<String>,
    /// Files selected in Finder.
    pub selected_files: Vec<PathBuf>,
    /// Authorized folders (the roots the workspace may access).
    pub authorized_dirs: Vec<PathBuf>,
    /// Screenshot file paths (captured only when authorized and enabled).
    pub screenshots: Vec<PathBuf>,
    pub source: ContextSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextSource {
    /// Native system sensing (macOS Accessibility / AppleScript).
    Native,
    /// Filled in manually by the user, or injected by tests.
    Manual,
    /// Empty context when no sensing capability is available.
    Empty,
}

impl Default for ContextSource {
    fn default() -> Self {
        Self::Empty
    }
}

impl EnvironmentSnapshot {
    /// One-line environment summary, used by the confirmation card and routing hints.
    pub fn summary_line(&self) -> String {
        match (self.app_name.as_str(), self.window_title.as_str()) {
            ("", "") => "Current context: unknown".to_string(),
            (app, "") => format!("Current context: {app}"),
            ("", title) => format!("Current context: {title}"),
            (app, title) => format!("Current context: {app} · {title}"),
        }
    }

    /// Compact text description for the model.
    pub fn to_prompt(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "Frontmost app: {}\n",
            empty_as_unknown(&self.app_name)
        ));
        out.push_str(&format!(
            "Window title: {}\n",
            empty_as_unknown(&self.window_title)
        ));
        if !self.selected_files.is_empty() {
            out.push_str("Selected files:\n");
            for path in &self.selected_files {
                out.push_str(&format!("- {}\n", path.display()));
            }
        }
        if let Some(clip) = self
            .clipboard_excerpt
            .as_deref()
            .filter(|s| !s.trim().is_empty())
        {
            out.push_str(&format!("Clipboard excerpt:\n{}\n", clip));
        }
        if !self.screenshots.is_empty() {
            out.push_str(
                "Screenshots captured (read with read_image when their content matters):\n",
            );
            for path in &self.screenshots {
                out.push_str(&format!("- {}\n", path.display()));
            }
        }
        if !self.authorized_dirs.is_empty() {
            out.push_str("Authorized folders:\n");
            for dir in &self.authorized_dirs {
                out.push_str(&format!("- {}\n", dir.display()));
            }
        }
        out
    }

    pub fn is_empty(&self) -> bool {
        self.app_name.is_empty() && self.window_title.is_empty()
    }
}

fn empty_as_unknown(value: &str) -> &str {
    if value.trim().is_empty() {
        "unknown"
    } else {
        value
    }
}
