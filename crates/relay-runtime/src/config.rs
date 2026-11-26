use std::path::{Path, PathBuf};
use std::sync::Arc;

use relay_core::{ExecutionPolicy, TaskId};
use relay_openai::WebTools;

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Application data directory: SQLite, task directories, logs and the state helper.
    pub data_dir: PathBuf,
    pub policy: ExecutionPolicy,
    pub include_clipboard: bool,
    pub include_screenshot: bool,
    /// Optional separate vision model; when unset, images go straight to the main model.
    pub vision_model: Option<String>,
    /// Capture a screenshot of the current window before understanding starts and put it in the first turn's context (saves one screenshot round trip).
    /// Off by default: the CLI and tests do not want implicit screenshots; the desktop app enables it in its own config.
    pub prefetch_screen: bool,
    pub auto_open: bool,
    /// Whether to send desktop notifications (tests and batch runs may disable them).
    pub notifications: bool,
    pub max_rounds: u32,
    /// Maximum exploration steps for the understanding agent (tool-call turns).
    pub max_understand_steps: u32,
    /// Maximum consecutive tool calls the executor agent may make in one turn.
    pub max_tool_steps_per_round: u32,
    /// Maximum tool steps for the verifier agent.
    pub max_verify_steps: u32,
    /// Model context window in tokens; 0 disables accounting and compaction.
    pub context_window: u32,
    /// When the estimated input exceeds this fraction of the window, summarize history with the model (default 0.9).
    pub context_compact_ratio: f32,
    /// Skip confirmation for CLI / automation; the GUI keeps this false.
    pub auto_confirm: bool,
    /// Web search / fetch client; `None` without an API key (the tools then say they are unavailable).
    pub web_tools: Option<Arc<WebTools>>,
}

impl RuntimeConfig {
    pub fn new(data_dir: impl Into<PathBuf>) -> Self {
        let mut policy = ExecutionPolicy::default();
        policy.max_rounds = 24;
        policy.timeout_secs = 300;
        Self {
            data_dir: data_dir.into(),
            policy,
            include_clipboard: true,
            include_screenshot: false,
            vision_model: None,
            prefetch_screen: false,
            auto_open: false,
            notifications: true,
            max_rounds: 24,
            max_understand_steps: 8,
            max_tool_steps_per_round: 12,
            max_verify_steps: 6,
            context_window: 256_000,
            context_compact_ratio: 0.9,
            auto_confirm: false,
            web_tools: None,
        }
    }

    /// Token budget for injected session history: 1/40 of the window (~6.4k for a 256k window), clamped.
    pub fn dialogue_budget(&self) -> u32 {
        if self.context_window > 0 {
            (self.context_window / 40).clamp(2_000, 8_000)
        } else {
            6_000
        }
    }

    /// Token budget for the execution transcript: 1/48 of the window (~5.3k for a 256k window).
    pub fn transcript_budget(&self) -> u32 {
        if self.context_window > 0 {
            (self.context_window / 48).clamp(2_000, 8_000)
        } else {
            5_000
        }
    }

    pub fn task_dir(&self, id: TaskId) -> PathBuf {
        self.data_dir.join("tasks").join(id.to_string())
    }

    pub fn workspaces_dir(&self) -> PathBuf {
        self.data_dir.join("workspaces")
    }

    pub fn db_path(&self) -> PathBuf {
        self.data_dir.join("relay.sqlite3")
    }

    /// Instance pid file: written while the GUI runs; the CLI reads it on startup to tell whether another instance is alive.
    pub fn pid_path(&self) -> PathBuf {
        self.data_dir.join("relay.pid")
    }

    pub fn state_helper(&self) -> PathBuf {
        self.data_dir.join("bin").join("relay-state")
    }

    pub fn ensure_dirs(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.data_dir)?;
        std::fs::create_dir_all(self.workspaces_dir())?;
        std::fs::create_dir_all(self.data_dir.join("tasks"))?;
        std::fs::create_dir_all(self.data_dir.join("bin"))?;
        Ok(())
    }

    /// Stable, readable directory name used when a workspace is created automatically.
    pub fn workspace_root(&self, name: &str, id: relay_core::WorkspaceId) -> PathBuf {
        self.workspaces_dir()
            .join(format!("{}-{}", slugify(name), &id.to_string()[..8]))
    }
}

pub fn slugify(name: &str) -> String {
    let slug: String = name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect();
    let mut slug = slug.trim_matches('-').to_string();
    while slug.contains("--") {
        slug = slug.replace("--", "-");
    }
    if slug.is_empty() {
        "workspace".to_string()
    } else {
        slug.chars().take(48).collect()
    }
}

/// Ensure the path exists and stays inside the workspace root; used to validate artifact paths from the model.
pub fn resolve_within(root: &Path, candidate: &Path) -> Option<PathBuf> {
    let joined = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        root.join(candidate)
    };
    let root_norm = root.canonicalize().ok()?;
    let joined_norm = joined
        .parent()
        .and_then(|p| p.canonicalize().ok())
        .map(|p| p.join(joined.file_name().unwrap_or_default()))
        .unwrap_or(joined);
    (joined_norm == root_norm || joined_norm.starts_with(&root_norm)).then_some(joined_norm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slug_is_filesystem_safe() {
        assert_eq!(slugify("Project A / materials"), "Project-A-materials");
        assert_eq!(slugify("   "), "workspace");
    }

    #[test]
    fn resolves_relative_artifact_inside_root() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("out.txt");
        std::fs::write(&file, "x").unwrap();
        let resolved = resolve_within(dir.path(), Path::new("out.txt")).unwrap();
        assert_eq!(resolved, file.canonicalize().unwrap());
    }

    #[test]
    fn rejects_paths_outside_root() {
        let dir = tempfile::tempdir().unwrap();
        assert!(resolve_within(dir.path(), Path::new("../escape.txt")).is_none());
    }
}
