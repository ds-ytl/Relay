use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{CoreError, Result};

/// Execution policy: injected into the executor by the runtime; the model cannot relax it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPolicy {
    /// Folders that may be read and written (including the task directory itself).
    pub allowed_dirs: Vec<PathBuf>,
    /// Whether Bash may access the network (denied by default).
    pub allow_network: bool,
    /// Timeout for a single execution turn.
    pub timeout_secs: u64,
    /// Maximum execution turns per task.
    pub max_rounds: u32,
    /// Collection limit for stdout / stderr of one turn.
    pub max_output_bytes: usize,
    /// Whether the user must confirm before execution.
    pub require_confirmation: bool,
    /// Whether to enable system-level isolation (macOS sandbox-exec). Off by default: scripts are not
    /// restricted by path or network rules.
    pub sandbox: bool,
    /// Whether artifacts may be opened automatically.
    pub auto_open: bool,
}

impl Default for ExecutionPolicy {
    fn default() -> Self {
        Self {
            allowed_dirs: Vec::new(),
            allow_network: false,
            timeout_secs: 300,
            max_rounds: 24,
            max_output_bytes: 128 * 1024,
            require_confirmation: true,
            sandbox: false,
            auto_open: false,
        }
    }
}

impl ExecutionPolicy {
    pub fn allow_dir(&mut self, dir: impl Into<PathBuf>) {
        let dir = dir.into();
        if !self.allowed_dirs.contains(&dir) {
            self.allowed_dirs.push(dir);
        }
    }

    /// Check whether a path falls inside an authorized folder. The path need not exist: the longest
    /// existing ancestor is canonicalized first.
    pub fn is_path_allowed(&self, path: &Path) -> bool {
        let candidate = normalize_existing_prefix(path);
        self.allowed_dirs.iter().any(|dir| {
            let root = normalize_existing_prefix(dir);
            candidate == root || candidate.starts_with(&root)
        })
    }

    pub fn check_path(&self, path: &Path) -> Result<()> {
        if self.is_path_allowed(path) {
            Ok(())
        } else {
            Err(CoreError::UnauthorizedPath(path.display().to_string()))
        }
    }
}

/// Canonicalize the existing part of a path and append the rest unchanged, so that the macOS
/// /var -> /private/var mapping does not cause false negatives.
fn normalize_existing_prefix(path: &Path) -> PathBuf {
    let mut existing = path.to_path_buf();
    let mut suffix: Vec<std::ffi::OsString> = Vec::new();
    loop {
        if let Ok(canonical) = existing.canonicalize() {
            let mut out = canonical;
            for part in suffix.iter().rev() {
                out.push(part);
            }
            return out;
        }
        match (existing.file_name(), existing.parent()) {
            (Some(name), Some(parent)) if parent != existing => {
                suffix.push(name.to_os_string());
                existing = parent.to_path_buf();
            }
            _ => return path.to_path_buf(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_nested_paths_and_rejects_siblings() {
        let root = std::env::temp_dir().join("relay-policy-test");
        std::fs::create_dir_all(root.join("project")).unwrap();
        let mut policy = ExecutionPolicy::default();
        policy.allow_dir(&root);

        assert!(policy.is_path_allowed(&root.join("project/report.xlsx")));
        assert!(!policy.is_path_allowed(&std::env::temp_dir().join("other/report.xlsx")));
    }

    #[test]
    fn denies_everything_when_no_dirs_are_granted() {
        let policy = ExecutionPolicy::default();
        assert!(!policy.is_path_allowed(Path::new("/tmp")));
    }
}
