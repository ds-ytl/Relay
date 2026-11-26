//! System-level isolation (macOS sandbox-exec).
//!
//! Policy: deny by default, only granting write access to the directories execution needs;
//! the network is denied by default and only opened explicitly by ExecutionPolicy.

use std::path::{Path, PathBuf};

use relay_core::ExecutionPolicy;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SandboxError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("sandbox unavailable: {0}")]
    Unavailable(String),
}

#[derive(Debug, Clone)]
pub struct Sandbox {
    profile: PathBuf,
}

impl Sandbox {
    /// Generate the sandbox profile; returns None when not on macOS or when disabled (the
    /// Executor then runs directly).
    pub fn prepare(
        policy: &ExecutionPolicy,
        task_dir: &Path,
        profile_dir: &Path,
    ) -> Result<Option<Self>, SandboxError> {
        if !policy.sandbox || !is_supported() {
            return Ok(None);
        }
        std::fs::create_dir_all(profile_dir)?;
        let profile = profile_dir.join("sandbox.sb");
        std::fs::write(&profile, profile_text(policy, task_dir))?;
        Ok(Some(Self { profile }))
    }

    pub fn profile_path(&self) -> &Path {
        &self.profile
    }
}

pub fn is_supported() -> bool {
    cfg!(target_os = "macos") && Path::new("/usr/bin/sandbox-exec").exists()
}

/// Generate the SBPL profile. All path arguments use literal/subpath; no glob expansion is done.
pub fn profile_text(policy: &ExecutionPolicy, task_dir: &Path) -> String {
    let mut out = String::new();
    out.push_str("(version 1)\n");
    out.push_str("(deny default)\n");
    out.push_str("(allow process-exec)\n");
    out.push_str("(allow process-fork)\n");
    out.push_str("(allow signal)\n");
    out.push_str("(allow sysctl-read)\n");
    out.push_str("(allow mach-lookup)\n");
    out.push_str("(allow ipc-posix-shm)\n");
    out.push_str("(allow file-read*)\n");

    out.push_str("(allow file-write*\n");
    for path in write_paths(policy, task_dir) {
        out.push_str(&format!(
            "  (subpath \"{}\")\n",
            escape(&path.to_string_lossy())
        ));
    }
    out.push_str(")\n");

    if policy.allow_network {
        out.push_str("(allow network*)\n");
    } else {
        out.push_str("(deny network*)\n");
    }
    out
}

fn write_paths(policy: &ExecutionPolicy, task_dir: &Path) -> Vec<PathBuf> {
    let mut paths = vec![
        task_dir.to_path_buf(),
        std::env::temp_dir(),
        PathBuf::from("/private/tmp"),
        PathBuf::from("/private/var/folders"),
    ];
    paths.extend(policy.allowed_dirs.iter().cloned());
    paths.dedup();
    paths
}

fn escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_by_default() {
        let policy = ExecutionPolicy::default();
        let dir = std::env::temp_dir();
        assert!(
            Sandbox::prepare(&policy, &dir, &dir)
                .expect("prepare")
                .is_none(),
            "the default policy should not create a sandbox, so the script stays unrestricted"
        );
    }

    #[test]
    fn denies_network_by_default() {
        let policy = ExecutionPolicy::default();
        let text = profile_text(&policy, Path::new("/tmp/task"));
        assert!(text.contains("(deny default)"));
        assert!(text.contains("(deny network*)"));
        assert!(text.contains("/tmp/task"));
    }

    #[test]
    fn allows_network_when_policy_permits() {
        let mut policy = ExecutionPolicy::default();
        policy.allow_network = true;
        let text = profile_text(&policy, Path::new("/tmp/task"));
        assert!(text.contains("(allow network*)"));
    }

    #[test]
    fn includes_authorized_dirs_as_writable() {
        let mut policy = ExecutionPolicy::default();
        policy.allow_dir("/Users/me/Projects/demo");
        let text = profile_text(&policy, Path::new("/tmp/task"));
        assert!(text.contains("/Users/me/Projects/demo"));
    }
}
