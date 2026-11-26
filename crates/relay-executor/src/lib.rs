//! Bash execution engine: one process per turn, a shared timeout and output cap, and an
//! independent state protocol read after execution.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use relay_core::{ControlSignal, ExecutionPolicy};
use relay_sandbox::Sandbox;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::io::AsyncReadExt;
use tokio::process::Command;

mod protocol;

pub use protocol::{read_control, resolve_phase, scan_artifacts, state_file};

#[derive(Debug, Error)]
pub enum ExecutorError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Path is not authorized: {0}")]
    Unauthorized(String),
    #[error("Failed to spawn process: {0}")]
    Spawn(String),
    #[error("Failed to parse state protocol: {0}")]
    Protocol(String),
    #[error("Failed to configure isolation: {0}")]
    Sandbox(#[from] relay_sandbox::SandboxError),
    /// The user pressed stop: the process group has been killed and this turn cannot count as
    /// an execution result.
    #[error("Execution cancelled")]
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct ExecutionRequest {
    pub script: String,
    pub cwd: PathBuf,
    pub task_dir: PathBuf,
    pub env: Vec<(String, String)>,
    pub timeout: Duration,
    pub turn: u32,
    /// Cancel flag: set when the user presses stop; the executor kills the running process group
    /// and returns immediately.
    pub cancel: Option<Arc<AtomicBool>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
    pub timed_out: bool,
    pub duration_ms: u64,
    pub stdout_truncated: bool,
    pub stderr_truncated: bool,
    /// Control signal read from the task directory after this turn; treated as Continue when absent.
    pub control: Option<ControlSignal>,
    pub script_path: Option<PathBuf>,
    pub sandboxed: bool,
}

impl ExecutionResult {
    pub fn succeeded(&self) -> bool {
        self.exit_code == Some(0) && !self.timed_out
    }

    pub fn output_tail(&self, max_chars: usize) -> String {
        let mut text = String::new();
        if !self.stdout.trim().is_empty() {
            text.push_str(self.stdout.trim_end());
        }
        if !self.stderr.trim().is_empty() {
            if !text.is_empty() {
                text.push('\n');
            }
            text.push_str("[stderr]\n");
            text.push_str(self.stderr.trim_end());
        }
        if text.chars().count() > max_chars {
            let skip = text.chars().count() - max_chars;
            text = text.chars().skip(skip).collect();
        }
        text
    }
}

#[derive(Debug, Clone)]
pub struct BashExecutor {
    policy: ExecutionPolicy,
    state_helper: Option<PathBuf>,
}

impl BashExecutor {
    pub fn new(policy: ExecutionPolicy) -> Self {
        Self {
            policy,
            state_helper: None,
        }
    }

    /// Install the `relay-state` helper script (called by the Runtime when a task is created).
    pub fn with_state_helper(mut self, path: impl Into<PathBuf>) -> Self {
        self.state_helper = Some(path.into());
        self
    }

    pub fn policy(&self) -> &ExecutionPolicy {
        &self.policy
    }

    pub async fn execute(
        &self,
        request: ExecutionRequest,
    ) -> Result<ExecutionResult, ExecutorError> {
        if !self.policy.is_path_allowed(&request.task_dir) {
            return Err(ExecutorError::Unauthorized(
                request.task_dir.display().to_string(),
            ));
        }
        if is_cancelled(&request.cancel) {
            return Err(ExecutorError::Cancelled);
        }
        std::fs::create_dir_all(&request.cwd)?;
        let relay_dir = request.task_dir.join(".relay");
        std::fs::create_dir_all(relay_dir.join("bin"))?;

        // Clear the previous turn's state so a model that forgets to write it does not inherit stale values.
        let state_path = state_file(&request.task_dir);
        let _ = std::fs::remove_file(&state_path);

        if let Some(helper) = &self.state_helper {
            let target = relay_dir
                .join("bin")
                .join(relay_platform::shell::helper_name("relay-state"));
            std::fs::copy(helper, &target)?;
            make_executable(&target)?;
        }

        let shell = relay_platform::shell::spec();
        let script_path = relay_dir.join(relay_platform::shell::script_name(request.turn));
        std::fs::write(&script_path, &request.script)?;

        let sandbox = Sandbox::prepare(&self.policy, &request.task_dir, &relay_dir)?;
        let mut command = relay_platform::shell::command(&script_path);
        if let Some(sandbox) = &sandbox {
            // Only macOS has sandbox-exec; here the shell is wrapped in the profile before running.
            let mut wrapped = Command::new("/usr/bin/sandbox-exec");
            wrapped.arg("-f").arg(sandbox.profile_path());
            wrapped
                .arg(shell.program)
                .args(shell.args)
                .arg(&script_path);
            command = wrapped;
        }

        command
            .current_dir(&request.cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env_clear()
            .env(
                shell.home_var,
                std::env::var(shell.home_var)
                    .unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().to_string()),
            )
            .env("LANG", "en_US.UTF-8")
            .env("TERM", "dumb")
            .env("RELAY_SHELL", shell.language)
            .env("RELAY_TASK_DIR", request.task_dir.clone())
            .env("RELAY_STATE_FILE", state_path.clone())
            .env("PATH", relay_platform::shell::env_path(&request.task_dir));

        for (key, value) in &request.env {
            command.env(key, value);
        }

        #[cfg(unix)]
        {
            command.process_group(0);
        }

        let started = Instant::now();
        let mut child = command
            .spawn()
            .map_err(|e| ExecutorError::Spawn(e.to_string()))?;

        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let cap = self.policy.max_output_bytes;
        let stdout_task = tokio::spawn(read_capped(stdout, cap));
        let stderr_task = tokio::spawn(read_capped(stderr, cap));

        let wait = tokio::select! {
            waited = tokio::time::timeout(request.timeout, child.wait()) => waited,
            // The user pressed stop: kill the whole process group at once instead of waiting for the script to finish.
            _ = wait_cancelled(&request.cancel) => {
                kill_tree(&mut child);
                let _ = child.wait().await;
                let _ = stdout_task.await;
                let _ = stderr_task.await;
                return Err(ExecutorError::Cancelled);
            }
        };
        let (exit_code, timed_out) = match wait {
            Ok(status) => (status.ok().and_then(|s| s.code()), false),
            Err(_) => {
                kill_tree(&mut child);
                let _ = child.wait().await;
                (None, true)
            }
        };
        let duration_ms = started.elapsed().as_millis() as u64;

        let (stdout, stdout_truncated) = stdout_task.await.unwrap_or_default();
        let (stderr, stderr_truncated) = stderr_task.await.unwrap_or_default();

        let control = if state_path.exists() {
            Some(
                read_control(&request.task_dir)
                    .map_err(|e| ExecutorError::Protocol(e.to_string()))?,
            )
        } else {
            None
        };

        Ok(ExecutionResult {
            stdout,
            stderr,
            exit_code,
            timed_out,
            duration_ms,
            stdout_truncated,
            stderr_truncated,
            control,
            script_path: Some(script_path),
            sandboxed: sandbox.is_some(),
        })
    }
}

fn make_executable(path: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(path)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(path, perms)?;
    }
    Ok(())
}

fn is_cancelled(flag: &Option<Arc<AtomicBool>>) -> bool {
    flag.as_ref()
        .is_some_and(|flag| flag.load(Ordering::SeqCst))
}

/// Returns when a cancel signal arrives; waits forever when there is no signal source (timeout is the only fallback).
async fn wait_cancelled(flag: &Option<Arc<AtomicBool>>) {
    let Some(flag) = flag else {
        std::future::pending::<()>().await;
        return;
    };
    while !flag.load(Ordering::SeqCst) {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

fn kill_tree(child: &mut tokio::process::Child) {
    #[cfg(unix)]
    {
        if let Some(pid) = child.id() {
            // Process group: a negative pid ends bash and its children (python, cli, ...) in one go.
            let _ = std::process::Command::new("/bin/kill")
                .args(["-9", &format!("-{pid}")])
                .status();
        }
    }
    let _ = child.start_kill();
}

async fn read_capped<R>(reader: Option<R>, cap: usize) -> (String, bool)
where
    R: tokio::io::AsyncRead + Unpin,
{
    let Some(reader) = reader else {
        return (String::new(), false);
    };
    let mut buffer = Vec::new();
    let mut limited = reader.take(cap as u64 + 1);
    let _ = limited.read_to_end(&mut buffer).await;
    let truncated = buffer.len() > cap;
    buffer.truncate(cap);
    (String::from_utf8_lossy(&buffer).to_string(), truncated)
}
