//! Shell adapter layer used to run scripts.
//!
//! Task scripts run under bash on macOS / Linux and under PowerShell on Windows;
//! this module only decides "which interpreter + script extension + key environment variables" so the Executor no longer hardcodes `/bin/bash`.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use tokio::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShellSpec {
    /// Interpreter executable.
    pub program: &'static str,
    /// Arguments appended before the script path when running a script.
    pub args: &'static [&'static str],
    /// Script file extension (without the dot).
    pub script_ext: &'static str,
    /// Name of the user home directory environment variable.
    pub home_var: &'static str,
    /// Executable file extension (`.exe` on Windows).
    pub exe_ext: &'static str,
    /// Interpreter language name, written into prompts so the model knows which scripting language to use.
    pub language: &'static str,
}

#[cfg(not(windows))]
const SPEC: ShellSpec = ShellSpec {
    program: "/bin/bash",
    args: &["--noprofile", "--norc"],
    script_ext: "sh",
    home_var: "HOME",
    exe_ext: "",
    language: "bash",
};

#[cfg(windows)]
const SPEC: ShellSpec = ShellSpec {
    program: "powershell",
    args: &[
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
    ],
    script_ext: "ps1",
    home_var: "USERPROFILE",
    exe_ext: ".exe",
    language: "powershell",
};

pub fn spec() -> ShellSpec {
    SPEC
}

/// Build the command that runs `script_path` (cwd / environment are not set yet).
pub fn command(script_path: &Path) -> Command {
    let mut command = Command::new(SPEC.program);
    command.args(SPEC.args).arg(script_path);
    command
}

/// Task script file name, e.g. `step-01.sh` / `step-01.ps1`.
pub fn script_name(turn: u32) -> String {
    format!("step-{turn:02}.{}", SPEC.script_ext)
}

/// File name used when copying the small tools bundled with a task (Windows needs `.exe`).
pub fn helper_name(base: &str) -> String {
    format!("{base}{}", SPEC.exe_ext)
}

/// PATH visible inside a task: `.relay/bin` comes first, common directories are appended on Unix, and the system PATH is kept on Windows.
pub fn env_path(task_dir: &Path) -> OsString {
    let bin = task_dir.join(".relay").join("bin");
    let mut paths: Vec<PathBuf> = vec![bin];
    if cfg!(windows) {
        paths.extend(std::env::split_paths(
            &std::env::var_os("PATH").unwrap_or_default(),
        ));
    } else {
        paths.extend(
            [
                "/usr/local/bin",
                "/opt/homebrew/bin",
                "/usr/bin",
                "/bin",
                "/usr/sbin",
                "/sbin",
            ]
            .iter()
            .map(PathBuf::from),
        );
    }
    std::env::join_paths(paths).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn script_name_has_platform_extension() {
        let name = script_name(3);
        assert!(name.ends_with(SPEC.script_ext), "{name}");
        assert!(name.contains("03"));
    }

    #[test]
    fn command_starts_with_platform_interpreter() {
        let command = command(Path::new("/tmp/demo"));
        assert_eq!(
            command.as_std().get_program(),
            std::ffi::OsStr::new(SPEC.program)
        );
    }

    #[test]
    fn env_path_puts_task_bin_first() {
        let value = env_path(Path::new("/tmp/task"));
        let text = value.to_string_lossy();
        assert!(text.contains("/tmp/task/.relay/bin"), "{text}");
        #[cfg(windows)]
        assert_eq!(helper_name("relay-state"), "relay-state.exe");
        #[cfg(not(windows))]
        assert_eq!(helper_name("relay-state"), "relay-state");
    }
}
