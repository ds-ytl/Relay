use std::path::{Path, PathBuf};

use relay_core::{ControlPhase, ControlSignal};

pub fn state_file(task_dir: &Path) -> PathBuf {
    task_dir.join(".relay").join("state.json")
}

pub fn read_control(task_dir: &Path) -> Result<ControlSignal, String> {
    let path = state_file(task_dir);
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
    if raw.trim().is_empty() {
        return Ok(ControlSignal::default());
    }
    let mut signal: ControlSignal =
        serde_json::from_str(&raw).map_err(|e| format!("Failed to parse state file: {e}"))?;
    if signal.summary.trim().is_empty() {
        signal.summary = format!("Turn {} finished", 0);
    }
    Ok(signal)
}

/// Parse the state file written by the model or script; returns Continue when it is missing.
pub fn resolve_phase(signal: &Option<ControlSignal>) -> ControlPhase {
    signal
        .as_ref()
        .map(|s| s.phase)
        .unwrap_or(ControlPhase::Continue)
}

/// Scan the task directory for artifacts created or modified in this turn (skipping .relay).
pub fn scan_artifacts(task_dir: &Path, since: std::time::SystemTime) -> Vec<PathBuf> {
    let mut out = Vec::new();
    walk(task_dir, since, 0, &mut out);
    out.sort();
    out
}

fn walk(dir: &Path, since: std::time::SystemTime, depth: usize, out: &mut Vec<PathBuf>) {
    if depth > 4 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if matches!(
            name.as_ref(),
            ".relay"
                | ".git"
                | "target"
                | "node_modules"
                | ".venv"
                | "__pycache__"
                | ".next"
                | "dist"
                | "build"
        ) {
            continue;
        }
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        if meta.is_dir() {
            walk(&path, since, depth + 1, out);
        } else if meta.is_file() {
            if let Ok(modified) = meta.modified() {
                if modified >= since {
                    out.push(path);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_state_file() {
        let dir = std::env::temp_dir().join(format!("relay-state-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(dir.join(".relay")).unwrap();
        std::fs::write(
            state_file(&dir),
            r#"{"phase":"done","summary":"Generation finished","artifacts":["out.txt"]}"#,
        )
        .unwrap();
        let signal = read_control(&dir).unwrap();
        assert_eq!(signal.phase, ControlPhase::Done);
        assert_eq!(signal.artifacts, vec!["out.txt"]);
    }
}
