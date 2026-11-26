//! Delivery engine: system notification + opening the result in place (no focus stealing,
//! no main window popping up).

use std::path::{Path, PathBuf};

use relay_core::DeliveryMode;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DeliveryError {
    #[error("Failed to send notification: {0}")]
    Notify(String),
    #[error("Failed to open the result: {0}")]
    Open(String),
}

#[derive(Debug, Clone, Default)]
pub struct DeliveryOutcome {
    pub notified: bool,
    pub opened: Option<PathBuf>,
}

/// Pick the path to open: prefer the artifact file named by the user or model, otherwise the
/// first file that exists.
pub fn choose_target(
    mode: DeliveryMode,
    artifacts: &[PathBuf],
    preferred: Option<&Path>,
) -> Option<PathBuf> {
    if mode == DeliveryMode::NotifyOnly {
        return None;
    }
    if let Some(path) = preferred.filter(|p| p.exists()) {
        return Some(match mode {
            DeliveryMode::OpenFolder => path.parent().unwrap_or(path).to_path_buf(),
            _ => path.to_path_buf(),
        });
    }
    let file = artifacts.iter().find(|p| p.is_file())?;
    Some(match mode {
        DeliveryMode::OpenFolder => file.parent().unwrap_or(file).to_path_buf(),
        _ => file.clone(),
    })
}

pub fn deliver(
    title: &str,
    body: &str,
    mode: DeliveryMode,
    artifacts: &[PathBuf],
    preferred: Option<&Path>,
    auto_open: bool,
) -> DeliveryOutcome {
    let mut outcome = DeliveryOutcome::default();
    if let Err(err) = notify(title, body) {
        tracing::warn!(%err, "System notification failed");
    } else {
        outcome.notified = true;
    }

    if auto_open {
        if let Some(target) = choose_target(mode, artifacts, preferred) {
            match open::that_detached(&target) {
                Ok(_) => outcome.opened = Some(target),
                Err(err) => {
                    tracing::warn!(%err, path = %target.display(), "Failed to open the result")
                }
            }
        }
    }
    outcome
}

pub fn notify(title: &str, body: &str) -> Result<(), DeliveryError> {
    #[cfg(target_os = "macos")]
    {
        // Send the notification via osascript to avoid extra permission prompts and bundle dependencies.
        let script = format!(
            "display notification {} with title {}",
            applescript_string(body),
            applescript_string(title)
        );
        let output = std::process::Command::new("osascript")
            .args(["-e", &script])
            .output()
            .map_err(|e| DeliveryError::Notify(e.to_string()))?;
        if !output.status.success() {
            return Err(DeliveryError::Notify(
                String::from_utf8_lossy(&output.stderr).trim().to_string(),
            ));
        }
        Ok(())
    }
    #[cfg(not(target_os = "macos"))]
    {
        notify_rust::Notification::new()
            .summary(title)
            .body(body)
            .show()
            .map(|_| ())
            .map_err(|e| DeliveryError::Notify(e.to_string()))
    }
}

#[cfg(target_os = "macos")]
fn applescript_string(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notify_only_opens_nothing() {
        let artifacts = vec![PathBuf::from("/tmp/report.xlsx")];
        assert!(choose_target(DeliveryMode::NotifyOnly, &artifacts, None).is_none());
    }

    #[test]
    fn open_folder_uses_parent_directory() {
        let file = std::env::temp_dir().join("relay-delivery-test.txt");
        std::fs::write(&file, "x").unwrap();
        let target = choose_target(DeliveryMode::OpenFolder, &[file.clone()], Some(&file)).unwrap();
        assert_eq!(target, file.parent().unwrap());
    }

    #[test]
    fn prefers_existing_preferred_path() {
        let file = std::env::temp_dir().join("relay-delivery-preferred.txt");
        std::fs::write(&file, "x").unwrap();
        let target = choose_target(DeliveryMode::OpenFile, &[], Some(&file)).unwrap();
        assert_eq!(target, file);
    }
}
