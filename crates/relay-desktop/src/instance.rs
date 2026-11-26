//! Single-instance guard: the desktop shell refuses to start a second copy.
//!
//! Relay is tray-resident, so a second launch would add a duplicate menu-bar icon, fight the first
//! process for the global hotkey and put a second writer on the same SQLite file. The first launch
//! therefore owns a socket in the data directory, and every later launch hands over to it.
//!
//! - The socket is the lock. Binding it is atomic, and the kernel releases it when the process dies,
//!   so a killed shell never leaves the slot occupied.
//! - A later launch connects instead of binding. Connecting proves the owner is alive and is also
//!   how the owner learns to surface its panel: relaunching `relay` acts as "open the window".
//! - A socket file left behind by a crash refuses the connection, so it is removed and rebound.

use std::path::{Path, PathBuf};

use anyhow::Result;
use tokio::sync::mpsc::UnboundedSender;

use crate::config::AppConfig;

/// What this process may do after asking for the instance slot.
pub enum Launch {
    /// Nobody else is running: hold the guard for as long as the shell runs.
    Primary(InstanceGuard),
    /// Another instance is alive and has been asked to show itself; this process must exit.
    AlreadyRunning,
}

/// Owns the instance slot; dropping it releases the slot for the next start.
pub struct InstanceGuard {
    /// `None` when the guard could not be installed and the shell started unprotected.
    path: Option<PathBuf>,
}

impl Drop for InstanceGuard {
    fn drop(&mut self) {
        if let Some(path) = &self.path {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// Take the single-instance slot, waking the panel through `wake` when a later launch arrives.
pub fn claim(wake: UnboundedSender<()>) -> Result<Launch> {
    let path = AppConfig::home().join("relay.sock");
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    match acquire(&path, wake) {
        Ok(Some(guard)) => Ok(Launch::Primary(guard)),
        Ok(None) => Ok(Launch::AlreadyRunning),
        // A guard that cannot be installed is not worth refusing to start over: fall back to the
        // previous behaviour (no protection) and say so, instead of leaving the user with no app.
        Err(err) => {
            tracing::warn!(
                %err,
                path = %path.display(),
                "single-instance guard unavailable; starting anyway"
            );
            Ok(Launch::Primary(InstanceGuard { path: None }))
        }
    }
}

#[cfg(unix)]
fn acquire(path: &Path, wake: UnboundedSender<()>) -> std::io::Result<Option<InstanceGuard>> {
    use std::io::Write;
    use std::os::unix::net::{UnixListener, UnixStream};

    let listener = match UnixListener::bind(path) {
        Ok(listener) => listener,
        Err(err) if err.kind() == std::io::ErrorKind::AddrInUse => {
            // A crash leaves the socket file behind and nothing is listening on it, so a refused
            // connection means "stale, take it over"; an accepted one means "another shell is alive".
            if UnixStream::connect(path).is_ok() {
                return Ok(None);
            }
            std::fs::remove_file(path)?;
            UnixListener::bind(path)?
        }
        Err(err) => return Err(err),
    };

    std::thread::Builder::new()
        .name("relay-instance".into())
        .spawn(move || {
            // The payload carries no meaning: any connection is a launch asking for the panel.
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let _ = stream.write_all(b"summon");
                if wake.send(()).is_err() {
                    break;
                }
            }
        })?;

    Ok(Some(InstanceGuard {
        path: Some(path.to_path_buf()),
    }))
}

#[cfg(not(unix))]
fn acquire(path: &Path, _wake: UnboundedSender<()>) -> std::io::Result<Option<InstanceGuard>> {
    // No unix sockets in std: hold the slot with a pid file plus a liveness check, enough to refuse
    // a second shell (which then simply exits; there is no channel to wake the owner with).
    if let Ok(raw) = std::fs::read_to_string(path) {
        let owner = raw.trim().parse::<u32>().ok();
        if owner.is_some_and(process_alive) {
            return Ok(None);
        }
        // Stale pid from a crash: take the slot over.
    }
    std::fs::write(path, std::process::id().to_string())?;
    Ok(Some(InstanceGuard {
        path: Some(path.to_path_buf()),
    }))
}

#[cfg(windows)]
fn process_alive(pid: u32) -> bool {
    // A normal-privilege process cannot query an elevated one, which reads as "gone": the next
    // launch then takes the slot over instead of refusing to start.
    std::process::Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}")])
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).contains(&pid.to_string()))
        .unwrap_or(false)
}

#[cfg(not(any(unix, windows)))]
fn process_alive(_pid: u32) -> bool {
    false
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn second_launch_is_refused_and_asks_the_panel_to_open() {
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("relay.sock");
        let (wake, mut raised) = tokio::sync::mpsc::unbounded_channel();
        let guard = acquire(&socket, wake)
            .unwrap()
            .expect("the first launch owns the slot");

        let (wake, _raised_again) = tokio::sync::mpsc::unbounded_channel();
        assert!(
            acquire(&socket, wake).unwrap().is_none(),
            "a second launch must not own the slot"
        );

        // The second launch reaches the owner through the socket; that wake-up raises its panel.
        let mut awoken = false;
        for _ in 0..100 {
            if raised.try_recv().is_ok() {
                awoken = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(
            awoken,
            "the running instance must be woken by a second launch"
        );

        drop(guard);
        assert!(
            !socket.exists(),
            "dropping the guard must release the socket for the next start"
        );
    }

    #[test]
    fn a_socket_left_by_a_crash_is_taken_over() {
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("relay.sock");
        std::fs::write(&socket, b"stale").unwrap();
        let (wake, _raised) = tokio::sync::mpsc::unbounded_channel();
        assert!(
            acquire(&socket, wake).unwrap().is_some(),
            "a stale socket file must not block the next launch"
        );
    }
}
