//! Global hotkey: parsing and registration live in `relay-platform` (global-hotkey);
//! this only forwards press events into the GPUI async channel.

use std::sync::OnceLock;

use tokio::sync::mpsc::UnboundedSender;

/// Registration must hold it: the hotkey dies as soon as the manager is dropped.
static HANDLE: OnceLock<relay_platform::HotkeyHandle> = OnceLock::new();

pub fn register(spec: &str, tx: UnboundedSender<()>) -> Result<(), String> {
    let handle = relay_platform::register_hotkey(spec)?;
    HANDLE
        .set(handle)
        .map_err(|_| "hotkey already registered".to_string())?;

    std::thread::Builder::new()
        .name("relay-hotkey-events".into())
        .spawn(move || {
            let receiver = relay_platform::global_hotkey_events();
            // The global channel sees every hotkey event; only one is registered today.
            while let Ok(_event) = receiver.recv() {
                if tx.send(()).is_err() {
                    break;
                }
            }
        })
        .map_err(|err| err.to_string())?;
    Ok(())
}
