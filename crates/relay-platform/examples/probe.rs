//! Manually verify cross-platform capabilities: `cargo run -p relay-platform --example probe`

fn main() {
    match relay_platform::front_window() {
        Some(window) => println!(
            "Front window: app={:?} title={:?}",
            window.app, window.title
        ),
        None => println!(
            "Front window: <unavailable> (accessibility/screen recording permission may be missing)"
        ),
    }

    match relay_platform::clipboard_text() {
        Some(text) => println!("Clipboard: {} chars", text.chars().count()),
        None => println!("Clipboard: <empty or unreadable>"),
    }

    let dir = std::env::temp_dir().join("relay-probe");
    match relay_platform::capture_primary_png(&dir) {
        Ok(path) => {
            let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            println!("Screenshot: {} ({} bytes)", path.display(), size);
        }
        Err(err) => println!("Screenshot failed: {err}"),
    }
}
