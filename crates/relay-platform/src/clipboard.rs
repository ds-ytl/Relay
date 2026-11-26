//! Clipboard read/write (arboard, supports macOS / Windows / Linux).

use arboard::Clipboard;

/// Read plain text; returns None when empty or access is denied.
pub fn read_text() -> Option<String> {
    let mut clipboard = Clipboard::new().ok()?;
    clipboard
        .get_text()
        .ok()
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
}

pub fn write_text(text: &str) -> Result<(), String> {
    let mut clipboard = Clipboard::new().map_err(|err| err.to_string())?;
    clipboard
        .set_text(text.to_string())
        .map_err(|err| err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_plain_text() {
        let marker = format!("relay-clipboard-{}", std::process::id());
        if write_text(&marker).is_err() {
            // Skip when there is no graphical session (CI/SSH).
            return;
        }
        assert_eq!(read_text().as_deref(), Some(marker.as_str()));
    }
}
