//! Global hotkeys (global-hotkey, supports macOS / Windows / Linux).

use std::str::FromStr;

use global_hotkey::hotkey::{Code, HotKey, Modifiers};
use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager};
use thiserror::Error;

pub use global_hotkey::GlobalHotKeyEventReceiver;

#[derive(Debug, Error)]
pub enum HotkeyParse {
    #[error("Shortcut is empty")]
    Empty,
    #[error("Unsupported key: {0}")]
    UnknownKey(String),
    #[error("A shortcut needs at least one modifier (ctrl / alt / shift / cmd)")]
    MissingModifier,
}

/// Must be held after a successful registration: the hotkey stops working once the manager is dropped.
pub struct HotkeyHandle {
    _manager: GlobalHotKeyManager,
}

pub fn register(accelerator: &str) -> Result<HotkeyHandle, String> {
    let hotkey = parse(accelerator).map_err(|err| err.to_string())?;
    let manager = GlobalHotKeyManager::new().map_err(|err| err.to_string())?;
    manager.register(hotkey).map_err(|err| err.to_string())?;
    Ok(HotkeyHandle { _manager: manager })
}

/// Global event receiver; after the callback exits, new hotkey events are read from here.
pub fn global_hotkey_events() -> &'static GlobalHotKeyEventReceiver {
    GlobalHotKeyEvent::receiver()
}

pub fn parse(accelerator: &str) -> Result<HotKey, HotkeyParse> {
    let mut modifiers = Modifiers::empty();
    let mut code: Option<Code> = None;
    for part in accelerator
        .split('+')
        .map(|part| part.trim().to_lowercase())
    {
        if part.is_empty() {
            continue;
        }
        match part.as_str() {
            "ctrl" | "control" => modifiers |= Modifiers::CONTROL,
            "alt" | "option" | "opt" => modifiers |= Modifiers::ALT,
            "shift" => modifiers |= Modifiers::SHIFT,
            "cmd" | "command" | "meta" | "super" | "win" => modifiers |= Modifiers::SUPER,
            other => {
                if code.is_some() {
                    return Err(HotkeyParse::UnknownKey(other.to_string()));
                }
                code = Some(parse_code(other)?);
            }
        }
    }
    if modifiers.is_empty() {
        return Err(HotkeyParse::MissingModifier);
    }
    let code = code.ok_or_else(|| HotkeyParse::UnknownKey(String::new()))?;
    Ok(HotKey::new(Some(modifiers), code))
}

fn parse_code(token: &str) -> Result<Code, HotkeyParse> {
    let code = match token {
        "space" => Code::Space,
        "enter" | "return" => Code::Enter,
        "escape" | "esc" => Code::Escape,
        "tab" => Code::Tab,
        "backspace" => Code::Backspace,
        "delete" => Code::Delete,
        "up" => Code::ArrowUp,
        "down" => Code::ArrowDown,
        "left" => Code::ArrowLeft,
        "right" => Code::ArrowRight,
        "home" => Code::Home,
        "end" => Code::End,
        "pageup" => Code::PageUp,
        "pagedown" => Code::PageDown,
        _ => parse_letter_or_digit(token).ok_or_else(|| HotkeyParse::UnknownKey(token.into()))?,
    };
    Ok(code)
}

fn parse_letter_or_digit(token: &str) -> Option<Code> {
    if token.len() == 1 {
        let ch = token.chars().next()?;
        if ch.is_ascii_alphabetic() {
            return Code::from_str(&format!("Key{}", ch.to_ascii_uppercase())).ok();
        }
        if ch.is_ascii_digit() {
            return Code::from_str(&format!("Digit{ch}")).ok();
        }
    }
    if let Some(rest) = token.strip_prefix('f') {
        if let Ok(index) = rest.parse::<u8>() {
            if (1..=20).contains(&index) {
                return Code::from_str(&format!("F{index}")).ok();
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_default_accelerator() {
        let hotkey = parse("ctrl+alt+space").unwrap();
        assert_eq!(hotkey.key, Code::Space);
        assert!(hotkey.mods.contains(Modifiers::CONTROL));
        assert!(hotkey.mods.contains(Modifiers::ALT));
    }

    #[test]
    fn rejects_bare_key() {
        assert!(matches!(parse("space"), Err(HotkeyParse::MissingModifier)));
    }

    #[test]
    fn rejects_unknown_key() {
        assert!(matches!(
            parse("ctrl+alt+whatever"),
            Err(HotkeyParse::UnknownKey(_))
        ));
    }
}
