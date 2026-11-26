//! Platform backend: the only module that touches `enigo`.
//!
//! The planner speaks in [`Event`]s; this module is the translation layer that
//! turns them into real synthetic input — macOS `CGEvent`, Windows `SendInput`,
//! Linux X11/XTest — through enigo. Keeping the two apart means the human-like
//! parameters can change without touching a platform API, and the whole
//! humanization model stays testable without a desktop.
//!
//! Two properties are deliberate:
//!
//! * **A backend is opened per call, not kept alive.** An enigo instance is
//!   bound to the thread that created it (the platform handles it wraps are not
//!   thread safe), so each call opens one on its own blocking-pool thread and
//!   drops it afterwards. Construction is cheap, and no half-pressed key state
//!   leaks from one call into the next.
//! * **A missing desktop is a result, not a crash.** No display server, no X11
//!   connection, or a denied macOS Accessibility permission all surface as a
//!   classified [`BackendError`], which the runtime reports as an envelope.
//!
//! `unsafe` is forbidden workspace-wide, which is why the FFI lives in enigo
//! rather than here — this module only ever touches safe APIs.

use enigo::{
    Axis, Coordinate, Direction, Enigo, InputError, Keyboard, Mouse, NewConError, Settings,
};

use crate::envelope::codes;
use crate::plan::Event;
use crate::request::Point;

/// Why the platform could not accept an event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackendError {
    /// The OS denied synthetic input; a human must grant it.
    PermissionDenied(String),
    /// There is no desktop to drive, or the platform bridge is missing.
    Unavailable(String),
    /// The platform cannot express this event at all.
    Unsupported(String),
    /// The platform refused one individual event mid-action.
    Rejected(String),
}

impl BackendError {
    /// Stable envelope code for this failure.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::PermissionDenied(_) => codes::PERMISSION_DENIED,
            Self::Unavailable(_) => codes::BACKEND_UNAVAILABLE,
            Self::Unsupported(_) => codes::UNSUPPORTED,
            Self::Rejected(_) => codes::INPUT_REJECTED,
        }
    }

    /// Human-readable detail for the envelope.
    #[must_use]
    pub fn message(&self) -> &str {
        match self {
            Self::PermissionDenied(message)
            | Self::Unavailable(message)
            | Self::Unsupported(message)
            | Self::Rejected(message) => message,
        }
    }
}

/// One opened input backend, usable from the thread that opened it.
pub trait InputBackend {
    /// Emits one planned event.
    ///
    /// # Errors
    ///
    /// Returns a classified [`BackendError`] when the platform refuses it.
    fn emit(&mut self, event: &Event) -> Result<(), BackendError>;

    /// Reads the current pointer position.
    ///
    /// # Errors
    ///
    /// Returns a classified [`BackendError`] when the platform cannot report it.
    fn position(&self) -> Result<Point, BackendError>;
}

/// Opens one [`InputBackend`] per call.
///
/// The factory is the seam the runtime is tested through: production uses
/// [`EnigoFactory`], tests inject a recorder that needs no desktop.
pub trait InputBackendFactory: Send + Sync + 'static {
    /// Opens a backend on the calling thread.
    ///
    /// # Errors
    ///
    /// Returns a classified [`BackendError`] when there is no desktop to drive
    /// or no permission to drive it.
    fn open(&self) -> Result<Box<dyn InputBackend>, BackendError>;
}

/// The real platform backend, built on enigo.
#[derive(Clone, Copy, Debug, Default)]
pub struct EnigoFactory;

impl InputBackendFactory for EnigoFactory {
    fn open(&self) -> Result<Box<dyn InputBackend>, BackendError> {
        let settings = Settings {
            // Never leave a simulated key held down after a call.
            release_keys_when_dropped: true,
            // Do not pop the macOS permission dialog from a background server
            // process: the envelope tells the model to ask a human instead,
            // exactly like `computer_accessibility` does.
            open_prompt_to_get_permissions: false,
            ..Settings::default()
        };
        let enigo = Enigo::new(&settings).map_err(classify)?;
        Ok(Box::new(EnigoBackend { enigo }))
    }
}

/// Maps enigo's connection errors onto envelope codes.
fn classify(error: NewConError) -> BackendError {
    match error {
        NewConError::NoPermission => BackendError::PermissionDenied(error.to_string()),
        NewConError::EstablishCon(detail) => {
            BackendError::Unavailable(format!("no input backend available: {detail}"))
        }
        other => BackendError::Unavailable(other.to_string()),
    }
}

/// Maps one per-event failure onto envelope codes.
fn classify_input(error: InputError) -> BackendError {
    match error {
        InputError::Mapping(_) | InputError::Unmapping(_) | InputError::NoEmptyKeycodes => {
            BackendError::Unsupported(error.to_string())
        }
        other => BackendError::Rejected(other.to_string()),
    }
}

fn direction(press: bool) -> Direction {
    if press {
        Direction::Press
    } else {
        Direction::Release
    }
}

/// An open enigo connection.
struct EnigoBackend {
    enigo: Enigo,
}

impl InputBackend for EnigoBackend {
    fn emit(&mut self, event: &Event) -> Result<(), BackendError> {
        match event {
            Event::Move { x, y } => self.enigo.move_mouse(*x, *y, Coordinate::Abs),
            Event::Button { button, press } => self.enigo.button(button.enigo(), direction(*press)),
            Event::Key { key, press } => self.enigo.key(*key, direction(*press)),
            Event::Text { text } => self.enigo.fast_text(text).map(|_| ()),
            Event::Scroll { steps } => self.enigo.scroll(*steps, Axis::Vertical),
            // Pauses never reach the backend; the runtime consumes them.
            Event::Pause => Ok(()),
        }
        .map_err(classify_input)
    }

    fn position(&self) -> Result<Point, BackendError> {
        self.enigo
            .location()
            .map(|(x, y)| Point { x, y })
            .map_err(|error| BackendError::Unavailable(error.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_denied_permission_is_not_an_unavailable_desktop() {
        let denied = classify(NewConError::NoPermission);
        assert_eq!(denied.code(), codes::PERMISSION_DENIED);
        assert!(denied.message().contains("permission"));

        let missing = classify(NewConError::EstablishCon("cannot open display"));
        assert_eq!(missing.code(), codes::BACKEND_UNAVAILABLE);
        assert!(missing.message().contains("cannot open display"));
    }

    #[test]
    fn missing_keyboard_entries_are_unsupported_not_rejected() {
        assert_eq!(
            classify_input(InputError::Mapping("no keysym".into())).code(),
            codes::UNSUPPORTED
        );
        assert_eq!(
            classify_input(InputError::Simulate("refused")).code(),
            codes::INPUT_REJECTED
        );
    }

    #[test]
    fn the_factory_only_fails_with_a_classified_error() {
        // On a headless machine this is an `Unavailable`; on a desktop it opens
        // a real connection. Either way it must never panic, and the factory
        // must be shareable across blocking threads.
        let factory: std::sync::Arc<dyn InputBackendFactory> = std::sync::Arc::new(EnigoFactory);
        match factory.open() {
            Ok(_) => {}
            Err(error) => assert!(!error.code().is_empty(), "unclassified: {error:?}"),
        }
    }
}
