//! JSON envelope shared by every `computer_accessibility` operation.
//!
//! One call returns exactly one envelope, on success and on semantic failure
//! alike:
//!
//! ```json
//! {"ok": true, "data": {...}, "meta": {...}}
//! {"ok": false, "error": {"code": "element_not_found", "message": "...", "retryable": true}}
//! ```
//!
//! Semantic failures stay *results* instead of tool errors so the model can
//! read `code` and `retryable` and pick its next move: `stale_element` means
//! "query again", `permission_denied` means "stop and ask a human", `timeout`
//! means "observe again". [`ToolError`](relay_tool_api::ToolError) stays reserved
//! for a malformed call, which the request layer rejects before any backend
//! work happens.

use serde_json::{Value, json};
use xa11y::Error as Xa11yError;

/// Stable failure codes. The model is told to branch on these, so they are
/// part of the tool's contract and never change spelling.
pub mod codes {
    /// No application matched, or nothing is focused.
    pub const APP_NOT_FOUND: &str = "app_not_found";
    /// No element matched the selector, or the coordinate hit nothing.
    pub const ELEMENT_NOT_FOUND: &str = "element_not_found";
    /// A selector resolved to more than one element where one was required.
    pub const MULTIPLE_MATCHES: &str = "multiple_matches";
    /// The element handle no longer resolves (UI changed, or handles were
    /// trimmed); query again.
    pub const STALE_ELEMENT: &str = "stale_element";
    /// The OS denied accessibility access; a human must grant it.
    pub const PERMISSION_DENIED: &str = "permission_denied";
    /// The platform cannot express this operation at all.
    pub const UNSUPPORTED: &str = "unsupported";
    /// The element does not support the requested action.
    pub const UNSUPPORTED_ACTION: &str = "unsupported_action";
    /// The observation budget elapsed.
    pub const TIMEOUT: &str = "timeout";
    /// The element exists but is not visible.
    pub const NOT_VISIBLE: &str = "not_visible";
    /// The element exists but is disabled.
    pub const NOT_ENABLED: &str = "not_enabled";
    /// The accessibility backend is missing or broken (no AT-SPI bus, no
    /// accessibility bridge for the application, ...).
    pub const BACKEND_UNAVAILABLE: &str = "backend_unavailable";
    /// A bug or an unexpected platform failure.
    pub const INTERNAL_ERROR: &str = "internal_error";
    /// A screenshot could not be written to the requested path: the directory
    /// does not exist and cannot be created, the file already exists (a
    /// capture never overwrites), or the disk refused the write.
    pub const WRITE_FAILED: &str = "write_failed";
}

/// Whether a failure of this class is worth retrying unchanged.
///
/// `retryable` only ever describes "call again as-is": `permission_denied`
/// never changes on its own, `element_not_found` and `timeout` do.
#[must_use]
pub fn retryable_by_default(code: &str) -> bool {
    matches!(
        code,
        codes::APP_NOT_FOUND
            | codes::ELEMENT_NOT_FOUND
            | codes::STALE_ELEMENT
            | codes::TIMEOUT
            | codes::BACKEND_UNAVAILABLE
    )
}

/// One semantic failure, ready to be rendered as an error envelope.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Failure {
    /// Stable machine-readable code; see [`codes`].
    pub code: &'static str,
    /// Human-readable explanation, aimed at the model.
    pub message: String,
    /// Whether an identical retry could plausibly succeed.
    pub retryable: bool,
}

impl Failure {
    /// Creates a failure, defaulting `retryable` from the code.
    #[must_use]
    pub fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            retryable: retryable_by_default(code),
        }
    }

    /// Renders the error envelope the model receives.
    #[must_use]
    pub fn to_value(&self) -> Value {
        json!({
            "ok": false,
            "error": {
                "code": self.code,
                "message": self.message,
                "retryable": self.retryable,
            },
        })
    }

    /// Translates one `xa11y` failure into the tool's stable code set.
    ///
    /// The message is kept verbatim so platform detail (selector, diagnosis,
    /// permission instructions) survives; only the code is normalized.
    #[must_use]
    pub fn from_xa11y(error: &Xa11yError) -> Self {
        let message = error.to_string();
        let (code, retryable) = match error {
            Xa11yError::PermissionDenied { .. } => (codes::PERMISSION_DENIED, false),
            Xa11yError::AccessibilityNotEnabled { .. } => (codes::BACKEND_UNAVAILABLE, false),
            Xa11yError::SelectorNotMatched { .. } => (codes::ELEMENT_NOT_FOUND, true),
            Xa11yError::ElementStale { .. } => (codes::STALE_ELEMENT, true),
            Xa11yError::ActionNotSupported { .. } | Xa11yError::TextValueNotSupported => {
                (codes::UNSUPPORTED_ACTION, false)
            }
            Xa11yError::Timeout { .. } => (codes::TIMEOUT, true),
            Xa11yError::InvalidActionData { .. } => (codes::UNSUPPORTED_ACTION, false),
            // A selector we built ourselves failing to parse is our bug, not
            // the model's; an unknown variant joins the same fallback.
            Xa11yError::InvalidSelector { .. } | Xa11yError::InvalidConfig { .. } => {
                (codes::INTERNAL_ERROR, false)
            }
            Xa11yError::NoElementBounds | Xa11yError::Unsupported { .. } => {
                (codes::UNSUPPORTED, false)
            }
            // Platform errors are the backend's own call failures (an AX/HWND
            // question that got no answer, a dead D-Bus peer): the same call
            // can succeed on the next attempt.
            Xa11yError::Platform { .. } => (codes::BACKEND_UNAVAILABLE, true),
            _ => (codes::INTERNAL_ERROR, false),
        };
        Self {
            code,
            message,
            retryable,
        }
    }
}

/// Renders the success envelope.
#[must_use]
pub fn success(data: &Value, meta: &Value) -> Value {
    json!({ "ok": true, "data": data, "meta": meta })
}

/// Renders a failure envelope for a backend that could not even start.
#[must_use]
pub fn internal_error(message: impl Into<String>) -> Value {
    Failure::new(codes::INTERNAL_ERROR, message).to_value()
}

#[cfg(test)]
mod tests {
    use super::*;
    use xa11y::{Diagnosis, Role};

    #[test]
    fn success_envelope_is_machine_readable() {
        let value = success(&json!({"elements": []}), &json!({"op": "query"}));
        assert_eq!(value["ok"], json!(true));
        assert_eq!(value["data"]["elements"], json!([]));
        assert_eq!(value["meta"]["op"], json!("query"));
    }

    #[test]
    fn failure_envelope_carries_code_message_and_retry_advice() {
        let value = Failure::new(codes::ELEMENT_NOT_FOUND, "no match").to_value();
        assert_eq!(value["ok"], json!(false));
        assert_eq!(value["error"]["code"], json!("element_not_found"));
        assert_eq!(value["error"]["message"], json!("no match"));
        assert_eq!(value["error"]["retryable"], json!(true));
    }

    #[test]
    fn permission_denied_is_never_retryable() {
        let failure = Failure::from_xa11y(&Xa11yError::PermissionDenied {
            instructions: "grant Accessibility in System Settings".into(),
        });
        assert_eq!(failure.code, codes::PERMISSION_DENIED);
        assert!(!failure.retryable);
        assert!(failure.message.contains("grant Accessibility"));
    }

    #[test]
    fn transient_backend_failures_stay_retryable() {
        let stale = Failure::from_xa11y(&Xa11yError::ElementStale {
            selector: "el_71".into(),
        });
        assert_eq!(stale.code, codes::STALE_ELEMENT);
        assert!(stale.retryable);

        let timeout = Failure::from_xa11y(&Xa11yError::Timeout {
            elapsed: std::time::Duration::from_secs(2),
            diagnosis: None,
        });
        assert_eq!(timeout.code, codes::TIMEOUT);
        assert!(timeout.retryable);

        let platform = Failure::from_xa11y(&Xa11yError::Platform {
            code: -25204,
            message: "AXUIElementCopyAttributeValue failed".into(),
        });
        assert_eq!(platform.code, codes::BACKEND_UNAVAILABLE);
        assert!(platform.retryable);
    }

    #[test]
    fn unsupported_actions_are_terminal() {
        let unsupported = Failure::from_xa11y(&Xa11yError::ActionNotSupported {
            action: "press".into(),
            role: Role::Group,
        });
        assert_eq!(unsupported.code, codes::UNSUPPORTED_ACTION);
        assert!(!unsupported.retryable);

        let no_bounds = Failure::from_xa11y(&Xa11yError::NoElementBounds);
        assert_eq!(no_bounds.code, codes::UNSUPPORTED);
        assert!(!no_bounds.retryable);
    }

    #[test]
    fn matched_selector_failures_keep_their_diagnosis() {
        let missing = Failure::from_xa11y(&Xa11yError::SelectorNotMatched {
            selector: r#"button[name="Continue"]"#.into(),
            diagnosis: Some(Box::new(
                Diagnosis::new()
                    .condition("Continue button")
                    .candidates(["button \"Cancel\"".to_string()]),
            )),
        });
        assert_eq!(missing.code, codes::ELEMENT_NOT_FOUND);
        assert!(missing.retryable);
        assert!(missing.message.contains("Continue"), "{}", missing.message);
    }
}
