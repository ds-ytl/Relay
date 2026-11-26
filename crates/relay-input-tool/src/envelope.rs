//! JSON envelope shared by every `computer_input` operation.
//!
//! One call returns exactly one envelope, on success and on semantic failure
//! alike:
//!
//! ```json
//! {"ok": true, "data": {"action": "click", "position": {"x": 1247, "y": 671}}, "meta": {"op": "click", "elapsed_ms": 384}}
//! {"ok": false, "error": {"code": "permission_denied", "message": "...", "retryable": false}}
//! ```
//!
//! Semantic failures stay *results* instead of tool errors so the model can
//! read `code` and `retryable` and pick its next move: `permission_denied`
//! means "stop and ask a human to grant Accessibility", `backend_unavailable`
//! means "there is no desktop to drive", `unsupported_key` means "spell the
//! key differently". [`ToolError`](relay_tool_api::ToolError) stays reserved for
//! a malformed call, which the request layer rejects before any input is
//! emitted.
//!
//! The reply deliberately never contains the planned intermediate pointer
//! positions: a few dozen `(411,303) (416,306) …` samples are noise in the
//! model's context. They go to the debug log instead, and only the landing
//! point, the event count, and the elapsed time are reported.

use serde_json::{Value, json};

/// Stable failure codes. The model is told to branch on these, so they are
/// part of the tool's contract and never change spelling.
pub mod codes {
    /// No desktop to drive: no display server, no input backend, or the
    /// platform bridge could not be opened.
    pub const BACKEND_UNAVAILABLE: &str = "backend_unavailable";
    /// The OS denied synthetic input; a human must grant it.
    pub const PERMISSION_DENIED: &str = "permission_denied";
    /// The key name is not one this runtime can express.
    pub const UNSUPPORTED_KEY: &str = "unsupported_key";
    /// The platform cannot express the requested event.
    pub const UNSUPPORTED: &str = "unsupported";
    /// The backend stopped accepting events mid-action.
    pub const INPUT_REJECTED: &str = "input_rejected";
    /// A bug or an unexpected platform failure.
    pub const INTERNAL_ERROR: &str = "internal_error";
}

/// Whether a failure of this class is worth retrying unchanged.
///
/// `retryable` only ever describes "call again as-is": `permission_denied`
/// never changes on its own, a missing display may appear later, and a rejected
/// individual event may succeed the next time.
#[must_use]
pub fn retryable_by_default(code: &str) -> bool {
    matches!(
        code,
        codes::BACKEND_UNAVAILABLE | codes::INPUT_REJECTED | codes::INTERNAL_ERROR
    )
}

/// Builds the success envelope.
#[must_use]
pub fn success(data: Value, meta: Value) -> Value {
    let mut envelope = json!({"ok": true});
    envelope["data"] = data;
    envelope["meta"] = meta;
    envelope
}

/// Builds the failure envelope for one stable code.
#[must_use]
pub fn failure(code: &str, message: impl Into<String>) -> Value {
    json!({
        "ok": false,
        "error": {
            "code": code,
            "message": message.into(),
            "retryable": retryable_by_default(code),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permission_denied_is_not_retryable_while_missing_desktops_are() {
        assert!(!retryable_by_default(codes::PERMISSION_DENIED));
        assert!(!retryable_by_default(codes::UNSUPPORTED_KEY));
        assert!(retryable_by_default(codes::BACKEND_UNAVAILABLE));
        assert!(retryable_by_default(codes::INPUT_REJECTED));
    }

    #[test]
    fn envelopes_carry_the_documented_shape() {
        let ok = success(json!({"action": "move"}), json!({"op": "move"}));
        assert_eq!(ok["ok"], json!(true));
        assert_eq!(ok["data"]["action"], json!("move"));

        let error = failure(codes::PERMISSION_DENIED, "grant Accessibility");
        assert_eq!(error["ok"], json!(false));
        assert_eq!(error["error"]["code"], json!("permission_denied"));
        assert_eq!(error["error"]["retryable"], json!(false));
    }
}
