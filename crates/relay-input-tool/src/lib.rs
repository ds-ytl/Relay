//! `computer_input` — one tool, nine strictly typed desktop operations.
//!
//! The model sees a single function tool whose `op` field selects one of
//! `move`, `click`, `double_click`, `drag`, `scroll`, `type`, `key`, `hotkey`,
//! or `sequence`. Arguments are never forwarded as an untyped bag: they are
//! parsed into the discriminated request in [`request`] and validated before
//! anything runs, so unknown operations, unknown fields, wrong types, missing
//! fields, and fields that belong to another operation are all rejected with
//! [`ToolError::InvalidArguments`].
//!
//! The split between intent and detail is the whole point of the design:
//!
//! * **The model decides what to do** — click that control, type that text,
//!   drag from here to there, optionally with a one-word [`Behavior`] profile.
//! * **The runtime decides how a hand would do it** — [`plan`] generates an
//!   arced, accelerated pointer path with a settling tremor, a safe landing
//!   pixel inside a control's bounds, a real press-and-hold, a double click
//!   inside the OS threshold, a wheel that ticks instead of jumping, and a
//!   typing rhythm with word and sentence pauses. [`backend`] only emits the
//!   resulting events on the platform.
//!
//! Semantic failures are results, exactly like `computer_accessibility`:
//!
//! ```json
//! {"ok": true, "data": {"action": "click", "position": {"x": 1247, "y": 671}}, "meta": {"op": "click", "elapsed_ms": 384}}
//! {"ok": false, "error": {"code": "permission_denied", "message": "...", "retryable": false}}
//! ```
//!
//! The server process needs permission to synthesize input (macOS: *System
//! Settings → Privacy & Security → Accessibility*; Linux: a reachable X11
//! display). Without it every call answers `permission_denied` — no retry can
//! fix that, a human has to grant it.
//!
//! One safety property is worth stating: this tool does not try to defeat bot
//! detection. Human-like timing exists to make *interactions* land — the right
//! pixel, the right key order, the UI given time to react — not to disguise
//! automation.

use std::sync::Arc;

use relay_tool_api::{Tool, ToolDefinition, ToolError, ToolFuture};
use serde_json::Value;
use tracing::debug;

pub mod backend;
pub mod envelope;
pub mod native;
pub mod plan;
pub mod request;

pub use backend::{BackendError, EnigoFactory, InputBackend, InputBackendFactory};
pub use native::NativeInputRuntime;
pub use plan::{Child, Event, Plan, Step};
pub use request::{
    Action, ActionKind, Behavior, InputRequest, MAX_ACTION_MS, MAX_CLICKS, MAX_COORDINATE,
    MAX_HOTKEY_KEYS, MAX_SCROLL_NOTCHES, MAX_SEQUENCE_ACTIONS, MAX_TEXT_CHARS, MouseButton, Op,
    Point, ScrollDirection, Target, is_modifier, key_label, parse_key, tool_schema,
};

/// Name the model sees for this tool.
pub const TOOL_NAME: &str = "computer_input";

/// Backend able to execute one validated action.
///
/// Implemented by [`NativeInputRuntime`] for production and by fakes in tests,
/// so the tool's parsing, validation, and envelope handling stay testable
/// without a desktop.
pub trait InputRuntime: Send + Sync + 'static {
    /// Executes one action and returns the runtime's JSON envelope.
    fn execute(&self, action: Action) -> ToolFuture<'_>;
}

/// The single input tool exposed to the model.
#[derive(Clone)]
pub struct ComputerInputTool {
    definition: ToolDefinition,
    runtime: Arc<dyn InputRuntime>,
}

impl ComputerInputTool {
    /// Creates the tool over one runtime backend.
    ///
    /// # Panics
    ///
    /// Panics only if the embedded JSON Schema is not an object, which cannot
    /// happen because it is a literal.
    #[must_use]
    pub fn new(runtime: Arc<dyn InputRuntime>) -> Self {
        let definition = ToolDefinition::new(
            TOOL_NAME,
            Some(
                "Control mouse and keyboard by op, preferably using accessibility bounds; requires OS input permission and returns ok/data or error/retryable.",
            ),
            tool_schema(),
        )
        .expect("input tool schema is valid");
        Self {
            definition: definition.with_strict(false),
            runtime,
        }
    }

    /// Creates the tool over the in-process platform backend.
    ///
    /// The backend itself is created per call, so constructing the tool never
    /// touches the platform (and never fails), even on a machine without a
    /// desktop.
    #[must_use]
    pub fn native() -> Self {
        Self::new(Arc::new(NativeInputRuntime::new()))
    }

    /// Creates the tool over an already-built runtime.
    #[must_use]
    pub fn with_runtime(runtime: NativeInputRuntime) -> Self {
        Self::new(Arc::new(runtime))
    }
}

impl Default for ComputerInputTool {
    fn default() -> Self {
        Self::native()
    }
}

impl Tool for ComputerInputTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    fn invoke(&self, arguments: Value) -> ToolFuture<'_> {
        Box::pin(async move {
            let action = parse_action(arguments)?;
            debug!(op = action.label(), "input action dispatched");
            self.runtime.execute(action).await
        })
    }
}

/// Parses and validates one model call into the typed [`Action`] the runtime
/// executes.
fn parse_action(arguments: Value) -> Result<Action, ToolError> {
    let request: InputRequest = serde_json::from_value(arguments).map_err(|error| {
        debug!(error = %error, "rejected malformed input request");
        ToolError::invalid_arguments(
            TOOL_NAME,
            format!(
                "{error}; expected one of op=move|click|double_click|drag|scroll|type|key|hotkey|sequence \
                 with that operation's fields"
            ),
        )
    })?;
    request
        .validate()
        .map_err(|message| ToolError::invalid_arguments(TOOL_NAME, message))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Mutex;

    /// Runtime that records the validated actions and answers with a canned
    /// envelope.
    #[derive(Default)]
    struct RecordingRuntime {
        seen: Mutex<Vec<Action>>,
    }

    impl InputRuntime for RecordingRuntime {
        fn execute(&self, action: Action) -> ToolFuture<'_> {
            Box::pin(async move {
                self.seen.lock().unwrap().push(action);
                Ok(envelope::success(json!({"action": "ok"}), json!({})))
            })
        }
    }

    #[tokio::test]
    async fn one_tool_carries_every_operation() {
        let runtime = Arc::new(RecordingRuntime::default());
        let tool = ComputerInputTool::new(runtime.clone());
        assert_eq!(tool.definition().name, TOOL_NAME);
        assert!(!tool.definition().strict);

        for op in Op::all() {
            let arguments = match op {
                Op::Move => json!({"op": "move", "x": 10, "y": 20}),
                Op::Click => json!({"op": "click", "x": 10, "y": 20}),
                Op::DoubleClick => json!({"op": "double_click", "x": 10, "y": 20}),
                Op::Drag => json!({"op": "drag", "from": {"x": 1, "y": 2}, "to": {"x": 3, "y": 4}}),
                Op::Scroll => json!({"op": "scroll", "direction": "down"}),
                Op::Type => json!({"op": "type", "text": "hi"}),
                Op::Key => json!({"op": "key", "key": "enter"}),
                Op::Hotkey => json!({"op": "hotkey", "keys": ["META", "L"]}),
                Op::Sequence => {
                    json!({"op": "sequence", "actions": [{"op": "key", "key": "escape"}]})
                }
            };
            let reply = tool.invoke(arguments).await.expect("operation is valid");
            assert_eq!(reply["ok"], json!(true));
        }
        assert_eq!(runtime.seen.lock().unwrap().len(), Op::all().len());
    }

    #[tokio::test]
    async fn a_native_tool_is_constructible_without_touching_the_desktop() {
        let tool = ComputerInputTool::native();
        assert_eq!(tool.definition().name, TOOL_NAME);
        assert_eq!(ComputerInputTool::default().definition().name, TOOL_NAME);
    }

    #[tokio::test]
    async fn malformed_calls_never_reach_the_runtime() {
        let runtime = Arc::new(RecordingRuntime::default());
        let tool = ComputerInputTool::new(runtime.clone());

        for arguments in [
            json!({"op": "swipe", "x": 1, "y": 2}),
            json!({"op": "click"}),
            json!({"op": "move", "x": 1, "y": 2, "text": "hello"}),
            json!({"op": "click", "x": 1, "y": 2, "clicks": 9}),
        ] {
            let error = tool.invoke(arguments).await.expect_err("invalid call");
            match error {
                ToolError::InvalidArguments { name, .. } => assert_eq!(name, TOOL_NAME),
                other => panic!("expected invalid arguments, got {other}"),
            }
        }
        assert!(runtime.seen.lock().unwrap().is_empty());
    }
}
