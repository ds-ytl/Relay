//! Tests for the public `computer_input` tool surface.
//!
//! The tool is exercised through a recording runtime, so request parsing,
//! validation, schema exposure, and envelope passthrough are covered without a
//! desktop. The humanization itself is covered by the unit tests next to the
//! planner, and the replay loop by the tests next to the runtime.

use std::sync::{Arc, Mutex};

use relay_input_tool::{Action, ComputerInputTool, InputRuntime, Op, TOOL_NAME, tool_schema};
use relay_tool_api::{Tool, ToolError, ToolFuture};
use serde_json::{Value, json};

/// Runtime that records validated actions and answers with a canned envelope.
#[derive(Default)]
struct RecordingRuntime {
    seen: Mutex<Vec<Action>>,
    reply: Mutex<Value>,
}

impl RecordingRuntime {
    fn answering(reply: Value) -> Arc<Self> {
        let runtime = Self::default();
        *runtime.reply.lock().unwrap() = reply;
        Arc::new(runtime)
    }

    fn recorded(&self) -> Vec<Action> {
        self.seen.lock().unwrap().clone()
    }
}

impl InputRuntime for RecordingRuntime {
    fn execute(&self, action: Action) -> ToolFuture<'_> {
        Box::pin(async move {
            self.seen.lock().unwrap().push(action);
            Ok(self.reply.lock().unwrap().clone())
        })
    }
}

fn arguments_error(error: ToolError) -> String {
    match error {
        ToolError::InvalidArguments { name, message } => {
            assert_eq!(name, TOOL_NAME);
            message
        }
        other => panic!("expected invalid arguments, got {other}"),
    }
}

#[test]
fn exposes_one_tool_with_nine_operations() {
    let tool = ComputerInputTool::new(RecordingRuntime::answering(json!({})));
    let definition = tool.definition();
    assert_eq!(definition.name, TOOL_NAME);
    assert!(!definition.strict);
    let schema = &definition.parameters;
    assert_eq!(schema["required"], json!(["op"]));
    assert_eq!(schema["additionalProperties"], json!(false));
    let operations = schema["properties"]["op"]["enum"].as_array().unwrap();
    let expected: Vec<&str> = Op::all().iter().map(|op| op.label()).collect();
    assert_eq!(operations.len(), expected.len());
    for op in expected {
        assert!(
            operations.contains(&json!(op)),
            "{op} missing from the schema"
        );
    }
    assert_eq!(
        schema["properties"]["behavior"]["enum"],
        json!(["fast", "natural", "precise"])
    );
    assert_eq!(definition.as_responses_tool()["type"], json!("function"));
    // The schema is the same one the planner's limits are derived from.
    assert_eq!(*schema, tool_schema());
}

#[test]
fn the_native_tool_is_constructible_without_touching_the_desktop() {
    // Building the tool must not open a platform connection: a machine without
    // a desktop still starts the server, and only a call reports
    // `backend_unavailable` / `permission_denied`.
    let tool = ComputerInputTool::native();
    assert_eq!(tool.definition().name, TOOL_NAME);
    assert_eq!(ComputerInputTool::default().definition().name, TOOL_NAME);
}

#[tokio::test]
async fn forwards_validated_actions_and_returns_the_envelope() {
    let envelope = json!({
        "ok": true,
        "data": {"action": "click", "position": {"x": 1247, "y": 671}, "duration_ms": 384},
        "meta": {"op": "click", "elapsed_ms": 390}
    });
    let runtime = RecordingRuntime::answering(envelope.clone());
    let tool = ComputerInputTool::new(runtime.clone());

    let output = tool
        .invoke(json!({
            "op": "click",
            "target": {"x": 820, "y": 640, "width": 120, "height": 36},
            "behavior": "precise"
        }))
        .await
        .unwrap();
    assert_eq!(output, envelope);

    let recorded = runtime.recorded();
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0].label(), "click");
    assert_eq!(recorded[0].behavior, relay_input_tool::Behavior::Precise);
    match &recorded[0].kind {
        relay_input_tool::ActionKind::Click {
            target,
            button,
            count,
        } => {
            assert_eq!(target.area(), Some((820, 640, 120, 36)));
            assert_eq!(*button, relay_input_tool::MouseButton::Left);
            assert_eq!(*count, 1);
        }
        other => panic!("expected a click, got {other:?}"),
    }
}

#[tokio::test]
async fn a_sequence_is_normalized_into_typed_children() {
    let runtime = RecordingRuntime::answering(json!({"ok": true, "data": {}}));
    let tool = ComputerInputTool::new(runtime.clone());

    tool.invoke(json!({
        "op": "sequence",
        "behavior": "fast",
        "actions": [
            {"op": "click", "x": 800, "y": 600},
            {"op": "type", "text": "hello@example.com"},
            {"op": "key", "key": "Enter"}
        ]
    }))
    .await
    .unwrap();

    let recorded = runtime.recorded();
    assert_eq!(recorded.len(), 1);
    let relay_input_tool::ActionKind::Sequence(actions) = &recorded[0].kind else {
        panic!("expected a sequence");
    };
    assert_eq!(actions.len(), 3);
    assert_eq!(actions[0].label(), "click");
    assert_eq!(actions[1].label(), "type");
    assert_eq!(actions[2].label(), "key");
    // The sequence's profile reaches the children that did not set one.
    assert!(
        actions
            .iter()
            .all(|action| action.behavior == relay_input_tool::Behavior::Fast)
    );
}

#[tokio::test]
async fn rejects_unknown_operations_fields_and_unusable_targets() {
    let runtime = RecordingRuntime::answering(json!({"ok": true}));
    let tool = ComputerInputTool::new(runtime.clone());

    let error = tool
        .invoke(json!({"op": "swipe", "x": 1, "y": 2}))
        .await
        .unwrap_err();
    assert!(arguments_error(error).contains("unknown variant"));

    let error = tool.invoke(json!({"op": "click"})).await.unwrap_err();
    assert!(arguments_error(error).contains("target"));

    let error = tool
        .invoke(json!({"op": "click", "x": 1, "y": 2, "text": "hello"}))
        .await
        .unwrap_err();
    assert!(arguments_error(error).contains("does not accept `text`"));

    let error = tool
        .invoke(json!({"op": "scroll", "direction": "down", "amount": 500}))
        .await
        .unwrap_err();
    assert!(arguments_error(error).contains("amount"));

    let error = tool
        .invoke(json!({"op": "hotkey", "keys": ["CTRL", "SHIFT"]}))
        .await
        .unwrap_err();
    assert!(arguments_error(error).contains("non-modifier"));

    assert!(runtime.recorded().is_empty());
}

#[tokio::test]
async fn returns_runtime_error_envelopes_verbatim() {
    let envelope = json!({
        "ok": false,
        "error": {
            "code": "permission_denied",
            "message": "grant Accessibility to the server process",
            "retryable": false
        }
    });
    let tool = ComputerInputTool::new(RecordingRuntime::answering(envelope.clone()));

    let output = tool
        .invoke(json!({"op": "key", "key": "escape"}))
        .await
        .unwrap();
    assert_eq!(output, envelope);
}
