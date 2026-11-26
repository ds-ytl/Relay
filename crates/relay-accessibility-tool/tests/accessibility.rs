//! Tests for the `computer_accessibility` tool surface.
//!
//! The tool is exercised through a recording runtime, so request parsing,
//! validation, schema exposure, and envelope passthrough are covered without
//! accessibility permissions. The operations themselves are covered
//! end-to-end against `xa11y`'s in-memory provider in `native.rs`.

use std::sync::{Arc, Mutex};

use relay_accessibility_tool::{
    AccessibilityRequest, AccessibilityRuntime, ComputerAccessibilityTool, TOOL_NAME,
};
use relay_tool_api::{Tool, ToolError, ToolFuture};
use serde_json::{Value, json};

/// Runtime that records requests and answers with a canned envelope.
#[derive(Default)]
struct RecordingRuntime {
    seen: Mutex<Vec<Value>>,
    reply: Mutex<Value>,
}

impl RecordingRuntime {
    fn answering(reply: Value) -> Arc<Self> {
        let runtime = Self::default();
        *runtime.reply.lock().unwrap() = reply;
        Arc::new(runtime)
    }

    fn recorded(&self) -> Vec<Value> {
        self.seen.lock().unwrap().clone()
    }
}

impl AccessibilityRuntime for RecordingRuntime {
    fn execute(&self, request: AccessibilityRequest) -> ToolFuture<'_> {
        Box::pin(async move {
            self.seen
                .lock()
                .unwrap()
                .push(serde_json::to_value(request).unwrap());
            Ok(self.reply.lock().unwrap().clone())
        })
    }
}

fn tool_over(runtime: Arc<RecordingRuntime>) -> ComputerAccessibilityTool {
    ComputerAccessibilityTool::new(runtime)
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
fn exposes_one_tool_with_eight_operations() {
    let tool = tool_over(RecordingRuntime::answering(json!({})));
    let definition = tool.definition();
    assert_eq!(definition.name, TOOL_NAME);
    assert!(!definition.strict);
    let schema = &definition.parameters;
    let operations = schema["properties"]["op"]["enum"].as_array().unwrap();
    assert_eq!(
        operations,
        &vec![
            json!("apps"),
            json!("query"),
            json!("inspect"),
            json!("tree"),
            json!("action"),
            json!("hit_test"),
            json!("screenshot"),
            json!("wait"),
        ]
    );
    assert_eq!(schema["required"], json!(["op"]));
    assert_eq!(schema["additionalProperties"], json!(false));
    assert_eq!(definition.as_responses_tool()["type"], json!("function"));
}

#[test]
fn the_native_tool_is_constructible_without_touching_the_accessibility_api() {
    // Building the tool must not create the platform provider: a machine
    // without a desktop (CI, a headless container) still starts the server,
    // and only a call reports `permission_denied` / `backend_unavailable`.
    let tool = ComputerAccessibilityTool::native();
    assert_eq!(tool.definition().name, TOOL_NAME);
    assert_eq!(
        ComputerAccessibilityTool::default().definition().name,
        TOOL_NAME
    );
}

#[tokio::test]
async fn forwards_normalized_requests_and_returns_the_envelope() {
    let runtime = RecordingRuntime::answering(json!({
        "ok": true,
        "data": {"elements": [{"id": "el_71", "role": "button", "name": "Submit"}]},
        "meta": {"op": "query", "elapsed_ms": 3}
    }));
    let tool = tool_over(Arc::clone(&runtime));

    let output = tool
        .invoke(json!({
            "op": "query",
            "app": "app_12",
            "selector": {"role": "button", "name": "Submit"},
            "max_results": 5
        }))
        .await
        .unwrap();
    assert_eq!(output["ok"], json!(true));
    assert_eq!(output["data"]["elements"][0]["id"], json!("el_71"));

    let output = tool
        .invoke(json!({
            "op": "action",
            "element": "el_71",
            "action": "press"
        }))
        .await
        .unwrap();
    assert_eq!(output["ok"], json!(true));

    let seen = runtime.recorded();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0]["op"], json!("query"));
    assert_eq!(seen[0]["selector"]["name"], json!("Submit"));
    assert_eq!(seen[0]["max_results"], json!(5));
    assert!(seen[0].get("element").is_none());
    assert_eq!(seen[1]["op"], json!("action"));
    assert_eq!(seen[1]["element"], json!("el_71"));
    assert_eq!(seen[1]["action"], json!("press"));
}

#[tokio::test]
async fn accepts_selector_targets_and_selector_free_queries() {
    let runtime = RecordingRuntime::answering(json!({"ok": true, "data": {}}));
    let tool = tool_over(Arc::clone(&runtime));

    tool.invoke(json!({
        "op": "action",
        "app": "app_12",
        "selector": {"role": "button", "name_contains": "sub"},
        "action": "set_value",
        "value": "hello"
    }))
    .await
    .unwrap();
    tool.invoke(json!({"op": "query"})).await.unwrap();
    tool.invoke(json!({"op": "apps"})).await.unwrap();

    let seen = runtime.recorded();
    assert_eq!(seen[0]["app"], json!("app_12"));
    assert_eq!(seen[0]["selector"]["name_contains"], json!("sub"));
    assert_eq!(seen[0]["value"], json!("hello"));
    assert_eq!(seen[1], json!({"op": "query"}));
    assert_eq!(seen[2], json!({"op": "apps"}));
}

#[tokio::test]
async fn rejects_unknown_operations_and_fields() {
    let runtime = RecordingRuntime::answering(json!({"ok": true}));
    let tool = tool_over(Arc::clone(&runtime));

    let error = tool
        .invoke(json!({"op": "click", "x": 1, "y": 2}))
        .await
        .unwrap_err();
    assert!(arguments_error(error).contains("unknown variant"));

    let error = tool
        .invoke(json!({"op": "apps", "selector": {"role": "button"}}))
        .await
        .unwrap_err();
    assert!(arguments_error(error).contains("unknown field"));

    let error = tool.invoke(json!({"op": "inspect"})).await.unwrap_err();
    assert!(arguments_error(error).contains("missing field"));

    let error = tool
        .invoke(json!({
            "op": "hit_test",
            "x": "middle",
            "y": 720
        }))
        .await
        .unwrap_err();
    assert!(arguments_error(error).contains("invalid type"));

    assert!(runtime.recorded().is_empty());
}

#[tokio::test]
async fn rejects_semantically_unusable_requests() {
    let runtime = RecordingRuntime::answering(json!({"ok": true}));
    let tool = tool_over(Arc::clone(&runtime));

    let error = tool
        .invoke(json!({"op": "wait", "condition": "exists"}))
        .await
        .unwrap_err();
    assert!(arguments_error(error).contains("element"));

    let error = tool
        .invoke(json!({"op": "action", "element": "el_1", "action": "set_value"}))
        .await
        .unwrap_err();
    assert!(arguments_error(error).contains("requires a value"));

    let error = tool
        .invoke(json!({
            "op": "action",
            "element": "el_1",
            "action": "type_text",
            "value": {"text": "hello"}
        }))
        .await
        .unwrap_err();
    assert!(arguments_error(error).contains("scalar"));

    let error = tool
        .invoke(json!({
            "op": "wait",
            "element": "el_1",
            "condition": "value_changed"
        }))
        .await
        .unwrap_err();
    assert!(arguments_error(error).contains("value_changed"));

    let error = tool
        .invoke(json!({
            "op": "tree",
            "depth": 64
        }))
        .await
        .unwrap_err();
    assert!(arguments_error(error).contains("depth"));

    let error = tool
        .invoke(json!({
            "op": "tree",
            "max_nodes": 0
        }))
        .await
        .unwrap_err();
    assert!(arguments_error(error).contains("max_nodes"));

    let error = tool
        .invoke(json!({
            "op": "query",
            "max_results": 10_000
        }))
        .await
        .unwrap_err();
    assert!(arguments_error(error).contains("max_results"));

    let error = tool
        .invoke(json!({
            "op": "wait",
            "element": "el_1",
            "condition": "visible",
            "timeout_ms": 3_600_000
        }))
        .await
        .unwrap_err();
    assert!(arguments_error(error).contains("timeout_ms"));

    assert!(runtime.recorded().is_empty());
}

#[tokio::test]
async fn forwards_screenshot_arguments_in_logical_pixels() {
    let runtime = RecordingRuntime::answering(json!({"ok": true, "data": {}}));
    let tool = tool_over(Arc::clone(&runtime));

    // No geometry at all is a whole-screen capture, and sends nothing extra.
    tool.invoke(json!({"op": "screenshot"})).await.unwrap();
    tool.invoke(json!({
        "op": "screenshot",
        "x": 100.5,
        "y": 80,
        "width": 720,
        "height": 450,
        "path": "shots/after-click.png",
        "max_width": 1024
    }))
    .await
    .unwrap();

    let seen = runtime.recorded();
    assert_eq!(seen[0], json!({"op": "screenshot"}));
    assert_eq!(seen[1]["op"], json!("screenshot"));
    assert_eq!(seen[1]["x"], json!(100.5));
    assert_eq!(seen[1]["y"], json!(80.0));
    assert_eq!(seen[1]["width"], json!(720.0));
    assert_eq!(seen[1]["height"], json!(450.0));
    assert_eq!(seen[1]["path"], json!("shots/after-click.png"));
    assert_eq!(seen[1]["max_width"], json!(1024));
    assert!(seen[1].get("max_height").is_none());
}

#[tokio::test]
async fn rejects_screenshot_requests_that_cannot_be_captured() {
    let runtime = RecordingRuntime::answering(json!({"ok": true}));
    let tool = tool_over(Arc::clone(&runtime));

    // Half a rectangle is a modelling mistake, not "the rest of the screen".
    let error = tool
        .invoke(json!({"op": "screenshot", "x": 10, "y": 20}))
        .await
        .unwrap_err();
    assert!(arguments_error(error).contains("all of x, y, width and height"));

    let error = tool
        .invoke(json!({"op": "screenshot", "x": 0, "y": 0, "width": 0, "height": 10}))
        .await
        .unwrap_err();
    assert!(arguments_error(error).contains("width"));

    // 4000 × 4000 logical pixels is above the supported area.
    let error = tool
        .invoke(json!({
            "op": "screenshot",
            "x": 0,
            "y": 0,
            "width": 4000,
            "height": 4000
        }))
        .await
        .unwrap_err();
    assert!(arguments_error(error).contains("logical pixels"));

    let error = tool
        .invoke(json!({"op": "screenshot", "path": "   "}))
        .await
        .unwrap_err();
    assert!(arguments_error(error).contains("path"));

    let error = tool
        .invoke(json!({"op": "screenshot", "max_width": 0}))
        .await
        .unwrap_err();
    assert!(arguments_error(error).contains("max_width"));

    assert!(runtime.recorded().is_empty());
}

#[tokio::test]
async fn returns_runtime_error_envelopes_verbatim() {
    let envelope = json!({
        "ok": false,
        "error": {
            "code": "stale_element",
            "message": "element el_71 no longer exists",
            "retryable": true
        }
    });
    let tool = tool_over(RecordingRuntime::answering(envelope.clone()));

    let output = tool
        .invoke(json!({"op": "inspect", "element": "el_71"}))
        .await
        .unwrap();
    assert_eq!(output, envelope);
}
