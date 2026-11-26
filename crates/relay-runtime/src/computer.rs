//! Desktop tool integration layer: `computer_accessibility` and `computer_input`.
//!
//! Both tool implementations were moved over from the harness (`accessibility-tool` / `input-tool`),
//! so this module only does three things:
//!
//! - convert the tool implementation's [`ToolDefinition`] into relay's [`ToolDef`];
//! - reuse one tool instance per process (accessibility handle tables and input backend connections live on the instance);
//! - collapse one call into a block of text that can be fed back, truncating when oversized.
//!
//! Semantic errors (stale element, permission denied, out-of-range coordinates) come back from the tool itself as envelope results,
//! so the model can read `code` and `retryable` and decide the next step; nothing is raised here.

use std::sync::OnceLock;

use relay_accessibility_tool::ComputerAccessibilityTool;
use relay_input_tool::ComputerInputTool;
use relay_openai::ToolDef;
use relay_tool_api::{Tool, ToolDefinition};
use serde_json::Value;

use crate::tools::TOOL_OUTPUT_LIMIT;

fn input() -> &'static ComputerInputTool {
    static TOOL: OnceLock<ComputerInputTool> = OnceLock::new();
    TOOL.get_or_init(ComputerInputTool::native)
}

fn accessibility() -> &'static ComputerAccessibilityTool {
    static TOOL: OnceLock<ComputerAccessibilityTool> = OnceLock::new();
    TOOL.get_or_init(ComputerAccessibilityTool::native)
}

fn convert(definition: &ToolDefinition) -> ToolDef {
    ToolDef::new(
        definition.name.clone(),
        definition.description.clone().unwrap_or_default(),
        definition.parameters.clone(),
    )
}

/// Definition of `computer_accessibility`.
pub(crate) fn accessibility_def() -> ToolDef {
    convert(accessibility().definition())
}

/// Definition of `computer_input`.
pub(crate) fn input_def() -> ToolDef {
    convert(input().definition())
}

/// Run one desktop tool call and return the result text fed back to the model.
pub(crate) async fn invoke(name: &str, arguments: &Value) -> String {
    let result = match name {
        "computer_accessibility" => accessibility().invoke(arguments.clone()).await,
        "computer_input" => input().invoke(arguments.clone()).await,
        other => return format!("unknown tool: {other}"),
    };
    match result {
        Ok(value) => render(&value),
        Err(err) => err.to_string(),
    }
}

/// Flatten the envelope onto one line: tool results go into context verbatim, so indentation is wasted.
///
/// No hard truncation when oversized: half a JSON document can neither be parsed nor read structurally, so emit a valid summary
/// that tells the model the output was compacted and it should narrow the query.
fn render(value: &Value) -> String {
    let text = serde_json::to_string(value)
        .unwrap_or_else(|err| format!("result could not be serialized: {err}"));
    if text.chars().count() <= TOOL_OUTPUT_LIMIT {
        return text;
    }
    let summary = serde_json::json!({
        "ok": value.get("ok").cloned().unwrap_or(Value::Bool(true)),
        "truncated": true,
        "hint": "output exceeded the limit and most of it was omitted; narrow the query and retry, e.g. specify app / element or lower max_nodes / max_depth / limit.",
        "preview": text.chars().take(2000).collect::<String>(),
    });
    serde_json::to_string(&summary).unwrap_or(text)
}
