//! `computer_accessibility` — one tool, eight strictly typed operations.
//!
//! The model sees a single function tool whose `op` field selects one of
//! `apps`, `query`, `inspect`, `tree`, `action`, `hit_test`, `screenshot`, or
//! `wait`. The arguments are never forwarded as an untyped bag: they are parsed
//! into the discriminated union [`AccessibilityRequest`], which rejects unknown
//! operations, unknown fields, wrong types, and missing required fields before
//! anything runs. The goal is a small tool surface for the model and a strict
//! operation set underneath it.
//!
//! Execution happens in-process on [`Xa11yRuntime`], which talks to the
//! platform accessibility API (macOS `AXUIElement`, Windows UIA, Linux AT-SPI)
//! through the `xa11y` crate. Two design consequences follow:
//!
//! * **Semantic operations first.** The model addresses elements by semantic
//!   selector or by the handle a previous observation returned; coordinates
//!   only enter through `op=hit_test`, whose job is to turn a screenshot
//!   location back into an element — and `op=screenshot`, which produces the
//!   image that location comes from. Capture geometry is given in *logical*
//!   pixels, the same space as `hit_test` and `computer_input`; the reply
//!   reports the physical pixel size, the scale, and the image's logical
//!   origin, so the two spaces stay apart.
//! * **Semantic failures are results.** Every call answers with one envelope:
//!
//!   ```json
//!   {"ok": true, "data": {}, "meta": {}}
//!   {"ok": false, "error": {"code": "element_not_found", "message": "...", "retryable": true}}
//!   ```
//!
//!   `element_not_found`, `stale_element`, `permission_denied`, and friends go
//!   back to the model verbatim so it can read `code` and `retryable` and pick
//!   its next step. [`ToolError`] is reserved for a malformed call, which
//!   [`AccessibilityRequest`] rejects before any backend work happens.
//!
//! The accessibility FFI lives in `xa11y`, not here: this workspace forbids
//! `unsafe`, and the platform bindings need it. The server process still needs
//! the OS accessibility permission (macOS: *Accessibility* in System Settings →
//! Privacy & Security); without it every operation answers `permission_denied`
//! until a human grants it and the process restarts — `xa11y` caches the
//! failed backend construction.

use std::sync::Arc;

use relay_tool_api::{Tool, ToolDefinition, ToolError, ToolFuture};
use serde_json::Value;
use tracing::debug;

mod envelope;
mod native;
mod registry;
mod request;

pub use native::{DEFAULT_TIMEOUT, Xa11yRuntime};
pub use request::{
    AccessibilityRequest, ActionRequest, ActionTarget, AppsRequest, ElementAction, ElementTarget,
    HitTestRequest, InspectRequest, MAX_QUERY_RESULTS, MAX_TREE_DEPTH, MAX_TREE_NODES,
    MAX_WAIT_TIMEOUT_MS, QueryRequest, Selector, SelectorTarget, TreeRequest, WaitCondition,
    WaitRequest, tool_schema,
};

/// Name the model sees for this tool.
pub const TOOL_NAME: &str = "computer_accessibility";

/// Backend able to execute one structured accessibility request.
///
/// Implemented by [`Xa11yRuntime`] for production and by fakes in tests, so the
/// tool's parsing, validation, and envelope handling stay testable without any
/// accessibility permissions.
pub trait AccessibilityRuntime: Send + Sync + 'static {
    /// Executes one request and returns the runtime's JSON envelope.
    fn execute(&self, request: AccessibilityRequest) -> ToolFuture<'_>;
}

/// The single accessibility tool exposed to the model.
#[derive(Clone)]
pub struct ComputerAccessibilityTool {
    definition: ToolDefinition,
    runtime: Arc<dyn AccessibilityRuntime>,
}

impl ComputerAccessibilityTool {
    /// Creates the tool over one runtime backend.
    ///
    /// # Panics
    ///
    /// Panics only if the embedded JSON Schema is not an object, which cannot
    /// happen because it is a literal.
    #[must_use]
    pub fn new(runtime: Arc<dyn AccessibilityRuntime>) -> Self {
        let definition = ToolDefinition::new(
            TOOL_NAME,
            Some(
                "Inspect and operate desktop UI by op using semantic handles/selectors; re-query stale handles and check retryable errors, with OS accessibility or screenshot permission required.",
            ),
            tool_schema(),
        )
        .expect("accessibility tool schema is valid");
        Self {
            definition: definition.with_strict(false),
            runtime,
        }
    }

    /// Creates the tool over the in-process `xa11y` backend.
    ///
    /// The backend itself is created lazily on the first call, so constructing
    /// the tool never touches the accessibility API (and never fails).
    #[must_use]
    pub fn native() -> Self {
        Self::new(Arc::new(Xa11yRuntime::new()))
    }

    /// Creates the tool over an already-built runtime.
    #[must_use]
    pub fn with_runtime(runtime: Xa11yRuntime) -> Self {
        Self::new(Arc::new(runtime))
    }
}

impl Default for ComputerAccessibilityTool {
    fn default() -> Self {
        Self::native()
    }
}

impl Tool for ComputerAccessibilityTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    fn invoke(&self, arguments: Value) -> ToolFuture<'_> {
        Box::pin(async move {
            let request = parse_request(arguments)?;
            request
                .validate()
                .map_err(|message| ToolError::invalid_arguments(TOOL_NAME, message))?;
            debug!(op = request.op(), "accessibility request dispatched");
            self.runtime.execute(request).await
        })
    }
}

/// Parses one model call into the typed request union.
fn parse_request(arguments: Value) -> Result<AccessibilityRequest, ToolError> {
    serde_json::from_value(arguments).map_err(|error| {
        debug!(error = %error, "rejected malformed accessibility request");
        ToolError::invalid_arguments(
            TOOL_NAME,
            format!(
                "{error}; expected one of \
                 op=apps|query|inspect|tree|action|hit_test|screenshot|wait \
                 with that operation's fields"
            ),
        )
    })
}
