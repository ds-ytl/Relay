//! Strictly typed request union behind the single `computer_accessibility` tool.
//!
//! The model-facing schema is one flat object with an `op` discriminator, but
//! nothing reaches the runtime before it has been parsed into this
//! discriminated union: unknown operations, unknown fields, wrong field types,
//! and missing required fields are rejected with
//! [`ToolError::InvalidArguments`](relay_tool_api::ToolError::InvalidArguments).

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// Largest accepted `timeout_ms` for `op=wait` (five minutes).
pub const MAX_WAIT_TIMEOUT_MS: u64 = 300_000;

/// Largest accepted `max_results` for `op=query`.
pub const MAX_QUERY_RESULTS: u32 = 500;

/// Largest accepted `depth` for `op=tree`.
pub const MAX_TREE_DEPTH: u32 = 32;

/// Largest accepted `max_nodes` for `op=tree`.
pub const MAX_TREE_NODES: u32 = 10_000;

/// Largest accepted edge of an `op=screenshot` region, in **logical** pixels.
///
/// Also the ceiling for `max_width` / `max_height`, which are **physical**
/// pixels: no real display is wider than this, so anything beyond it is a
/// modelling mistake rather than a very large monitor.
pub const MAX_CAPTURE_EDGE: u32 = 16_384;

/// Largest accepted area of an `op=screenshot` region, in **logical** pixels.
///
/// 8 388 608 logical pixels is 3840 × 2160 — a 4K logical rectangle — which is
/// already 33 Mpx of physical pixels to allocate and PNG-encode on a 2×
/// display.
pub const MAX_CAPTURE_LOGICAL_PIXELS: u64 = 8_388_608;

/// One structured request for the accessibility runtime.
///
/// `op` selects the variant; every variant carries only the fields that
/// operation understands.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum AccessibilityRequest {
    /// `op=apps` — list the applications the runtime can reach.
    Apps(AppsRequest),
    /// `op=query` — find elements by semantic selector.
    Query(QueryRequest),
    /// `op=inspect` — read every property of one known element.
    Inspect(InspectRequest),
    /// `op=tree` — explore a UI subtree, bounded by depth and node budget.
    Tree(TreeRequest),
    /// `op=action` — perform one semantic action on an element or selector.
    Action(ActionRequest),
    /// `op=hit_test` — resolve a screen coordinate to an element.
    HitTest(HitTestRequest),
    /// `op=screenshot` — capture the screen, or one region of it, as a PNG.
    Screenshot(ScreenshotRequest),
    /// `op=wait` — wait for an element state, event driven where possible.
    Wait(WaitRequest),
}

impl AccessibilityRequest {
    /// Stable wire spelling of the operation, also used for logging and for
    /// output bookkeeping.
    #[must_use]
    pub const fn op(&self) -> &'static str {
        match self {
            Self::Apps(_) => "apps",
            Self::Query(_) => "query",
            Self::Inspect(_) => "inspect",
            Self::Tree(_) => "tree",
            Self::Action(_) => "action",
            Self::HitTest(_) => "hit_test",
            Self::Screenshot(_) => "screenshot",
            Self::Wait(_) => "wait",
        }
    }

    /// Cross-field validation that JSON Schema cannot express.
    ///
    /// # Errors
    ///
    /// Returns a human-readable message for requests that are individually
    /// well-formed but semantically unusable.
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Apps(_) | Self::Inspect(_) | Self::HitTest(_) => Ok(()),
            Self::Query(request) => request.validate(),
            Self::Tree(request) => request.validate(),
            Self::Action(request) => request.validate(),
            Self::Screenshot(request) => request.validate(),
            Self::Wait(request) => request.validate(),
        }
    }
}

/// `op=apps` arguments: the runtime decides which applications are visible.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AppsRequest {}

/// `op=query` arguments.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QueryRequest {
    /// Application handle (`app_12`) returned by `op=apps`. Omitted means the
    /// focused application.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub app: Option<String>,
    /// Element scope: omitted starts at the application root.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub element: Option<String>,
    /// Semantic selector; an empty selector matches every element (bounded by
    /// `max_results`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selector: Option<Selector>,
    /// Result cap; omitted uses the runtime default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_results: Option<u32>,
}

impl QueryRequest {
    fn validate(&self) -> Result<(), String> {
        if let Some(max_results) = self.max_results
            && max_results > MAX_QUERY_RESULTS
        {
            return Err(format!(
                "max_results {max_results} exceeds the supported maximum {MAX_QUERY_RESULTS}"
            ));
        }
        Ok(())
    }
}

/// First-version selector: a small set of high-value, semantic fields.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Selector {
    /// Exact role, e.g. `button`, `text_field`, `window`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// Exact accessible name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Case-insensitive substring of the accessible name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name_contains: Option<String>,
    /// Exact value (text field contents, slider value, ...).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    /// Exact accessible description / help text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Match the keyboard-focus state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub focused: Option<bool>,
    /// Match the enabled state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    /// Match the on-screen visibility state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub visible: Option<bool>,
}

/// `op=inspect` arguments.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InspectRequest {
    /// Element handle returned by `op=query` / `op=tree` / `op=hit_test`.
    pub element: String,
}

/// `op=tree` arguments.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TreeRequest {
    /// Application handle; omitted means the focused application.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub app: Option<String>,
    /// Subtree root element handle; omitted means the application root.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub root: Option<String>,
    /// Maximum traversal depth.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth: Option<u32>,
    /// Maximum number of nodes materialized in one call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_nodes: Option<u32>,
}

impl TreeRequest {
    fn validate(&self) -> Result<(), String> {
        if let Some(depth) = self.depth
            && depth > MAX_TREE_DEPTH
        {
            return Err(format!(
                "depth {depth} exceeds the supported maximum {MAX_TREE_DEPTH}"
            ));
        }
        if let Some(max_nodes) = self.max_nodes
            && (max_nodes == 0 || max_nodes > MAX_TREE_NODES)
        {
            return Err(format!(
                "max_nodes must be between 1 and {MAX_TREE_NODES} (got {max_nodes})"
            ));
        }
        Ok(())
    }
}

/// What `op=action` operates on.
///
/// Either a handle obtained from a previous observation, or an inline selector
/// that the runtime resolves (and checks for uniqueness) right before acting.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum ActionTarget {
    /// A previously returned element handle.
    Element(ElementTarget),
    /// A semantic selector resolved at action time.
    Selector(SelectorTarget),
}

/// Handle-based action target.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ElementTarget {
    /// Element handle such as `el_71`.
    pub element: String,
}

/// Selector-based action target.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SelectorTarget {
    /// Application handle; omitted means the focused application.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub app: Option<String>,
    /// Semantic selector; must resolve to exactly one element.
    pub selector: Selector,
}

/// `op=action` arguments.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct ActionRequest {
    /// Handle or selector to act on.
    #[serde(flatten)]
    pub target: ActionTarget,
    /// Semantic action to perform.
    pub action: ElementAction,
    /// Payload for `set_value`, `type_text`, and `select`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<Value>,
}

impl ActionRequest {
    fn validate(&self) -> Result<(), String> {
        if self.action.requires_value() {
            match self.value.as_ref() {
                None => {
                    return Err(format!(
                        "action {:?} requires a value",
                        self.action.as_str()
                    ));
                }
                // The platform layer takes text: a string is passed through, a
                // number (a slider position, an index) is rendered, and
                // anything structural is a modelling mistake worth rejecting
                // before a desktop is touched.
                Some(Value::String(_) | Value::Number(_) | Value::Bool(_)) => {}
                Some(other) => {
                    return Err(format!(
                        "action {:?} takes a scalar value (string or number), got {other}",
                        self.action.as_str()
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Semantic actions the runtime knows how to perform.
///
/// The runtime owns native-action fallbacks (accessibility action, then
/// clickable point, then bounds centre), so the model never has to choose
/// between a semantic and a coordinate interaction.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ElementAction {
    /// Activate the element (button, menu item, link, ...).
    Press,
    /// Move keyboard focus to the element.
    Focus,
    /// Replace the element's value.
    SetValue,
    /// Type text into the element.
    TypeText,
    /// Flip a check box / switch.
    Toggle,
    /// Select an entry in a list, tab group, or combo box.
    Select,
    /// Expand a disclosure / tree node.
    Expand,
    /// Collapse a disclosure / tree node.
    Collapse,
    /// Increase a slider or stepper value.
    Increment,
    /// Decrease a slider or stepper value.
    Decrement,
    /// Scroll the element into view without activating it.
    ScrollIntoView,
}

impl ElementAction {
    /// Stable wire spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Press => "press",
            Self::Focus => "focus",
            Self::SetValue => "set_value",
            Self::TypeText => "type_text",
            Self::Toggle => "toggle",
            Self::Select => "select",
            Self::Expand => "expand",
            Self::Collapse => "collapse",
            Self::Increment => "increment",
            Self::Decrement => "decrement",
            Self::ScrollIntoView => "scroll_into_view",
        }
    }

    /// Whether the action is meaningless without `value`.
    #[must_use]
    pub const fn requires_value(self) -> bool {
        matches!(self, Self::SetValue | Self::TypeText | Self::Select)
    }
}

/// `op=hit_test` arguments.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HitTestRequest {
    /// Horizontal screen coordinate, in logical pixels.
    pub x: f64,
    /// Vertical screen coordinate, in logical pixels.
    pub y: f64,
}

/// `op=screenshot` arguments.
///
/// Geometry is given in **logical** screen pixels — the same coordinate space
/// as `op=hit_test` and `computer_input` — while the saved file holds
/// **physical** pixels. The reply carries both sides (`image.pixel_width`,
/// `image.pixel_height`, `image.scale`, and `origin`), so a pixel the model
/// sees in the image can be turned back into a screen coordinate with
/// `logical = origin + pixel / scale`.
///
/// The four geometry fields are all-or-nothing: all of them capture one
/// region, none of them capture the whole screen. Half a rectangle is a
/// modelling mistake, not a shorthand for "the rest of the screen".
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScreenshotRequest {
    /// Left edge of the region, in logical screen pixels. Negative on a
    /// desktop that extends left of the primary display.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub x: Option<f64>,
    /// Top edge of the region, in logical screen pixels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub y: Option<f64>,
    /// Region width, in logical screen pixels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<f64>,
    /// Region height, in logical screen pixels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<f64>,
    /// Where to write the PNG: an absolute path, or one relative to the
    /// default screenshot directory under the system temp directory. Omitted
    /// picks a unique name in that directory. An existing file is never
    /// overwritten, so a capture cannot destroy earlier output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    /// Downscale the saved image to at most this many physical pixels wide,
    /// preserving the aspect ratio.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_width: Option<u32>,
    /// Downscale the saved image to at most this many physical pixels high,
    /// preserving the aspect ratio.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_height: Option<u32>,
}

impl ScreenshotRequest {
    /// The requested region as `(x, y, width, height)` in logical pixels, or
    /// `None` for a whole-screen capture.
    ///
    /// Balanced only for a request that passed [`Self::validate`]; until then
    /// a half-specified rectangle also reads as `None`.
    #[must_use]
    pub fn region(&self) -> Option<(f64, f64, f64, f64)> {
        match (self.x, self.y, self.width, self.height) {
            (Some(x), Some(y), Some(width), Some(height)) => Some((x, y, width, height)),
            _ => None,
        }
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // checked to be 1..=MAX_CAPTURE_EDGE
    fn validate(&self) -> Result<(), String> {
        let given = [self.x, self.y, self.width, self.height]
            .iter()
            .filter(|value| value.is_some())
            .count();
        if given != 0 && given != 4 {
            return Err(
                "op=screenshot takes either all of x, y, width and height (one region) \
                        or none of them (the whole screen)"
                    .into(),
            );
        }
        for (name, value) in [
            ("x", self.x),
            ("y", self.y),
            ("width", self.width),
            ("height", self.height),
        ] {
            if let Some(value) = value
                && !value.is_finite()
            {
                return Err(format!("{name} must be a finite number, got {value}"));
            }
        }
        if let Some((x, y, width, height)) = self.region() {
            for (name, value) in [("x", x), ("y", y)] {
                if value.abs() > f64::from(MAX_CAPTURE_EDGE) {
                    return Err(format!(
                        "{name} {value} is outside the supported range ±{MAX_CAPTURE_EDGE} \
                         logical pixels"
                    ));
                }
            }
            for (name, value) in [("width", width), ("height", height)] {
                if !(1.0..=f64::from(MAX_CAPTURE_EDGE)).contains(&value) {
                    return Err(format!(
                        "{name} must be between 1 and {MAX_CAPTURE_EDGE} logical pixels \
                         (got {value})"
                    ));
                }
            }
            let pixels = u64::from(width.round() as u32) * u64::from(height.round() as u32);
            if pixels > MAX_CAPTURE_LOGICAL_PIXELS {
                return Err(format!(
                    "a {width}×{height} region is about {pixels} logical pixels, above the \
                     supported maximum {MAX_CAPTURE_LOGICAL_PIXELS}; capture a smaller region"
                ));
            }
        }
        if let Some(path) = &self.path
            && path.trim().is_empty()
        {
            return Err(
                "path must not be empty; omit it to use the default screenshot \
                            directory"
                    .into(),
            );
        }
        for (name, value) in [
            ("max_width", self.max_width),
            ("max_height", self.max_height),
        ] {
            if let Some(value) = value
                && (value == 0 || value > MAX_CAPTURE_EDGE)
            {
                return Err(format!(
                    "{name} must be between 1 and {MAX_CAPTURE_EDGE} physical pixels \
                         (got {value})"
                ));
            }
        }
        Ok(())
    }
}

/// `op=wait` arguments.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WaitRequest {
    /// Application handle; omitted means the focused application.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub app: Option<String>,
    /// Element handle to watch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub element: Option<String>,
    /// Selector to watch; mutually usable with `element`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selector: Option<Selector>,
    /// Condition to wait for.
    pub condition: WaitCondition,
    /// Comparison value for `value_changed`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<Value>,
    /// Budget in milliseconds; omitted uses the runtime default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
}

impl WaitRequest {
    fn validate(&self) -> Result<(), String> {
        if self.element.is_none() && self.selector.is_none() {
            return Err("wait requires either `element` or `selector`".into());
        }
        if self.condition == WaitCondition::ValueChanged && self.value.is_none() {
            return Err("condition \"value_changed\" requires a value".into());
        }
        if let Some(timeout_ms) = self.timeout_ms
            && timeout_ms > MAX_WAIT_TIMEOUT_MS
        {
            return Err(format!(
                "timeout_ms {timeout_ms} exceeds the supported maximum {MAX_WAIT_TIMEOUT_MS}"
            ));
        }
        Ok(())
    }

    /// Wait budget, when the caller asked for one.
    #[must_use]
    pub const fn timeout(&self) -> Option<u64> {
        self.timeout_ms
    }
}

/// Conditions `op=wait` can observe.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WaitCondition {
    /// The element appears.
    Exists,
    /// The element disappears.
    NotExists,
    /// The element becomes visible.
    Visible,
    /// The element becomes enabled.
    Enabled,
    /// The element takes keyboard focus.
    Focused,
    /// The element's value changes away from `value`.
    ValueChanged,
}

/// The single JSON Schema exposed to the model.
///
/// One tool, one flat object, `op` as the discriminator; the per-operation
/// requirements live in the description and are enforced in Rust by
/// [`AccessibilityRequest`].
#[must_use]
pub fn tool_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "op": {
                "type": "string",
                "enum": ["apps", "query", "inspect", "tree", "action", "hit_test", "screenshot", "wait"],
                "description": "Operation to perform."
            },
            "app": {
                "type": "string",
                "description": "Application handle from apps (default focused app)."
            },
            "element": {
                "type": "string",
                "description": "Element handle from query/tree/hit_test, required for inspect or used as query scope."
            },
            "root": {
                "type": "string",
                "description": "Tree root handle (default application root)."
            },
            "selector": {
                "type": "object",
                "description": "Semantic filters; empty matches all, while actions require a unique match.",
                "properties": {
                    "role": {"type": "string", "description": "Exact role, e.g. button, text_field, window"},
                    "name": {"type": "string", "description": "Exact accessible name"},
                    "name_contains": {"type": "string", "description": "Case-insensitive substring of the accessible name"},
                    "value": {"type": "string", "description": "Exact value"},
                    "description": {"type": "string", "description": "Exact accessible description"},
                    "focused": {"type": "boolean", "description": "Match the focus state"},
                    "enabled": {"type": "boolean", "description": "Match the enabled state"},
                    "visible": {"type": "boolean", "description": "Match the visibility state"}
                },
                "additionalProperties": false
            },
            "action": {
                "type": "string",
                "enum": [
                    "press", "focus", "set_value", "type_text", "toggle", "select",
                    "expand", "collapse", "increment", "decrement", "scroll_into_view"
                ],
                "description": "Semantic action; set_value/type_text/select require value."
            },
            "value": {
                "description": "Action payload or comparison value for value_changed."
            },
            "x": {"type": "number", "description": "Hit-test/screenshot x in logical pixels; convert image pixels as origin + pixel/scale."},
            "y": {"type": "number", "description": "Hit-test or screenshot y coordinate in logical pixels."},
            "width": {"type": "number", "minimum": 1, "description": "Screenshot width in logical pixels; provide x/y/width/height together or omit all for full screen."},
            "height": {"type": "number", "minimum": 1, "description": "Screenshot height in logical pixels."},
            "path": {"type": "string", "description": "Screenshot PNG path, absolute or relative to the temp capture directory; omit for a unique name, with no overwrites."},
            "max_width": {"type": "integer", "minimum": 1, "maximum": MAX_CAPTURE_EDGE, "description": "Screenshot width cap in physical pixels, preserving aspect ratio."},
            "max_height": {"type": "integer", "minimum": 1, "maximum": MAX_CAPTURE_EDGE, "description": "Screenshot height cap in physical pixels, preserving aspect ratio."},
            "depth": {"type": "integer", "minimum": 1, "maximum": MAX_TREE_DEPTH, "description": "Maximum tree depth."},
            "max_nodes": {"type": "integer", "minimum": 1, "maximum": MAX_TREE_NODES, "description": "Tree node limit, marked truncated when reached."},
            "max_results": {"type": "integer", "minimum": 1, "maximum": MAX_QUERY_RESULTS, "description": "Maximum query results."},
            "condition": {
                "type": "string",
                "enum": ["exists", "not_exists", "visible", "enabled", "focused", "value_changed"],
                "description": "State to wait for."
            },
            "timeout_ms": {
                "type": "integer",
                "minimum": 0,
                "maximum": MAX_WAIT_TIMEOUT_MS,
                "description": "Wait timeout in milliseconds."
            }
        },
        "required": ["op"],
        "additionalProperties": false
    })
}
