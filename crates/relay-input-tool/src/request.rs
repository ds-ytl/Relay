//! Strictly typed request behind the single `computer_input` tool.
//!
//! The model-facing schema is one flat object discriminated by `op`, but
//! nothing reaches the planner before it has been parsed into [`Action`]:
//! unknown operations, unknown fields, wrong field types, missing required
//! fields, and fields that do not belong to the chosen operation are all
//! rejected with
//! [`ToolError::InvalidArguments`](relay_tool_api::ToolError::InvalidArguments).
//!
//! Two shape decisions are worth spelling out:
//!
//! * **Targets are rectangles or exact points.** `{"x": .., "y": ..}` aims at
//!   one pixel; adding `width`/`height` (or using the `target` object) hands
//!   the runtime a control's bounds, and the planner picks a safe pixel inside
//!   it. The model never has to compute a centre itself.
//! * **Behaviour is one word.** `fast`, `natural`, or `precise` replace the
//!   dozen timing knobs (duration, jitter, curve, hold, typing speed) that
//!   would otherwise leak into the schema.

use std::time::Duration;

use enigo::Key;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// Largest accepted `type` payload, in characters.
pub const MAX_TEXT_CHARS: usize = 500;

/// Largest accepted number of actions in `op=sequence`.
pub const MAX_SEQUENCE_ACTIONS: usize = 16;

/// Largest accepted `amount` for `op=scroll`, in wheel notches.
pub const MAX_SCROLL_NOTCHES: i32 = 50;

/// Largest accepted `clicks` for `op=click`.
pub const MAX_CLICKS: u8 = 3;

/// Largest accepted number of keys in `op=hotkey`.
pub const MAX_HOTKEY_KEYS: usize = 5;

/// Largest accepted absolute coordinate, so a typo cannot move the pointer
/// beyond any real display.
pub const MAX_COORDINATE: i32 = 32_000;

/// One call may busy the desktop for at most a minute; longer intents are
/// split by the model instead of blocking the tool loop.
pub const MAX_ACTION_MS: u64 = 60_000;

/// Operation discriminator of `computer.input`.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Op {
    /// Move the pointer to a target.
    Move,
    /// Click a target once (`clicks` may repeat it).
    Click,
    /// Click a target twice in a row.
    DoubleClick,
    /// Press at `from`, glide to `to`, release.
    Drag,
    /// Scroll vertically, optionally after moving to a target.
    Scroll,
    /// Type text at the cursor.
    Type,
    /// Press and release one key.
    Key,
    /// Press modifiers, tap one key, release the modifiers.
    Hotkey,
    /// Run up to [`MAX_SEQUENCE_ACTIONS`] actions with natural gaps.
    Sequence,
}

impl Op {
    /// Every operation, in schema order.
    #[must_use]
    pub const fn all() -> [Self; 9] {
        [
            Self::Move,
            Self::Click,
            Self::DoubleClick,
            Self::Drag,
            Self::Scroll,
            Self::Type,
            Self::Key,
            Self::Hotkey,
            Self::Sequence,
        ]
    }

    /// Stable wire spelling of the operation, also used for logging.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Move => "move",
            Self::Click => "click",
            Self::DoubleClick => "double_click",
            Self::Drag => "drag",
            Self::Scroll => "scroll",
            Self::Type => "type",
            Self::Key => "key",
            Self::Hotkey => "hotkey",
            Self::Sequence => "sequence",
        }
    }
}

/// Humanization profile: how much of a human hand and keyboard to imitate.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Behavior {
    /// Quick and direct: for routine navigation the user is watching.
    Fast,
    /// The default: arc, acceleration, jitter, and pauses a hand would have.
    #[default]
    Natural,
    /// Slow and steady: for small targets, sliders, and drags that must land.
    Precise,
}

impl Behavior {
    /// Stable wire spelling.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Natural => "natural",
            Self::Precise => "precise",
        }
    }
}

/// Mouse button used by click and drag.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MouseButton {
    /// Primary button.
    #[default]
    Left,
    /// Middle button (wheel click).
    Middle,
    /// Secondary button, usually a context menu.
    Right,
}

impl MouseButton {
    /// Stable wire spelling.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Left => "left",
            Self::Middle => "middle",
            Self::Right => "right",
        }
    }

    /// Platform button behind this name.
    #[must_use]
    pub const fn enigo(self) -> enigo::Button {
        match self {
            Self::Left => enigo::Button::Left,
            Self::Middle => enigo::Button::Middle,
            Self::Right => enigo::Button::Right,
        }
    }
}

/// Wheel direction for `op=scroll`.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScrollDirection {
    /// Scroll towards the top of the document.
    Up,
    /// Scroll towards the bottom of the document.
    Down,
}

impl ScrollDirection {
    /// Stable wire spelling.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Up => "up",
            Self::Down => "down",
        }
    }

    /// Sign of one wheel notch in the platform's convention (down is positive).
    #[must_use]
    pub const fn sign(self) -> i32 {
        match self {
            Self::Up => -1,
            Self::Down => 1,
        }
    }
}

/// One exact pixel.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Point {
    /// Horizontal position, in logical screen pixels.
    pub x: i32,
    /// Vertical position, in logical screen pixels.
    pub y: i32,
}

/// Where an action should land: an exact pixel, or a control's bounds from
/// which the planner picks a safe pixel.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Target {
    /// Left edge (or the exact x when no size is given).
    pub x: i32,
    /// Top edge (or the exact y when no size is given).
    pub y: i32,
    /// Width of the control; required together with `height`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<i32>,
    /// Height of the control; required together with `width`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<i32>,
}

impl Target {
    /// `Some((x, y, width, height))` when this target is a real rectangle.
    #[must_use]
    pub const fn area(self) -> Option<(i32, i32, i32, i32)> {
        match (self.width, self.height) {
            (Some(width), Some(height)) if width > 0 && height > 0 => {
                Some((self.x, self.y, width, height))
            }
            _ => None,
        }
    }

    /// Rejects half-given or absurd geometry.
    fn check(self) -> Result<Self, String> {
        match (self.width, self.height) {
            (Some(_), None) | (None, Some(_)) => {
                return Err("a target size needs `width` and `height` together".into());
            }
            (Some(width), Some(height)) if width <= 0 || height <= 0 => {
                return Err("target `width` and `height` must be positive".into());
            }
            _ => {}
        }
        for (name, value) in [
            ("x", self.x),
            ("y", self.y),
            ("width", self.width.unwrap_or(0)),
            ("height", self.height.unwrap_or(0)),
        ] {
            if value.abs() > MAX_COORDINATE {
                return Err(format!("target `{name}` must be within ±{MAX_COORDINATE}"));
            }
        }
        Ok(self)
    }
}

/// One validated desktop action handed to the runtime.
#[derive(Clone, Debug, PartialEq)]
pub struct Action {
    /// Humanization profile chosen for this action.
    pub behavior: Behavior,
    /// What the action does.
    pub kind: ActionKind,
}

impl Action {
    /// Stable wire spelling of the action, also used in result envelopes.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        self.kind.label()
    }
}

/// The operation itself, with every field already type-checked.
#[derive(Clone, Debug, PartialEq)]
pub enum ActionKind {
    /// Move the pointer to a target.
    Move {
        /// Where to land.
        target: Target,
    },
    /// Click a target `count` times.
    Click {
        /// Where to land.
        target: Target,
        /// Which button.
        button: MouseButton,
        /// How many clicks, in one uninterrupted burst.
        count: u8,
    },
    /// Hold the button from `from` to `to`.
    Drag {
        /// Where to press.
        from: Target,
        /// Where to release.
        to: Target,
        /// Which button.
        button: MouseButton,
    },
    /// Scroll vertically, optionally after moving to a target first.
    Scroll {
        /// Where to point the wheel, when the caller named a place.
        target: Option<Target>,
        /// Which way.
        direction: ScrollDirection,
        /// How many notches.
        amount: i32,
    },
    /// Type text at the current keyboard focus.
    Type {
        /// The text to enter.
        text: String,
    },
    /// Press and release one key.
    Key(Key),
    /// Press modifiers, tap one key, release the modifiers.
    Hotkey(Vec<Key>),
    /// Run several actions in one tool call.
    Sequence(Vec<Action>),
}

impl ActionKind {
    /// Stable wire spelling of the operation.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Move { .. } => "move",
            Self::Click { .. } => "click",
            Self::Drag { .. } => "drag",
            Self::Scroll { .. } => "scroll",
            Self::Type { .. } => "type",
            Self::Key(_) => "key",
            Self::Hotkey(_) => "hotkey",
            Self::Sequence(_) => "sequence",
        }
    }
}

/// One parsed `computer.input` call.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InputRequest {
    /// Which operation to run.
    pub op: Op,
    /// Humanization profile; inherits the sequence's profile when nested.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub behavior: Option<Behavior>,
    /// Exact horizontal position of an inline target.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub x: Option<i32>,
    /// Exact vertical position of an inline target.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub y: Option<i32>,
    /// Width of an inline target.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<i32>,
    /// Height of an inline target.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<i32>,
    /// Target as one object, either `{x, y}` or `{x, y, width, height}`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<Target>,
    /// Mouse button for `click` and `drag`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub button: Option<MouseButton>,
    /// Number of clicks for `op=click`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clicks: Option<u8>,
    /// Drag origin.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from: Option<Target>,
    /// Drag destination.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub to: Option<Target>,
    /// Wheel direction for `op=scroll`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub direction: Option<ScrollDirection>,
    /// Wheel notches for `op=scroll`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub amount: Option<i32>,
    /// Text for `op=type`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Single key name for `op=key`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    /// Key names for `op=hotkey`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keys: Option<Vec<String>>,
    /// Actions for `op=sequence`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actions: Option<Vec<InputRequest>>,
}

/// Fields accepted by every operation.
const COMMON: [&str; 1] = ["behavior"];

impl InputRequest {
    /// Turns one parsed call into a validated [`Action`].
    ///
    /// # Errors
    ///
    /// Returns a human-readable message for well-formed JSON that cannot be
    /// executed: a missing target, a field the operation does not own, an
    /// unknown key name, or an action that would busy the desktop for longer
    /// than [`MAX_ACTION_MS`].
    pub fn validate(self) -> Result<Action, String> {
        self.validate_with(Behavior::Natural)
    }

    /// Validates a nested action, inheriting the sequence's profile.
    fn validate_with(self, inherited: Behavior) -> Result<Action, String> {
        let behavior = self.behavior.unwrap_or(inherited);
        let kind = self.build_kind(behavior)?;
        let action = Action { behavior, kind };
        let nominal = crate::plan::nominal_duration(&action);
        if nominal > Duration::from_millis(MAX_ACTION_MS) {
            return Err(format!(
                "op={} would keep the desktop busy for about {}s, over the {}s limit; \
                 split it into several calls or use behavior=fast",
                action.label(),
                nominal.as_secs(),
                MAX_ACTION_MS / 1000
            ));
        }
        Ok(action)
    }

    #[allow(clippy::too_many_lines)]
    fn build_kind(&self, behavior: Behavior) -> Result<ActionKind, String> {
        match self.op {
            Op::Move => {
                self.reject_extra(&["x", "y", "width", "height", "target"])?;
                Ok(ActionKind::Move {
                    target: self.required_target()?,
                })
            }
            Op::Click => {
                self.reject_extra(&["x", "y", "width", "height", "target", "button", "clicks"])?;
                let count = self.clicks.unwrap_or(1);
                if count == 0 || count > MAX_CLICKS {
                    return Err(format!(
                        "`clicks` must be between 1 and {MAX_CLICKS}, got {count}"
                    ));
                }
                Ok(ActionKind::Click {
                    target: self.required_target()?,
                    button: self.button.unwrap_or_default(),
                    count,
                })
            }
            Op::DoubleClick => {
                self.reject_extra(&["x", "y", "width", "height", "target", "button"])?;
                Ok(ActionKind::Click {
                    target: self.required_target()?,
                    button: self.button.unwrap_or_default(),
                    count: 2,
                })
            }
            Op::Drag => {
                self.reject_extra(&["from", "to", "button"])?;
                Ok(ActionKind::Drag {
                    from: Self::required_endpoint("from", self.from)?,
                    to: Self::required_endpoint("to", self.to)?,
                    button: self.button.unwrap_or_default(),
                })
            }
            Op::Scroll => {
                self.reject_extra(&["x", "y", "width", "height", "target", "direction", "amount"])?;
                let direction = self
                    .direction
                    .ok_or_else(|| "op=scroll needs `direction` (up or down)".to_owned())?;
                let amount = self.amount.unwrap_or(3);
                if amount == 0 || amount > MAX_SCROLL_NOTCHES {
                    return Err(format!(
                        "`amount` must be between 1 and {MAX_SCROLL_NOTCHES} notches, got {amount}"
                    ));
                }
                Ok(ActionKind::Scroll {
                    target: self.optional_target()?,
                    direction,
                    amount,
                })
            }
            Op::Type => {
                self.reject_extra(&["text"])?;
                let text = self.text.clone().unwrap_or_default();
                let chars = text.chars().count();
                if chars == 0 {
                    return Err("op=type needs a non-empty `text`".into());
                }
                if chars > MAX_TEXT_CHARS {
                    return Err(format!(
                        "`text` is {chars} characters, over the {MAX_TEXT_CHARS} character limit; \
                         split it into several calls"
                    ));
                }
                Ok(ActionKind::Type { text })
            }
            Op::Key => {
                self.reject_extra(&["key"])?;
                let name = self
                    .key
                    .as_deref()
                    .ok_or_else(|| "op=key needs `key`".to_owned())?;
                Ok(ActionKind::Key(parse_key(name)?))
            }
            Op::Hotkey => {
                self.reject_extra(&["keys"])?;
                Ok(ActionKind::Hotkey(self.parse_hotkey()?))
            }
            Op::Sequence => {
                self.reject_extra(&["actions"])?;
                let actions = self
                    .actions
                    .as_ref()
                    .ok_or_else(|| "op=sequence needs `actions`".to_owned())?;
                if actions.is_empty() {
                    return Err("op=sequence needs at least one action".into());
                }
                if actions.len() > MAX_SEQUENCE_ACTIONS {
                    return Err(format!(
                        "op=sequence accepts at most {MAX_SEQUENCE_ACTIONS} actions, got {}",
                        actions.len()
                    ));
                }
                let mut validated = Vec::with_capacity(actions.len());
                for (index, action) in actions.iter().enumerate() {
                    if action.op == Op::Sequence {
                        return Err(format!(
                            "actions[{index}] is a nested sequence; flatten it into one action list"
                        ));
                    }
                    validated.push(action.clone().validate_with(behavior)?);
                }
                Ok(ActionKind::Sequence(validated))
            }
        }
    }

    /// Parses `keys` into modifiers plus one key, in press order.
    fn parse_hotkey(&self) -> Result<Vec<Key>, String> {
        let names = self
            .keys
            .as_ref()
            .ok_or_else(|| "op=hotkey needs `keys`".to_owned())?;
        if names.is_empty() {
            return Err("op=hotkey needs at least one key".into());
        }
        if names.len() > MAX_HOTKEY_KEYS {
            return Err(format!(
                "op=hotkey accepts at most {MAX_HOTKEY_KEYS} keys, got {}",
                names.len()
            ));
        }
        let mut keys = Vec::with_capacity(names.len());
        for name in names {
            keys.push(parse_key(name)?);
        }
        if !keys.iter().any(|key| !is_modifier(*key)) {
            return Err("op=hotkey needs one non-modifier key besides CTRL/SHIFT/ALT/META".into());
        }
        Ok(keys)
    }

    /// Names of the optional fields this request actually set.
    fn set_fields(&self) -> Vec<&'static str> {
        let mut fields = Vec::new();
        for (name, present) in [
            ("behavior", self.behavior.is_some()),
            ("x", self.x.is_some()),
            ("y", self.y.is_some()),
            ("width", self.width.is_some()),
            ("height", self.height.is_some()),
            ("target", self.target.is_some()),
            ("button", self.button.is_some()),
            ("clicks", self.clicks.is_some()),
            ("from", self.from.is_some()),
            ("to", self.to.is_some()),
            ("direction", self.direction.is_some()),
            ("amount", self.amount.is_some()),
            ("text", self.text.is_some()),
            ("key", self.key.is_some()),
            ("keys", self.keys.is_some()),
            ("actions", self.actions.is_some()),
        ] {
            if present {
                fields.push(name);
            }
        }
        fields
    }

    fn reject_extra(&self, allowed: &[&str]) -> Result<(), String> {
        for field in self.set_fields() {
            let known = COMMON.contains(&field) || allowed.contains(&field);
            if !known {
                let mut allowed: Vec<&str> = allowed.to_vec();
                allowed.extend(COMMON);
                allowed.sort_unstable();
                return Err(format!(
                    "op={} does not accept `{field}`; it accepts op, {}",
                    self.op.label(),
                    allowed.join(", ")
                ));
            }
        }
        Ok(())
    }

    /// The target of a pointer operation, from `target` or from `x`/`y`.
    fn required_target(&self) -> Result<Target, String> {
        self.optional_target()?.ok_or_else(|| {
            format!(
                "op={} needs a `target` object ({{x, y}} or {{x, y, width, height}}) \
                 or inline `x` and `y`",
                self.op.label()
            )
        })
    }

    /// The target of a pointer operation, when the operation allows none.
    fn optional_target(&self) -> Result<Option<Target>, String> {
        let inline = match (self.x, self.y) {
            (Some(x), Some(y)) => Some(Target {
                x,
                y,
                width: self.width,
                height: self.height,
            }),
            (None, None) if self.width.is_none() && self.height.is_none() => None,
            _ => return Err("`x` and `y` must be given together".into()),
        };
        let target = match (inline, self.target) {
            (Some(_), Some(_)) => return Err("give either `target` or `x`/`y`, not both".into()),
            (Some(target), None) | (None, Some(target)) => target,
            (None, None) => return Ok(None),
        };
        target.check().map(Some)
    }

    fn required_endpoint(field: &str, value: Option<Target>) -> Result<Target, String> {
        let target = value.ok_or_else(|| {
            format!("op=drag needs `{field}` ({{x, y}} or {{x, y, width, height}})")
        })?;
        target.check()
    }
}

/// The cross-platform modifiers the schema knows.
#[must_use]
pub const fn is_modifier(key: Key) -> bool {
    matches!(key, Key::Control | Key::Shift | Key::Alt | Key::Meta)
}

/// `f1` … `f20`, the only multi-character names that are not spelled out.
const FUNCTION_KEYS: [Key; 20] = [
    Key::F1,
    Key::F2,
    Key::F3,
    Key::F4,
    Key::F5,
    Key::F6,
    Key::F7,
    Key::F8,
    Key::F9,
    Key::F10,
    Key::F11,
    Key::F12,
    Key::F13,
    Key::F14,
    Key::F15,
    Key::F16,
    Key::F17,
    Key::F18,
    Key::F19,
    Key::F20,
];

/// Parses one cross-platform key name into enigo's key vocabulary.
///
/// `CTRL`/`SHIFT`/`ALT`/`META` are the modifier spellings the model uses;
/// `META` is Command on macOS, Super on Linux, and Windows on Windows. A single
/// character is typed with that character, and letters are matched by physical
/// position, so `Cmd+A` is correct on every layout.
///
/// # Errors
///
/// Returns a message listing the accepted names when the key is unknown.
pub fn parse_key(name: &str) -> Result<Key, String> {
    let normalized: String = name
        .trim()
        .to_lowercase()
        .chars()
        .filter(|character| !matches!(character, '-' | '_' | ' '))
        .collect();
    let named = match normalized.as_str() {
        "enter" | "return" => Some(Key::Return),
        "escape" | "esc" => Some(Key::Escape),
        "tab" => Some(Key::Tab),
        "space" | "spacebar" => Some(Key::Space),
        "backspace" => Some(Key::Backspace),
        "delete" | "del" | "forwarddelete" => Some(Key::Delete),
        "up" | "uparrow" => Some(Key::UpArrow),
        "down" | "downarrow" => Some(Key::DownArrow),
        "left" | "leftarrow" => Some(Key::LeftArrow),
        "right" | "rightarrow" => Some(Key::RightArrow),
        "home" => Some(Key::Home),
        "end" => Some(Key::End),
        "pageup" | "pgup" => Some(Key::PageUp),
        "pagedown" | "pgdn" => Some(Key::PageDown),
        "capslock" => Some(Key::CapsLock),
        "ctrl" | "control" => Some(Key::Control),
        "shift" => Some(Key::Shift),
        "alt" | "option" => Some(Key::Alt),
        "meta" | "cmd" | "command" | "super" | "win" => Some(Key::Meta),
        _ => None,
    };
    if let Some(key) = named {
        return Ok(key);
    }
    if let Some(index) = (1..=FUNCTION_KEYS.len()).find(|index| normalized == format!("f{index}")) {
        return Ok(FUNCTION_KEYS[index - 1]);
    }
    let mut characters = name.trim().chars();
    match (characters.next(), characters.next()) {
        (Some(character), None) => Ok(Key::Unicode(character)),
        _ => Err(format!(
            "unknown key {name:?}; use a single character, F1..F20, or one of \
             enter, escape, tab, space, backspace, delete, up, down, left, right, \
             home, end, pageup, pagedown, ctrl, shift, alt, meta"
        )),
    }
}

/// The wire spelling of a key, for result envelopes.
#[must_use]
pub fn key_label(key: Key) -> String {
    let named = match key {
        Key::Return => Some("enter"),
        Key::Escape => Some("escape"),
        Key::Tab => Some("tab"),
        Key::Space => Some("space"),
        Key::Backspace => Some("backspace"),
        Key::Delete => Some("delete"),
        Key::UpArrow => Some("up"),
        Key::DownArrow => Some("down"),
        Key::LeftArrow => Some("left"),
        Key::RightArrow => Some("right"),
        Key::Home => Some("home"),
        Key::End => Some("end"),
        Key::PageUp => Some("pageup"),
        Key::PageDown => Some("pagedown"),
        Key::CapsLock => Some("capslock"),
        Key::Control => Some("ctrl"),
        Key::Shift => Some("shift"),
        Key::Alt => Some("alt"),
        Key::Meta => Some("meta"),
        _ => None,
    };
    if let Some(name) = named {
        return name.to_owned();
    }
    if let Some(index) = FUNCTION_KEYS.iter().position(|candidate| *candidate == key) {
        return format!("f{}", index + 1);
    }
    match key {
        Key::Unicode(character) => character.to_string(),
        other => format!("{other:?}"),
    }
}

/// The JSON Schema exposed to the model: one flat object with `op`.
#[must_use]
pub fn tool_schema() -> Value {
    let operations: Vec<&str> = Op::all().into_iter().map(Op::label).collect();
    let target_object = || {
        json!({
            "type": "object",
            "description": "Target point {x, y} or preferred control bounds {x, y, width, height}.",
            "properties": {
                "x": {"type": "integer", "minimum": -MAX_COORDINATE, "maximum": MAX_COORDINATE, "description": "Horizontal position in logical screen pixels."},
                "y": {"type": "integer", "minimum": -MAX_COORDINATE, "maximum": MAX_COORDINATE, "description": "Vertical position in logical screen pixels."},
                "width": {"type": "integer", "minimum": 1, "maximum": MAX_COORDINATE, "description": "Control width; optional, but only together with height."},
                "height": {"type": "integer", "minimum": 1, "maximum": MAX_COORDINATE, "description": "Control height; optional, but only together with width."}
            },
            "required": ["x", "y"],
            "additionalProperties": false
        })
    };
    json!({
        "type": "object",
        "properties": {
            "op": {
                "type": "string",
                "enum": operations,
                "description": "Operation to run."
            },
            "behavior": {
                "type": "string",
                "enum": ["fast", "natural", "precise"],
                "description": "Pointer style: fast, natural (default) or precise for small targets and drags."
            },
            "x": {"type": "integer", "minimum": -MAX_COORDINATE, "maximum": MAX_COORDINATE, "description": "Target x for move/click/double_click/scroll, paired with y."},
            "y": {"type": "integer", "minimum": -MAX_COORDINATE, "maximum": MAX_COORDINATE, "description": "Target y for move/click/double_click/scroll, paired with x."},
            "width": {"type": "integer", "minimum": 1, "maximum": MAX_COORDINATE, "description": "Target width, paired with height."},
            "height": {"type": "integer", "minimum": 1, "maximum": MAX_COORDINATE, "description": "Target height, paired with width."},
            "target": target_object(),
            "button": {
                "type": "string",
                "enum": ["left", "middle", "right"],
                "description": "Click/drag button (default left)."
            },
            "clicks": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_CLICKS,
                "description": "Click count, 1-3 (default 1)."
            },
            "from": target_object(),
            "to": target_object(),
            "direction": {
                "type": "string",
                "enum": ["up", "down"],
                "description": "Scroll direction."
            },
            "amount": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_SCROLL_NOTCHES,
                "description": "Scroll notches (default 3)."
            },
            "text": {
                "type": "string",
                "description": "Text to type at the current focus (max 500 characters)."
            },
            "key": {
                "type": "string",
                "description": "Single character, F1-F20, or enter/escape/tab/space/backspace/delete/up/down/left/right/home/end/pageup/pagedown/ctrl/shift/alt/meta."
            },
            "keys": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": MAX_HOTKEY_KEYS,
                "description": "Hotkey modifiers CTRL/SHIFT/ALT/META plus one key, with META meaning Command/Super/Windows."
            },
            "actions": {
                "type": "array",
                "items": {"type": "object"},
                "minItems": 1,
                "maxItems": MAX_SEQUENCE_ACTIONS,
                "description": "Sequence of 1-16 actions using normal call fields, without nested sequences."
            }
        },
        "required": ["op"],
        "additionalProperties": false
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request(value: Value) -> Result<Action, String> {
        serde_json::from_value::<InputRequest>(value)
            .map_err(|error| error.to_string())?
            .validate()
    }

    #[test]
    fn inline_coordinates_and_explicit_targets_are_equivalent() {
        let inline = request(json!({"op": "move", "x": 1240, "y": 680})).unwrap();
        let explicit = request(json!({"op": "move", "target": {"x": 1240, "y": 680}})).unwrap();
        assert_eq!(inline, explicit);
        assert_eq!(
            inline.kind,
            ActionKind::Move {
                target: Target {
                    x: 1240,
                    y: 680,
                    width: None,
                    height: None
                }
            }
        );
    }

    #[test]
    fn rectangles_are_targets_the_runtime_owns() {
        let action = request(json!({
            "op": "click",
            "target": {"x": 820, "y": 640, "width": 120, "height": 36}
        }))
        .unwrap();
        match action.kind {
            ActionKind::Click {
                target,
                button,
                count,
            } => {
                assert_eq!(target.area(), Some((820, 640, 120, 36)));
                assert_eq!(button, MouseButton::Left);
                assert_eq!(count, 1);
            }
            other => panic!("expected a click, got {other:?}"),
        }
    }

    #[test]
    fn double_click_is_one_burst_of_two() {
        let action = request(json!({"op": "double_click", "x": 600, "y": 400})).unwrap();
        match action.kind {
            ActionKind::Click { count, .. } => assert_eq!(count, 2),
            other => panic!("expected a click, got {other:?}"),
        }
        let error =
            request(json!({"op": "double_click", "x": 1, "y": 2, "clicks": 3})).unwrap_err();
        assert!(error.contains("clicks"), "{error}");
    }

    #[test]
    fn unknown_operations_fields_and_types_are_rejected_during_parsing() {
        let error = serde_json::from_value::<InputRequest>(json!({"op": "swipe", "x": 1, "y": 2}))
            .unwrap_err()
            .to_string();
        assert!(error.contains("unknown variant"), "{error}");

        let error = serde_json::from_value::<InputRequest>(json!({"op": "move", "angle": 30}))
            .unwrap_err()
            .to_string();
        assert!(error.contains("unknown field"), "{error}");

        let error =
            serde_json::from_value::<InputRequest>(json!({"op": "move", "x": "left", "y": 2}))
                .unwrap_err()
                .to_string();
        assert!(error.contains("invalid type"), "{error}");

        let error = serde_json::from_value::<InputRequest>(json!({}))
            .unwrap_err()
            .to_string();
        assert!(error.contains("missing field"), "{error}");
    }

    #[test]
    fn fields_of_other_operations_are_refused() {
        let error = request(json!({"op": "move", "x": 1, "y": 2, "text": "hello"})).unwrap_err();
        assert!(error.contains("does not accept `text`"), "{error}");
        assert!(error.contains("move"), "{error}");

        let error = request(json!({"op": "type", "text": "hi", "button": "left"})).unwrap_err();
        assert!(error.contains("does not accept `button`"), "{error}");
    }

    #[test]
    fn pointer_operations_need_usable_geometry() {
        let error = request(json!({"op": "click"})).unwrap_err();
        assert!(error.contains("target"), "{error}");

        let error = request(json!({"op": "click", "x": 10})).unwrap_err();
        assert!(error.contains("`x` and `y`"), "{error}");

        let error =
            request(json!({"op": "move", "x": 1, "y": 2, "target": {"x": 3, "y": 4}})).unwrap_err();
        assert!(error.contains("not both"), "{error}");

        let error = request(json!({"op": "move", "x": 1, "y": 2, "width": 40})).unwrap_err();
        assert!(error.contains("width"), "{error}");

        let error = request(json!({
            "op": "click",
            "target": {"x": 1, "y": 2, "width": 0, "height": 10}
        }))
        .unwrap_err();
        assert!(error.contains("positive"), "{error}");

        let error = request(json!({"op": "move", "x": 999_999, "y": 0})).unwrap_err();
        assert!(error.contains("32000"), "{error}");
    }

    #[test]
    fn drag_needs_both_ends() {
        let action = request(json!({
            "op": "drag",
            "from": {"x": 400, "y": 300},
            "to": {"x": 900, "y": 500}
        }))
        .unwrap();
        match action.kind {
            ActionKind::Drag { from, to, button } => {
                assert_eq!((from.x, from.y), (400, 300));
                assert_eq!((to.x, to.y), (900, 500));
                assert_eq!(button, MouseButton::Left);
            }
            other => panic!("expected a drag, got {other:?}"),
        }
        let error = request(json!({"op": "drag", "from": {"x": 1, "y": 2}})).unwrap_err();
        assert!(error.contains("`to`"), "{error}");
    }

    #[test]
    fn scroll_carries_a_direction_and_a_bounded_amount() {
        let action = request(json!({"op": "scroll", "direction": "down", "amount": 5})).unwrap();
        match action.kind {
            ActionKind::Scroll {
                direction,
                amount,
                target,
            } => {
                assert_eq!(direction, ScrollDirection::Down);
                assert_eq!(amount, 5);
                assert!(target.is_none());
            }
            other => panic!("expected a scroll, got {other:?}"),
        }
        assert!(
            request(json!({"op": "scroll"}))
                .unwrap_err()
                .contains("direction")
        );
        assert!(request(json!({"op": "scroll", "direction": "sideways"})).is_err());
        assert!(
            request(json!({"op": "scroll", "direction": "up", "amount": 99}))
                .unwrap_err()
                .contains("50")
        );
    }

    #[test]
    fn typing_is_bounded_and_key_names_are_normalized() {
        assert!(
            request(json!({"op": "type", "text": ""}))
                .unwrap_err()
                .contains("non-empty")
        );
        let long = "a".repeat(MAX_TEXT_CHARS + 1);
        assert!(
            request(json!({"op": "type", "text": long}))
                .unwrap_err()
                .contains("500")
        );
        assert!(request(json!({"op": "type", "text": "hi"})).is_ok());

        let action = request(json!({"op": "key", "key": "Page-Down"})).unwrap();
        assert_eq!(action.kind, ActionKind::Key(Key::PageDown));
        assert_eq!(parse_key("CTRL").unwrap(), Key::Control);
        assert_eq!(parse_key("cmd").unwrap(), Key::Meta);
        assert_eq!(parse_key("f12").unwrap(), Key::F12);
        assert_eq!(parse_key("a").unwrap(), Key::Unicode('a'));
        assert!(parse_key("hyperdrive").unwrap_err().contains("unknown key"));
    }

    #[test]
    fn hotkeys_need_a_real_key_besides_the_modifiers() {
        let action = request(json!({"op": "hotkey", "keys": ["CTRL", "L"]})).unwrap();
        assert_eq!(
            action.kind,
            ActionKind::Hotkey(vec![Key::Control, Key::Unicode('L')])
        );

        let error = request(json!({"op": "hotkey", "keys": ["CTRL", "SHIFT"]})).unwrap_err();
        assert!(error.contains("non-modifier"), "{error}");
        assert!(
            request(json!({"op": "hotkey", "keys": []}))
                .unwrap_err()
                .contains("at least one")
        );
        let many = json!(["CTRL", "SHIFT", "ALT", "META", "A", "B"]);
        assert!(
            request(json!({"op": "hotkey", "keys": many}))
                .unwrap_err()
                .contains('5')
        );
    }

    #[test]
    fn sequences_inherit_behavior_and_refuse_nesting() {
        let action = request(json!({
            "op": "sequence",
            "behavior": "precise",
            "actions": [
                {"op": "click", "target": {"x": 800, "y": 600, "width": 200, "height": 40}},
                {"op": "type", "text": "hello@example.com"},
                {"op": "key", "key": "Enter", "behavior": "fast"}
            ]
        }))
        .unwrap();
        let ActionKind::Sequence(actions) = action.kind else {
            panic!("expected a sequence");
        };
        assert_eq!(actions.len(), 3);
        assert_eq!(actions[0].behavior, Behavior::Precise);
        assert_eq!(actions[1].behavior, Behavior::Precise);
        assert_eq!(actions[2].behavior, Behavior::Fast);

        let error = request(json!({
            "op": "sequence",
            "actions": [{"op": "sequence", "actions": [{"op": "key", "key": "enter"}]}]
        }))
        .unwrap_err();
        assert!(error.contains("nested sequence"), "{error}");

        assert!(
            request(json!({"op": "sequence", "actions": [], "text": "hi"}))
                .unwrap_err()
                .contains("actions")
        );
    }

    #[test]
    fn calls_that_would_block_the_loop_for_a_minute_are_refused() {
        let text = "x".repeat(MAX_TEXT_CHARS);
        let error =
            request(json!({"op": "type", "text": text, "behavior": "precise"})).unwrap_err();
        assert!(error.contains("limit"), "{error}");
        assert!(request(json!({"op": "type", "text": "x".repeat(MAX_TEXT_CHARS)})).is_ok());
    }

    #[test]
    fn the_schema_lists_every_operation_and_forbids_extra_fields() {
        let schema = tool_schema();
        assert_eq!(schema["required"], json!(["op"]));
        assert_eq!(schema["additionalProperties"], json!(false));
        let operations = schema["properties"]["op"]["enum"].as_array().unwrap();
        assert_eq!(operations.len(), Op::all().len());
        for op in Op::all() {
            assert!(
                operations.contains(&json!(op.label())),
                "{} missing",
                op.label()
            );
        }
    }

    #[test]
    fn key_labels_round_trip_through_the_parser() {
        for key in [
            Key::Return,
            Key::Escape,
            Key::Tab,
            Key::Space,
            Key::Backspace,
            Key::Delete,
            Key::UpArrow,
            Key::DownArrow,
            Key::LeftArrow,
            Key::RightArrow,
            Key::Home,
            Key::End,
            Key::PageUp,
            Key::PageDown,
            Key::F1,
            Key::F20,
            Key::Control,
            Key::Shift,
            Key::Alt,
            Key::Meta,
            Key::Unicode('a'),
            Key::Unicode('5'),
        ] {
            assert_eq!(parse_key(&key_label(key)).unwrap(), key, "{key:?}");
        }
    }
}
