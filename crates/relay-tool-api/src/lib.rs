//! Minimal contract between tool implementations and the runtime.
//!
//! Migrated from the harness's `tool-runtime`, keeping only what `computer_input` /
//! `computer_accessibility` need: a model-visible tool definition, one strictly validated async
//! invocation, and [`ToolError`], which separates "bad arguments" from "execution failures".
//! The registry, approvals and event streams belong to relay's own tool layer and are not
//! duplicated here.

use std::{error::Error, fmt, future::Future, pin::Pin};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// Boxed future returned by a tool implementation.
pub type ToolFuture<'a> = Pin<Box<dyn Future<Output = Result<Value, ToolError>> + Send + 'a>>;

/// OpenAI-compatible function tool definition.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Value,
    pub strict: bool,
}

impl ToolDefinition {
    /// Create and validate a tool definition.
    ///
    /// The name must satisfy the model server's `^[a-zA-Z0-9_-]{1,64}$` and the parameters must be
    /// a JSON Schema object; anything else is rejected right here, so the whole request does not
    /// fail with a 400 at the model provider.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::InvalidDefinition`] when the name is empty, contains illegal
    /// characters, is too long, or when the parameters are not an object.
    pub fn new(
        name: impl Into<String>,
        description: Option<impl Into<String>>,
        parameters: Value,
    ) -> Result<Self, ToolError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(ToolError::InvalidDefinition(
                "tool name cannot be empty".into(),
            ));
        }
        if name.len() > 64
            || !name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            return Err(ToolError::InvalidDefinition(format!(
                "tool name {name:?} must match ^[a-zA-Z0-9_-]{{1,64}}$"
            )));
        }
        if !parameters.is_object() {
            return Err(ToolError::InvalidDefinition(format!(
                "tool {name:?} parameters must be a JSON Schema object"
            )));
        }
        Ok(Self {
            name,
            description: description.map(Into::into),
            parameters,
            strict: true,
        })
    }

    /// Turn strict off (used when the tool validates its own arguments).
    #[must_use]
    pub const fn with_strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Convert to a function tool object for the Responses API.
    #[must_use]
    pub fn as_responses_tool(&self) -> Value {
        json!({
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": self.strict,
        })
    }
}

/// Why a tool call failed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ToolError {
    /// The tool definition itself is broken; an integration-time error.
    InvalidDefinition(String),
    /// Arguments from the model are invalid: rejected before any side effect.
    InvalidArguments { name: String, message: String },
}

impl ToolError {
    /// Build an "invalid arguments" error.
    #[must_use]
    pub fn invalid_arguments(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self::InvalidArguments {
            name: name.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for ToolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDefinition(message) => {
                write!(formatter, "invalid tool definition: {message}")
            }
            Self::InvalidArguments { name, message } => {
                write!(formatter, "invalid arguments for tool {name:?}: {message}")
            }
        }
    }
}

impl Error for ToolError {}

/// A tool the model can call.
pub trait Tool: Send + Sync + 'static {
    /// The definition as seen by the model.
    fn definition(&self) -> &ToolDefinition;

    /// Run one invocation; returns [`ToolError::InvalidArguments`] before any side effect when the
    /// arguments are invalid.
    fn invoke(&self, arguments: Value) -> ToolFuture<'_>;
}
