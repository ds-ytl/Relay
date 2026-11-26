//! OpenAI Responses API adapter, plus a scriptable model that runs offline.
//!
//! All three agents use Function Calling: the understanding turn explores and then calls
//! `submit_plan`, the execution turn does the work and then calls `submit_result`, the verification
//! turn checks and then calls `submit_verification`; the legacy Markdown text protocol is only a
//! fallback.
//!
//! Compatibility note (verified against DeepSeek /v1/responses): with `store=false` the server
//! cannot continue tool history via `previous_response_id`, so we use **stateless replay**: the
//! `reasoning` and `function_call` items from the previous output are sent back verbatim as input,
//! and a `function_call_output` is added for every `call_id`.

use std::path::PathBuf;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod asr;
pub mod client;
pub mod mock;
pub mod web_tools;

pub use asr::{wav_bytes, AsrClient};
pub use client::{OpenAiClient, OpenAiConfig};
pub use mock::{MockModel, MockStep};
pub use web_tools::{FetchedPage, SearchRequest, SearchResult, WebTools};

/// Streamed delta: the model pushes output to the caller while generating, so the UI can render
/// character by character.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamDelta {
    /// Visible answer (Markdown / plain text).
    Text(String),
    /// Thinking process (reasoning summary / reasoning text).
    Reasoning(String),
}

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("no model API key configured: set RELAY_OPENAI_API_KEY or OPENAI_API_KEY")]
    MissingApiKey,
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("model returned an error ({status}): {body}")]
    Api { status: u16, body: String },
    #[error("failed to parse model response: {0}")]
    Parse(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// The output budget ran out before the model produced any text or tool call, typically
    /// because a thinking model spent it all on reasoning. A retry with a larger
    /// `max_output_tokens` can recover, unlike an oversized request.
    #[error("model output was truncated ({reason}) and produced no text or tool call; raise max_output_tokens and retry")]
    Truncated { reason: String },
    #[error("{0}")]
    Other(String),
}

impl ModelError {
    /// True when the model used up its output budget and returned nothing usable.
    pub fn is_output_truncated(&self) -> bool {
        matches!(self, ModelError::Truncated { .. })
    }
}

pub type Result<T, E = ModelError> = std::result::Result<T, E>;

/// Tool definition exposed to the model (the `tools` of the Responses API).
#[derive(Debug, Clone)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    /// JSON Schema fragment: `{"type":"object","properties":{...},"required":[...]}`.
    pub parameters: serde_json::Value,
}

impl ToolDef {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }
}

/// One tool call the model requested during a completion.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub call_id: String,
    pub name: String,
    /// Parsed arguments; degrades to an empty object when the model sends invalid JSON.
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct CompletionRequest {
    /// Role description (the system prompt of Router / Understand / Executor / Verifier).
    pub instructions: String,
    pub input: Vec<InputItem>,
    pub model: Option<String>,
    pub temperature: Option<f32>,
    pub max_output_tokens: u32,
    /// Reuse the previous response within the same session to avoid repeating context (has no effect
    /// when the server does not store responses).
    pub previous_response_id: Option<String>,
    /// Empty means this turn needs no tools (plain text output).
    pub tools: Vec<ToolDef>,
}

impl CompletionRequest {
    pub fn text(instructions: impl Into<String>, input: impl Into<String>) -> Self {
        Self {
            instructions: instructions.into(),
            input: vec![InputItem::Text(input.into())],
            model: None,
            temperature: Some(0.2),
            max_output_tokens: 4096,
            previous_response_id: None,
            tools: Vec::new(),
        }
    }

    /// Attach tools to this turn.
    pub fn with_tools(mut self, tools: Vec<ToolDef>) -> Self {
        self.tools = tools;
        self
    }
}

#[derive(Debug, Clone)]
pub enum InputItem {
    Text(String),
    /// Screenshot or attachment: inlined as a data URL by MIME type.
    Image {
        path: PathBuf,
        mime: String,
    },
    /// Prepared image bytes (resized / re-encoded), inlined directly as a data URL.
    ImageBytes {
        mime: String,
        bytes: Vec<u8>,
    },
    /// Replay a previous `reasoning` / `function_call` item verbatim, or submit a
    /// `function_call_output`.
    Raw(serde_json::Value),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelOutput {
    pub text: String,
    pub response_id: Option<String>,
    pub usage: Option<Usage>,
    /// Tools the model asked to call this turn; empty means it is ready to wrap up.
    #[serde(skip)]
    pub tool_calls: Vec<ToolCall>,
    /// Raw items to replay on the next turn (`reasoning` / `function_call`), exactly as the server
    /// sent them.
    #[serde(skip)]
    pub replay_items: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    /// Input tokens that hit the prefix cache (`input_tokens_details.cached_tokens`); 0 when the
    /// server does not report them.
    pub cached_tokens: u32,
}

#[async_trait]
pub trait LanguageModel: Send + Sync {
    async fn complete(&self, request: CompletionRequest) -> Result<ModelOutput>;

    /// Streaming completion: deltas are written to `deltas` as they arrive, and the return value
    /// matches the non-streaming `complete`.
    ///
    /// The default implementation degrades to one full delivery: models without SSE support still
    /// work, and the UI receives a single complete delta instead of character-by-character output.
    async fn complete_stream(
        &self,
        request: CompletionRequest,
        deltas: tokio::sync::mpsc::UnboundedSender<StreamDelta>,
    ) -> Result<ModelOutput> {
        let output = self.complete(request).await?;
        if !output.text.is_empty() {
            let _ = deltas.send(StreamDelta::Text(output.text.clone()));
        }
        Ok(output)
    }

    fn name(&self) -> &str;

    /// Scope historical model context to the configured provider.
    fn cache_namespace(&self) -> &str {
        self.name()
    }
}

#[async_trait]
impl<T: LanguageModel + ?Sized> LanguageModel for std::sync::Arc<T> {
    async fn complete(&self, request: CompletionRequest) -> Result<ModelOutput> {
        (**self).complete(request).await
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
        deltas: tokio::sync::mpsc::UnboundedSender<StreamDelta>,
    ) -> Result<ModelOutput> {
        (**self).complete_stream(request, deltas).await
    }

    fn cache_namespace(&self) -> &str {
        (**self).cache_namespace()
    }

    fn name(&self) -> &str {
        (**self).name()
    }
}
