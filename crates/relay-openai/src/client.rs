use std::time::Duration;

use base64::Engine;
use reqwest::Client;
use serde_json::{json, Value};

use tokio::sync::mpsc::UnboundedSender;

use crate::{
    CompletionRequest, InputItem, LanguageModel, ModelError, ModelOutput, Result, StreamDelta,
    ToolCall, Usage,
};

#[derive(Debug, Clone)]
pub struct OpenAiConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub timeout: Duration,
    pub max_retries: u32,
}

impl OpenAiConfig {
    /// Read the key from the environment: RELAY_OPENAI_API_KEY wins, OPENAI_API_KEY is also accepted.
    pub fn api_key_from_env() -> Option<String> {
        std::env::var("RELAY_OPENAI_API_KEY")
            .or_else(|_| std::env::var("OPENAI_API_KEY"))
            .ok()
            .map(|key| key.trim().to_string())
            .filter(|key| !key.is_empty())
    }

    /// Build a config from an explicit key; the other fields still come from the environment, and
    /// OpenAI defaults apply when they are not set.
    pub fn with_key(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: std::env::var("RELAY_OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".into()),
            model: std::env::var("RELAY_OPENAI_MODEL").unwrap_or_else(|_| "gpt-5".into()),
            timeout: Duration::from_secs(180),
            max_retries: 2,
        }
    }

    /// Read the config from the environment; returns None without a key (the runtime then falls back
    /// to the mock or asks for configuration).
    pub fn from_env() -> Option<Self> {
        Some(Self::with_key(Self::api_key_from_env()?))
    }
}

#[derive(Debug, Clone)]
pub struct OpenAiClient {
    http: Client,
    config: OpenAiConfig,
}

impl OpenAiClient {
    pub fn new(config: OpenAiConfig) -> Result<Self> {
        let http = Client::builder().timeout(config.timeout).build()?;
        Ok(Self { http, config })
    }

    pub fn model_name(&self) -> &str {
        &self.config.model
    }

    /// Streaming request body: `stream=true`, with every other field identical to the non-streaming
    /// call.
    fn stream_payload(&self, request: &CompletionRequest) -> Result<Value> {
        let mut body = self.payload(request)?;
        body["stream"] = json!(true);
        Ok(body)
    }

    fn payload(&self, request: &CompletionRequest) -> Result<Value> {
        request.responses_payload(&self.config.model)
    }
}

impl CompletionRequest {
    /// The wire representation used both for durable snapshots and for the HTTP request.
    /// Raw items form boundaries between user messages so an archived input can be extended.
    pub fn responses_payload(&self, default_model: &str) -> Result<Value> {
        let request = self;
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| default_model.to_owned());
        let mut input = Vec::new();
        let mut content = Vec::new();
        for item in &request.input {
            match item {
                InputItem::Text(text) => content.push(json!({
                    "type": "input_text",
                    "text": text,
                })),
                InputItem::Image { path, mime } => {
                    let bytes = std::fs::read(path)?;
                    let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
                    content.push(json!({
                        "type": "input_image",
                        "image_url": format!("data:{mime};base64,{encoded}"),
                    }));
                }
                InputItem::ImageBytes { mime, bytes } => {
                    let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
                    content.push(json!({
                        "type": "input_image",
                        "image_url": format!("data:{mime};base64,{encoded}"),
                    }));
                }
                InputItem::Raw(value) => {
                    if !content.is_empty() {
                        input.push(json!({"role":"user","content":std::mem::take(&mut content)}));
                    }
                    input.push(value.clone());
                }
            }
        }
        if !content.is_empty() || input.is_empty() {
            input.push(json!({"role":"user","content":content}));
        }

        let mut body = json!({
            "model": model,
            "instructions": request.instructions,
            "input": input,
            "max_output_tokens": request.max_output_tokens,
        });
        if !request.tools.is_empty() {
            let tools: Vec<Value> = request
                .tools
                .iter()
                .map(|tool| {
                    json!({
                        "type": "function",
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    })
                })
                .collect();
            body["tools"] = json!(tools);
        }
        if let Some(temperature) = request.temperature {
            body["temperature"] = json!(temperature);
        }
        if let Some(previous) = &request.previous_response_id {
            body["previous_response_id"] = json!(previous);
        }
        Ok(body)
    }
}

#[async_trait::async_trait]
impl LanguageModel for OpenAiClient {
    async fn complete(&self, request: CompletionRequest) -> Result<ModelOutput> {
        let body = self.payload(&request)?;
        let url = format!("{}/responses", self.config.base_url.trim_end_matches('/'));
        let mut attempt = 0;

        loop {
            let response = self
                .http
                .post(&url)
                .bearer_auth(&self.config.api_key)
                .json(&body)
                .send()
                .await;

            match response {
                Ok(resp) => {
                    let status = resp.status();
                    let text = resp.text().await?;
                    if status.is_success() {
                        return parse_response(&text);
                    }
                    let retryable = status.as_u16() == 429 || status.is_server_error();
                    if retryable && attempt < self.config.max_retries {
                        attempt += 1;
                        let backoff = Duration::from_millis(400 * 2u64.pow(attempt));
                        tracing::warn!(
                            status = status.as_u16(),
                            "model request failed, retrying in {}ms",
                            backoff.as_millis()
                        );
                        tokio::time::sleep(backoff).await;
                        continue;
                    }
                    return Err(ModelError::Api {
                        status: status.as_u16(),
                        body: truncate(&text, 2000),
                    });
                }
                Err(err) => {
                    let retryable = err.is_timeout() || err.is_connect();
                    if retryable && attempt < self.config.max_retries {
                        attempt += 1;
                        let backoff = Duration::from_millis(400 * 2u64.pow(attempt));
                        tracing::warn!(%err, "model network error, retrying in {}ms", backoff.as_millis());
                        tokio::time::sleep(backoff).await;
                        continue;
                    }
                    return Err(ModelError::Http(err));
                }
            }
        }
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
        deltas: UnboundedSender<StreamDelta>,
    ) -> Result<ModelOutput> {
        let body = self.stream_payload(&request)?;
        let url = format!("{}/responses", self.config.base_url.trim_end_matches('/'));
        let mut attempt = 0;

        loop {
            let response = self
                .http
                .post(&url)
                .bearer_auth(&self.config.api_key)
                .json(&body)
                .send()
                .await;

            match response {
                Ok(mut resp) => {
                    let status = resp.status();
                    if !status.is_success() {
                        let text = resp.text().await?;
                        let retryable = status.as_u16() == 429 || status.is_server_error();
                        if retryable && attempt < self.config.max_retries {
                            attempt += 1;
                            let backoff = Duration::from_millis(400 * 2u64.pow(attempt));
                            tracing::warn!(
                                status = status.as_u16(),
                                "streaming model request failed, retrying in {}ms",
                                backoff.as_millis()
                            );
                            tokio::time::sleep(backoff).await;
                            continue;
                        }
                        // Some compatible endpoints do not support stream=true (they answer 4xx
                        // right away): fall back to a single normal request so the task can still
                        // finish.
                        if matches!(status.as_u16(), 400 | 404 | 405 | 422) {
                            let output = self.complete(request).await?;
                            if !output.text.is_empty() {
                                let _ = deltas.send(StreamDelta::Text(output.text.clone()));
                            }
                            return Ok(output);
                        }
                        return Err(ModelError::Api {
                            status: status.as_u16(),
                            body: truncate(&text, 2000),
                        });
                    }

                    // Tolerate endpoints without SSE: parse the body as a plain JSON response.
                    let is_sse = resp
                        .headers()
                        .get(reqwest::header::CONTENT_TYPE)
                        .and_then(|value| value.to_str().ok())
                        .map(|value| value.contains("text/event-stream"))
                        .unwrap_or(false);
                    if !is_sse {
                        let text = resp.text().await?;
                        let output = parse_response(&text)?;
                        if !output.text.is_empty() {
                            let _ = deltas.send(StreamDelta::Text(output.text.clone()));
                        }
                        return Ok(output);
                    }
                    return consume_sse(&mut resp, &deltas).await;
                }
                Err(err) => {
                    if attempt < self.config.max_retries {
                        attempt += 1;
                        let backoff = Duration::from_millis(400 * 2u64.pow(attempt));
                        tracing::warn!(%err, "streaming model network error, retrying in {}ms", backoff.as_millis());
                        tokio::time::sleep(backoff).await;
                        continue;
                    }
                    return Err(ModelError::Http(err));
                }
            }
        }
    }

    fn cache_namespace(&self) -> &str {
        &self.config.base_url
    }

    fn name(&self) -> &str {
        &self.config.model
    }
}

/// Consume the Responses API SSE stream: forward deltas while accumulating the final output.
async fn consume_sse(
    resp: &mut reqwest::Response,
    deltas: &UnboundedSender<StreamDelta>,
) -> Result<ModelOutput> {
    let mut buffer = String::new();
    let mut text = String::new();
    let mut completed: Option<ModelOutput> = None;
    let mut failure: Option<String> = None;
    let mut truncated: Option<String> = None;

    while let Some(chunk) = resp.chunk().await? {
        let chunk = String::from_utf8_lossy(&chunk).replace("\r\n", "\n");
        buffer.push_str(&chunk);
        while let Some(index) = buffer.find("\n\n") {
            let block: String = buffer.drain(..index + 2).collect();
            handle_sse_block(
                &block,
                &mut text,
                deltas,
                &mut completed,
                &mut failure,
                &mut truncated,
            );
        }
    }
    if !buffer.trim().is_empty() {
        handle_sse_block(
            &buffer,
            &mut text,
            deltas,
            &mut completed,
            &mut failure,
            &mut truncated,
        );
    }

    if let Some(error) = failure {
        return Err(ModelError::Other(error));
    }
    if let Some(reason) = truncated {
        return Err(ModelError::Truncated { reason });
    }
    if let Some(output) = completed {
        return Ok(output);
    }
    if text.trim().is_empty() {
        return Err(ModelError::Parse(
            "the streaming response produced no output".into(),
        ));
    }
    Ok(ModelOutput {
        text,
        ..Default::default()
    })
}

fn handle_sse_block(
    block: &str,
    text: &mut String,
    deltas: &UnboundedSender<StreamDelta>,
    completed: &mut Option<ModelOutput>,
    failure: &mut Option<String>,
    truncated: &mut Option<String>,
) {
    let data = block
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    if data.is_empty() || data == "[DONE]" {
        return;
    }
    let Ok(value) = serde_json::from_str::<Value>(&data) else {
        return;
    };
    match value
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default()
    {
        "response.output_text.delta" => {
            if let Some(delta) = value.get("delta").and_then(Value::as_str) {
                text.push_str(delta);
                let _ = deltas.send(StreamDelta::Text(delta.to_string()));
            }
        }
        "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
            if let Some(delta) = value.get("delta").and_then(Value::as_str) {
                let _ = deltas.send(StreamDelta::Reasoning(delta.to_string()));
            }
        }
        "response.completed" | "response.incomplete" => {
            if let Some(response) = value.get("response") {
                let failed = response
                    .get("status")
                    .and_then(Value::as_str)
                    .map(|status| status == "failed")
                    .unwrap_or(false);
                if failed {
                    *failure = Some(sse_error_message(response));
                } else {
                    match parse_response(&response.to_string()) {
                        Ok(output) => *completed = Some(output),
                        // An incomplete response that yields nothing usable: keep the reason so
                        // the caller can retry with a larger output budget.
                        Err(ModelError::Truncated { reason }) => *truncated = Some(reason),
                        Err(_) => {}
                    }
                }
            }
        }
        "response.failed" | "error" => {
            *failure = Some(sse_error_message(&value));
        }
        _ => {}
    }
}

fn sse_error_message(value: &Value) -> String {
    value
        .get("error")
        .and_then(|error| error.get("message"))
        .or_else(|| value.get("message"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| truncate(&value.to_string(), 500))
}

fn parse_response(body: &str) -> Result<ModelOutput> {
    let value: Value = serde_json::from_str(body)
        .map_err(|e| ModelError::Parse(format!("{e}: {}", truncate(body, 400))))?;

    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut replay_items = Vec::new();
    if let Some(output) = value.get("output").and_then(Value::as_array) {
        for item in output {
            let kind = item.get("type").and_then(Value::as_str).unwrap_or_default();
            if kind == "function_call" {
                let name = item.get("name").and_then(Value::as_str).unwrap_or_default();
                let call_id = item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let arguments = item
                    .get("arguments")
                    .and_then(Value::as_str)
                    .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
                    .unwrap_or_else(|| json!({}));
                if !name.is_empty() && !call_id.is_empty() {
                    tool_calls.push(ToolCall {
                        call_id: call_id.to_string(),
                        name: name.to_string(),
                        arguments,
                    });
                    // The server requires function_call to be replayed verbatim on the next turn.
                    replay_items.push(item.clone());
                }
                continue;
            }
            if kind == "reasoning" {
                // In thinking mode reasoning_text must be sent back verbatim, otherwise the request is rejected.
                replay_items.push(item.clone());
                continue;
            }
            if kind != "message" {
                continue;
            }
            if item["role"] == "assistant" {
                replay_items.push(item.clone());
            }
            if let Some(parts) = item.get("content").and_then(Value::as_array) {
                for part in parts {
                    if part.get("type").and_then(Value::as_str) == Some("output_text") {
                        if let Some(chunk) = part.get("text").and_then(Value::as_str) {
                            text.push_str(chunk);
                        }
                    }
                }
            }
        }
    }
    if text.is_empty() {
        // Some compatible endpoints return output_text directly.
        if let Some(direct) = value.get("output_text").and_then(Value::as_str) {
            text.push_str(direct);
        }
    }
    if text.trim().is_empty() && tool_calls.is_empty() {
        // A thinking model may spend the whole budget on reasoning: say it was truncated instead of
        // reporting a parse failure.
        let incomplete = value
            .get("incomplete_details")
            .and_then(|details| details.get("reason"))
            .and_then(Value::as_str);
        if let Some(reason) = incomplete {
            return Err(ModelError::Truncated {
                reason: reason.to_string(),
            });
        }
        return Err(ModelError::Parse(format!(
            "response contains no output_text: {}",
            truncate(body, 400)
        )));
    }

    let usage = value.get("usage").map(|u| Usage {
        input_tokens: u.get("input_tokens").and_then(Value::as_u64).unwrap_or(0) as u32,
        output_tokens: u.get("output_tokens").and_then(Value::as_u64).unwrap_or(0) as u32,
        // The Responses API puts it in input_tokens_details.cached_tokens; some compatible
        // gateways expose prompt_cache_hit_tokens directly, so accept both.
        cached_tokens: u
            .get("input_tokens_details")
            .and_then(|details| details.get("cached_tokens"))
            .and_then(Value::as_u64)
            .or_else(|| u.get("prompt_cache_hit_tokens").and_then(Value::as_u64))
            .unwrap_or(0) as u32,
    });

    Ok(ModelOutput {
        text: text.trim().to_string(),
        response_id: value.get("id").and_then(Value::as_str).map(str::to_string),
        usage,
        tool_calls,
        replay_items,
    })
}

fn truncate(value: &str, max: usize) -> String {
    if value.chars().count() <= max {
        value.to_string()
    } else {
        value.chars().take(max).collect::<String>() + "…"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[test]
    fn parses_output_text_blocks() {
        let body = r#"{
            "id": "resp_1",
            "output": [
                {"type": "reasoning"},
                {"type": "message", "content": [
                    {"type": "output_text", "text": "Hello"},
                    {"type": "output_text", "text": " world"}
                ]}
            ],
            "usage": {"input_tokens": 12, "output_tokens": 34,
                      "input_tokens_details": {"cached_tokens": 7}}
        }"#;
        let parsed = parse_response(body).unwrap();
        assert_eq!(parsed.text, "Hello world");
        assert_eq!(parsed.response_id.as_deref(), Some("resp_1"));
        let usage = parsed.usage.unwrap();
        assert_eq!(usage.output_tokens, 34);
        // Prefix cache hit: the panel derives the hit rate from this.
        assert_eq!(usage.cached_tokens, 7);
    }

    #[test]
    fn reads_cache_hit_from_compatible_gateways() {
        let body = r#"{
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
            "usage": {"input_tokens": 10, "output_tokens": 1, "prompt_cache_hit_tokens": 6}
        }"#;
        let parsed = parse_response(body).unwrap();
        assert_eq!(parsed.usage.unwrap().cached_tokens, 6);
    }

    #[test]
    fn rejects_empty_output() {
        assert!(parse_response(r#"{"output":[]}"#).is_err());
    }

    #[test]
    fn reports_truncated_output_as_retryable() {
        // A thinking model spent the whole output budget on reasoning.
        let body = r#"{
            "output": [{"type": "reasoning", "summary": []}],
            "incomplete_details": {"reason": "max_output_tokens"}
        }"#;
        let err = parse_response(body).unwrap_err();
        assert!(err.is_output_truncated());
        assert!(err.to_string().contains("max_output_tokens"));
    }

    #[tokio::test]
    async fn streaming_incomplete_without_content_reports_truncation() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<StreamDelta>();
        let mut text = String::new();
        let mut completed = None;
        let mut failure = None;
        let mut truncated = None;
        let block = concat!(
            "event: response.incomplete\n",
            "data: {\"type\":\"response.incomplete\",\"response\":{\"status\":\"incomplete\",",
            "\"output\":[{\"type\":\"reasoning\"}],",
            "\"incomplete_details\":{\"reason\":\"max_output_tokens\"}}}\n\n"
        );
        handle_sse_block(
            block,
            &mut text,
            &tx,
            &mut completed,
            &mut failure,
            &mut truncated,
        );
        assert!(failure.is_none());
        assert!(completed.is_none());
        assert_eq!(truncated.as_deref(), Some("max_output_tokens"));
    }

    #[tokio::test]
    async fn sends_instructions_and_reads_response() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let captured: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
        let captured_clone = captured.clone();

        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buffer = vec![0u8; 64 * 1024];
            let read = socket.read(&mut buffer).await.unwrap();
            *captured_clone.lock().unwrap() = String::from_utf8_lossy(&buffer[..read]).to_string();
            let body = r#"{"id":"resp_x","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        });

        let client = OpenAiClient::new(OpenAiConfig {
            api_key: "test-key".into(),
            base_url: format!("http://{addr}"),
            model: "test-model".into(),
            timeout: Duration::from_secs(10),
            max_retries: 0,
        })
        .unwrap();

        let output = client
            .complete(CompletionRequest {
                instructions: "You are Relay".into(),
                input: vec![InputItem::Text("Hello".into())],
                model: None,
                temperature: None,
                max_output_tokens: 128,
                previous_response_id: None,
                tools: Vec::new(),
            })
            .await
            .unwrap();

        assert_eq!(output.text, "ok");
        let request = captured.lock().unwrap().clone();
        assert!(request.contains("You are Relay"), "{request}");
        assert!(request.contains("test-model"), "{request}");
        assert!(
            request.contains("authorization: Bearer test-key")
                || request.contains("Authorization: Bearer test-key"),
            "{request}"
        );
    }

    #[tokio::test]
    async fn streams_text_and_reasoning_deltas() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buffer = vec![0u8; 64 * 1024];
            let _ = socket.read(&mut buffer).await.unwrap();
            let body = concat!(
                "data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"listing files first\"}\n\n",
                "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hello\"}\n\n",
                "data: {\"type\":\"response.output_text.delta\",\"delta\":\", world\"}\n\n",
                "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_s\",\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hello, world\"}]}],\"usage\":{\"input_tokens\":3,\"output_tokens\":4}}}\n\n",
                "data: [DONE]\n\n",
            );
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n{body}"
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        });

        let client = OpenAiClient::new(OpenAiConfig {
            api_key: "test-key".into(),
            base_url: format!("http://{addr}"),
            model: "test-model".into(),
            timeout: Duration::from_secs(10),
            max_retries: 0,
        })
        .unwrap();

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let output = client
            .complete_stream(CompletionRequest::text("You are Relay", "Hello"), tx)
            .await
            .unwrap();

        assert_eq!(output.text, "Hello, world");
        assert_eq!(output.response_id.as_deref(), Some("resp_s"));
        assert_eq!(output.usage.unwrap().output_tokens, 4);

        let mut deltas = Vec::new();
        while let Ok(delta) = rx.try_recv() {
            deltas.push(delta);
        }
        assert_eq!(
            deltas,
            vec![
                StreamDelta::Reasoning("listing files first".into()),
                StreamDelta::Text("Hello".into()),
                StreamDelta::Text(", world".into()),
            ]
        );
    }
}
