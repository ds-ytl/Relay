use std::collections::VecDeque;
use std::sync::Mutex;

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::{CompletionRequest, InputItem, LanguageModel, ModelOutput, Result, ToolCall, Usage};

/// One scripted step: either return text directly or request tool calls.
#[derive(Debug, Clone)]
pub enum MockStep {
    Text(String),
    /// (tool name, argument JSON)
    ToolCalls(Vec<(String, Value)>),
}

impl MockStep {
    pub fn call(name: impl Into<String>, arguments: Value) -> Self {
        Self::ToolCalls(vec![(name.into(), arguments)])
    }
}

impl From<String> for MockStep {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for MockStep {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

/// Scripted model: returns preset steps in order, for offline demos and end-to-end tests.
pub struct MockModel {
    name: String,
    steps: Mutex<VecDeque<MockStep>>,
    requests: Mutex<Vec<CompletionRequest>>,
    calls: Mutex<u32>,
}

impl MockModel {
    pub fn new(responses: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self::scripted(responses.into_iter().map(|r| MockStep::Text(r.into())))
    }

    /// Scripted: text and tool calls may alternate, to rehearse "several tool turns -> submit plan".
    pub fn scripted(steps: impl IntoIterator<Item = MockStep>) -> Self {
        Self {
            name: "mock".into(),
            steps: Mutex::new(steps.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
            calls: Mutex::new(0),
        }
    }

    pub fn push(&self, response: impl Into<String>) {
        self.steps
            .lock()
            .unwrap()
            .push_back(MockStep::Text(response.into()));
    }

    pub fn push_step(&self, step: MockStep) {
        self.steps.lock().unwrap().push_back(step);
    }

    /// "Instructions + input" of each request; replayed raw items show up as compact JSON, which
    /// makes tool results easy to assert.
    pub fn requests(&self) -> Vec<String> {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .map(|r| {
                let mut text = r.instructions.clone();
                for item in &r.input {
                    text.push('\n');
                    match item {
                        InputItem::Text(t) => text.push_str(t),
                        InputItem::Image { path, .. } => {
                            text.push_str(&format!("[image:{}]", path.display()))
                        }
                        InputItem::ImageBytes { mime, bytes } => {
                            text.push_str(&format!("[image-bytes:{mime}:{}]", bytes.len()))
                        }
                        InputItem::Raw(value) => {
                            text.push_str(&serde_json::to_string(value).unwrap_or_default())
                        }
                    }
                }
                text
            })
            .collect()
    }

    /// Wire-shaped requests received by this offline model, in dispatch order.
    pub fn payloads(&self) -> Vec<Value> {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .map(|request| {
                request
                    .responses_payload(&self.name)
                    .expect("mock request payload")
            })
            .collect()
    }

    /// Tool names exposed per request, to assert that the understanding turn carries tools and the
    /// execution turn does not.
    pub fn tool_names(&self) -> Vec<Vec<String>> {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .map(|r| r.tools.iter().map(|t| t.name.clone()).collect())
            .collect()
    }

    /// Explicitly requested model name per request (None means the client default).
    pub fn models(&self) -> Vec<Option<String>> {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .map(|request| request.model.clone())
            .collect()
    }

    pub fn remaining(&self) -> usize {
        self.steps.lock().unwrap().len()
    }
}

/// Report usage like a real server: input is estimated heuristically, output is preset, and cache
/// hits are half of the input (a real server matches prefixes; here we only make sure the stats
/// pipeline has numbers).
fn mock_usage(prompt_chars: usize, output_tokens: u32) -> Usage {
    let input_tokens = 900 + (prompt_chars / 4) as u32;
    Usage {
        input_tokens,
        output_tokens,
        cached_tokens: input_tokens / 2,
    }
}

/// Fixed answer for summary requests: the structure matches what `prompts/compact.md` asks for.
fn mock_summary() -> ModelOutput {
    ModelOutput {
        text: "## Goal\nContinue the current task\n## Done\n- A few commands and checks\n## Key results\n- out.txt was generated\n## Failures and reasons\nNone\n## TODO\n- Wrap up and deliver".into(),
        response_id: Some("mock_summary".into()),
        usage: Some(Usage {
            input_tokens: 2_400,
            output_tokens: 160,
            cached_tokens: 1_800,
        }),
        tool_calls: Vec::new(),
        replay_items: Vec::new(),
    }
}

#[async_trait]
impl LanguageModel for MockModel {
    async fn complete(&self, request: CompletionRequest) -> Result<ModelOutput> {
        // Summary requests get a fixed answer: they consume no scripted step, so any number of
        // compactions leaves the orchestration untouched.
        let summary = request.instructions.contains("context compactor");
        self.requests.lock().unwrap().push(request);
        if summary {
            return Ok(mock_summary());
        }
        let step = self.steps.lock().unwrap().pop_front().unwrap_or_else(|| {
            MockStep::Text("VERDICT: PASS\nSUMMARY: mock script exhausted".into())
        });

        let mut calls = self.calls.lock().unwrap();
        match step {
            MockStep::Text(text) => Ok(ModelOutput {
                usage: Some(mock_usage(text.len(), 60)),
                text,
                response_id: Some(format!("mock_{}", *calls + 1)),
                tool_calls: Vec::new(),
                replay_items: Vec::new(),
            }),
            MockStep::ToolCalls(requested) => {
                let mut tool_calls = Vec::new();
                let mut replay_items = Vec::new();
                for (name, arguments) in requested {
                    *calls += 1;
                    let call_id = format!("mock_call_{}", *calls);
                    tool_calls.push(ToolCall {
                        call_id: call_id.clone(),
                        name: name.clone(),
                        arguments: arguments.clone(),
                    });
                    // Like a real server: function_call items must be replayed verbatim.
                    replay_items.push(json!({
                        "type": "function_call",
                        "call_id": call_id,
                        "name": name,
                        "arguments": arguments.to_string(),
                    }));
                }
                Ok(ModelOutput {
                    text: String::new(),
                    response_id: Some(format!("mock_{}", *calls)),
                    usage: Some(mock_usage(0, 240)),
                    tool_calls,
                    replay_items,
                })
            }
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}
