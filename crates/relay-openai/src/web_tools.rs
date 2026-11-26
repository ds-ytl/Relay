//! Web tools API: page search (`/tools/search`) and page fetch (`/tools/fetch`).
//!
//! These endpoints sit next to the model API on the same base URL, so they reuse the configured
//! key and provider: `{base_url}/tools/search` and `{base_url}/tools/fetch`. Both are POST with a
//! JSON body and answer with structured JSON; failures and formatting are left to the tool layer,
//! which turns them into text for the model.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{ModelError, Result};

/// Request timeout: fetching a slow page can take a while, so allow the documented 60 seconds.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// Client for the search and fetch endpoints, sharing the model's API key and base URL.
#[derive(Clone)]
pub struct WebTools {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
}

impl std::fmt::Debug for WebTools {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never render the key in logs or debug output.
        f.debug_struct("WebTools")
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

impl WebTools {
    pub fn new(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        let http = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .unwrap_or_default();
        Self {
            http,
            base_url: base_url.into(),
            api_key: api_key.into(),
        }
    }

    /// Full URL of one tool, e.g. `https://api.deepseek.com/v1` + `/tools/search`.
    pub fn endpoint(&self, tool: &str) -> String {
        format!(
            "{}/tools/{}",
            self.base_url.trim_end_matches('/'),
            tool.trim_start_matches('/')
        )
    }

    /// Search the web; an empty vector means no matching results.
    pub async fn search(&self, request: &SearchRequest) -> Result<Vec<SearchResult>> {
        let payload =
            serde_json::to_value(request).map_err(|err| ModelError::Parse(err.to_string()))?;
        let body = self.post("search", &payload).await?;
        let parsed: SearchResponse = serde_json::from_str(&body)
            .map_err(|err| ModelError::Parse(format!("{err}: {}", clip(&body, 400))))?;
        Ok(parsed.results)
    }

    /// Fetch one page as Markdown.
    pub async fn fetch(&self, url: &str) -> Result<FetchedPage> {
        let body = self.post("fetch", &json!({ "url": url })).await?;
        serde_json::from_str(&body)
            .map_err(|err| ModelError::Parse(format!("{err}: {}", clip(&body, 400))))
    }

    /// POST one JSON body and return the raw response text (success status only).
    async fn post(&self, tool: &str, payload: &serde_json::Value) -> Result<String> {
        let response = self
            .http
            .post(self.endpoint(tool))
            .bearer_auth(&self.api_key)
            .json(payload)
            .send()
            .await?;
        let status = response.status();
        let body = response.text().await?;
        if !status.is_success() {
            return Err(ModelError::Api {
                status: status.as_u16(),
                body: clip(&body, 400),
            });
        }
        Ok(body)
    }
}

/// Search parameters. `ori_question` is the user's original question and only helps intent.
#[derive(Debug, Clone, Serialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ori_question: Option<String>,
    pub need_content: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SearchResponse {
    #[serde(default)]
    pub results: Vec<SearchResult>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SearchResult {
    #[serde(default)]
    pub rerank_score: f64,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub url: String,
    #[serde(default)]
    pub snippet: String,
    #[serde(default)]
    pub summary: String,
    /// Page body; empty unless `need_content` was requested.
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub site_name: Option<String>,
    #[serde(default)]
    pub date: Option<String>,
    /// 0-4, -1 when unknown; 3 and above is highly authoritative.
    #[serde(default)]
    pub authority: i32,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct FetchedPage {
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub content: String,
}

/// Keep error bodies short so a gateway HTML page cannot flood the context.
fn clip(value: &str, max: usize) -> String {
    if value.chars().count() <= max {
        return value.to_string();
    }
    value.chars().take(max).collect::<String>() + "…"
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use super::*;

    #[test]
    fn joins_endpoint_under_the_model_base_url() {
        let tools = WebTools::new("k", "https://api.deepseek.com/v1/");
        assert_eq!(
            tools.endpoint("search"),
            "https://api.deepseek.com/v1/tools/search"
        );
        assert_eq!(
            tools.endpoint("/fetch"),
            "https://api.deepseek.com/v1/tools/fetch"
        );
    }

    #[test]
    fn debug_output_never_contains_the_key() {
        let tools = WebTools::new("sk-secret", "https://api.deepseek.com/v1");
        let rendered = format!("{tools:?}");
        assert!(!rendered.contains("sk-secret"), "{rendered}");
    }

    #[test]
    fn parses_search_results() {
        let body = r#"{"results":[{"rerank_score":2.4,"title":"T","url":"https://e.com",
            "snippet":"s","summary":"sum","content":"","site_name":"E",
            "site_icon":null,"date":"2026-06-01T00:00:00+00:00","authority":3}]}"#;
        let parsed: SearchResponse = serde_json::from_str(body).unwrap();
        assert_eq!(parsed.results.len(), 1);
        assert_eq!(parsed.results[0].authority, 3);
        assert_eq!(
            parsed.results[0].date.as_deref(),
            Some("2026-06-01T00:00:00+00:00")
        );
    }

    #[tokio::test]
    async fn posts_query_and_reads_results() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let captured: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
        let captured_clone = captured.clone();

        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buffer = vec![0u8; 32 * 1024];
            let read = socket.read(&mut buffer).await.unwrap();
            *captured_clone.lock().unwrap() = String::from_utf8_lossy(&buffer[..read]).to_string();
            let body = r#"{"results":[{"rerank_score":2.0,"title":"T","url":"https://e.com",
                "snippet":"s","summary":"sum","content":"","site_name":null,
                "site_icon":null,"date":null,"authority":2}]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        });

        let tools = WebTools::new("test-key", format!("http://{addr}/v1"));
        let results = tools
            .search(&SearchRequest {
                query: "relay agent".into(),
                count: Some(3),
                ori_question: None,
                need_content: false,
            })
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "T");
        let request = captured.lock().unwrap().clone();
        assert!(request.starts_with("POST /v1/tools/search "), "{request}");
        assert!(request.contains("relay agent"), "{request}");
        assert!(
            request.contains("authorization: Bearer test-key")
                || request.contains("Authorization: Bearer test-key"),
            "{request}"
        );
    }

    #[tokio::test]
    async fn reports_api_errors_as_text() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buffer = vec![0u8; 8 * 1024];
            let _ = socket.read(&mut buffer).await.unwrap();
            let body = r#"{"error":{"message":"insufficient balance"}}"#;
            let response = format!(
                "HTTP/1.1 402 Payment Required\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        });

        let tools = WebTools::new("test-key", format!("http://{addr}/v1"));
        let err = tools.fetch("https://example.com").await.unwrap_err();
        let rendered = err.to_string();
        assert!(rendered.contains("402"), "{rendered}");
        assert!(rendered.contains("insufficient balance"), "{rendered}");
    }
}
