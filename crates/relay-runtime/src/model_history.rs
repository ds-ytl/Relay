//! Capture model inputs at dispatch, after compaction and output-budget selection.
use relay_core::TaskId;
use relay_openai::{
    CompletionRequest, InputItem, LanguageModel, ModelError, ModelOutput, StreamDelta,
};
use relay_storage::Store;
use sha2::{Digest, Sha256};
use std::sync::Arc;
use tokio::sync::mpsc::UnboundedSender;

tokio::task_local! { pub(crate) static CURRENT_TASK: TaskId; }

pub(crate) struct RecordedModel {
    model: Arc<dyn LanguageModel>,
    store: Arc<Store>,
    pub(crate) provider: String,
}

impl RecordedModel {
    pub fn new(model: Arc<dyn LanguageModel>, store: Arc<Store>) -> Self {
        // Provider URLs may contain private configuration. Only store a fingerprint.
        let provider = format!("{:x}", Sha256::digest(model.cache_namespace().as_bytes()));
        Self {
            model,
            store,
            provider,
        }
    }

    fn record(
        &self,
        task: TaskId,
        request: &mut CompletionRequest,
        stream: bool,
    ) -> Result<i64, ModelError> {
        // The screenshot on disk can change between recording and sending. Read it once.
        for item in &mut request.input {
            if let InputItem::Image { path, mime } = item {
                *item = InputItem::ImageBytes {
                    bytes: std::fs::read(path)?,
                    mime: mime.clone(),
                };
            }
        }
        let mut payload = request.responses_payload(self.model.name())?;
        if stream {
            payload["stream"] = true.into();
        }
        self.store
            .record_model_request(task, &self.provider, payload)
            .map_err(|err| ModelError::Other(format!("persist model context: {err}")))
    }

    pub async fn stream(
        &self,
        task: TaskId,
        mut request: CompletionRequest,
        deltas: UnboundedSender<StreamDelta>,
    ) -> Result<(i64, ModelOutput), ModelError> {
        let id = self.record(task, &mut request, true)?;
        Ok((id, self.model.complete_stream(request, deltas).await?))
    }
}

#[async_trait::async_trait]
impl LanguageModel for RecordedModel {
    async fn complete(&self, mut request: CompletionRequest) -> relay_openai::Result<ModelOutput> {
        let task = CURRENT_TASK
            .try_with(|id| *id)
            .map_err(|_| ModelError::Other("missing model request task".into()))?;
        self.record(task, &mut request, false)?;
        self.model.complete(request).await
    }

    fn name(&self) -> &str {
        self.model.name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use relay_openai::{OpenAiClient, OpenAiConfig, ToolDef};
    use serde_json::json;
    use std::time::Duration;
    use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn reopened_snapshot_matches_http_body_after_original_image_is_removed() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let received = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut socket = BufReader::new(socket);
            let mut length = None;
            loop {
                let mut line = String::new();
                assert!(socket.read_line(&mut line).await.unwrap() > 0);
                if line == "\r\n" {
                    break;
                }
                if let Some((k, v)) = line.split_once(':') {
                    if k.eq_ignore_ascii_case("content-length") {
                        length = Some(v.trim().parse::<usize>().unwrap());
                    }
                }
            }
            let mut body = vec![0; length.unwrap()];
            socket.read_exact(&mut body).await.unwrap();
            let response = r#"{"output_text":"received"}"#;
            socket.get_mut().write_all(format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{response}",response.len()
            ).as_bytes()).await.unwrap();
            body
        });
        let temp = tempfile::tempdir().unwrap();
        let image = temp.path().join("frame.png");
        std::fs::write(&image, b"original image bytes").unwrap();
        let db = temp.path().join("relay.sqlite3");
        let store = Arc::new(Store::open(&db).unwrap());
        let model = RecordedModel::new(
            Arc::new(
                OpenAiClient::new(OpenAiConfig {
                    base_url: format!("http://{address}"),
                    api_key: "test-only".into(),
                    model: "chosen-model".into(),
                    timeout: Duration::from_secs(5),
                    max_retries: 0,
                })
                .unwrap(),
            ),
            store.clone(),
        );
        let mut request =
            CompletionRequest::text("stable instructions", "look at this").with_tools(vec![
                ToolDef::new("inspect", "read", json!({"type":"object","properties":{}})),
            ]);
        request.input.push(InputItem::Image {
            path: image.clone(),
            mime: "image/png".into(),
        });
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let (id, _) = model.stream(TaskId::new(), request, tx).await.unwrap();
        let wire = received.await.unwrap();
        drop(model);
        drop(store);
        std::fs::remove_file(image).unwrap();
        let reopened = Store::open(&db).unwrap();
        assert_eq!(
            serde_json::to_vec(&reopened.model_request(id).unwrap()).unwrap(),
            wire
        );
    }
}
