//! Speech recognition API: `{base_url}/tools/asr`.
//!
//! The endpoint takes a complete clip in one request (16-bit PCM WAV or Ogg Opus, at most 120
//! seconds and 8 MiB) and answers with the full transcript; the language is detected from the
//! audio. That makes it a batch call rather than a stream: the panel records a whole press and
//! uploads it once the user lets go.

use std::time::Duration;

use serde::Deserialize;

use crate::{ModelError, Result};

/// Upload and recognition of one clip. The documented upload window alone is 10 minutes, and a
/// 120-second clip recognises well inside a minute, so this is generous but bounded.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// What the endpoint answers with; only the transcript is used here.
#[derive(Debug, Deserialize)]
struct AsrResponse {
    text: String,
}

/// Client for the transcription endpoint, sharing the model's API key and base URL.
#[derive(Clone)]
pub struct AsrClient {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
}

impl std::fmt::Debug for AsrClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never render the key in logs or debug output.
        f.debug_struct("AsrClient")
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

impl AsrClient {
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

    /// Full URL of the endpoint, e.g. `https://api.deepseek.com/v1` + `/tools/asr`.
    pub fn endpoint(&self) -> String {
        format!("{}/tools/asr", self.base_url.trim_end_matches('/'))
    }

    /// Transcribe one clip of `wav` bytes; an empty clip is not sent.
    pub async fn transcribe(&self, wav: Vec<u8>) -> Result<String> {
        if wav.is_empty() {
            return Ok(String::new());
        }
        let part = reqwest::multipart::Part::bytes(wav)
            .file_name("speech.wav")
            .mime_str("audio/wav")
            .map_err(|err| ModelError::Other(err.to_string()))?;
        let form = reqwest::multipart::Form::new().part("file", part);
        let response = self
            .http
            .post(self.endpoint())
            .bearer_auth(&self.api_key)
            .multipart(form)
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
        let parsed: AsrResponse = serde_json::from_str(&body)
            .map_err(|err| ModelError::Parse(format!("{err}: {}", clip(&body, 400))))?;
        Ok(parsed.text)
    }
}

/// Wrap 16-bit PCM samples in a WAV container: the endpoint takes the audio as a file, and this is
/// the format it documents. Written by hand because a 44-byte header does not need a dependency.
pub fn wav_bytes(samples: &[i16], sample_rate: u32, channels: u16) -> Vec<u8> {
    let channels = channels.max(1);
    let block_align = channels * 2;
    let data_len = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * u32::from(block_align)).to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for sample in samples {
        out.extend_from_slice(&sample.to_le_bytes());
    }
    out
}

/// Keep error bodies readable in a one-line notice.
fn clip(body: &str, limit: usize) -> String {
    let trimmed = body.trim();
    if trimmed.chars().count() <= limit {
        return trimmed.to_string();
    }
    let cut: String = trimmed.chars().take(limit).collect();
    format!("{cut}…")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_sits_next_to_the_model_api() {
        let client = AsrClient::new("key", "https://api.deepseek.com/v1/");
        assert_eq!(client.endpoint(), "https://api.deepseek.com/v1/tools/asr");
    }

    #[test]
    fn wav_header_describes_the_samples() {
        let wav = wav_bytes(&[1, -1, 0], 16_000, 1);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(&wav[36..40], b"data");
        // Sample count, byte rate and block alignment for mono 16-bit 16 kHz.
        assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()), 6);
        assert_eq!(u32::from_le_bytes(wav[28..32].try_into().unwrap()), 32_000);
        assert_eq!(u16::from_le_bytes(wav[32..34].try_into().unwrap()), 2);
        assert_eq!(wav.len(), 44 + 6);
    }
}
