//! Voice input for the panel: a [`SpeechRecognizer`] that transcribes through DeepSeek ASR.
//!
//! The endpoint is a batch call — one complete clip in, one transcript out — so audio is buffered
//! for the whole press and uploaded when the user lets go. Partial results never arrive, which is
//! why the panel shows a live level meter instead of streaming words.
//!
//! Audio capture, resampling to the 16 kHz mono the endpoint prefers and the microphone permission
//! all come from `gpui-component`'s speech layer; only the recognition itself lives here.

use std::time::Duration;

use gpui_kit::component::speech::{
    AudioFormat, RecognitionSession, SpeechError, SpeechRecognizer, SpeechSink,
};
use gpui_kit::{App, AsyncApp};
use relay_openai::{wav_bytes, AsrClient};

/// The endpoint accepts at most 120 seconds of audio; staying short of it keeps a long press well
/// inside the limit.
const MAX_SECONDS: u32 = 115;

/// How long the panel waits for the transcript after the user lets go. A clip of this length
/// recognises in seconds, so the ceiling only matters when the service is unreachable.
pub const TRANSCRIBE_TIMEOUT: Duration = Duration::from_secs(60);

/// A recognizer that uploads each press to `{base_url}/tools/asr`.
pub struct DeepSeekAsr {
    /// `None` when no key is configured: the endpoint reuses the model key, and the panel then
    /// shows the microphone as unavailable instead of failing after the first press.
    client: Option<AsrClient>,
    /// Requests run on the runtime the rest of Relay uses, not on the UI executor.
    tokio: tokio::runtime::Handle,
}

impl DeepSeekAsr {
    pub fn new(client: Option<AsrClient>, tokio: tokio::runtime::Handle) -> Self {
        Self { client, tokio }
    }
}

impl SpeechRecognizer for DeepSeekAsr {
    fn audio_format(&self) -> AudioFormat {
        AudioFormat::default()
    }

    fn is_available(&self, _cx: &App) -> bool {
        self.client.is_some()
    }

    fn start(
        &self,
        sink: SpeechSink,
        _cx: &mut App,
    ) -> Result<Box<dyn RecognitionSession>, SpeechError> {
        let Some(client) = self.client.clone() else {
            return Err(SpeechError::Unsupported);
        };
        Ok(Box::new(Press {
            sink,
            client,
            tokio: self.tokio.clone(),
            samples: Vec::new(),
        }))
    }
}

/// One press: audio accumulates here until the recognizer reports the user stopped talking.
struct Press {
    sink: SpeechSink,
    client: AsrClient,
    tokio: tokio::runtime::Handle,
    samples: Vec<i16>,
}

impl RecognitionSession for Press {
    fn push_audio(&mut self, samples: &[i16], _cx: &mut App) {
        let limit = MAX_SECONDS as usize * AudioFormat::default().sample_rate() as usize;
        let room = limit.saturating_sub(self.samples.len());
        if room == 0 {
            return;
        }
        self.samples
            .extend_from_slice(&samples[..samples.len().min(room)]);
    }

    fn finish(&mut self, cx: &mut App) {
        let samples = std::mem::take(&mut self.samples);
        let sink = self.sink.clone();
        let client = self.client.clone();
        let format = AudioFormat::default();
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.tokio.spawn(async move {
            let wav = wav_bytes(&samples, format.sample_rate(), format.channels());
            let _ = tx.send(client.transcribe(wav).await);
        });

        // Report on the UI thread: the sink drops results from a session that has already ended.
        cx.spawn(async move |cx: &mut AsyncApp| {
            let Ok(result) = rx.await else {
                cx.update(|cx| {
                    sink.error(
                        SpeechError::recognizer(anyhow::anyhow!("voice upload stopped")),
                        cx,
                    )
                });
                return;
            };
            cx.update(|cx| match result {
                Ok(text) => {
                    let text = text.trim();
                    if !text.is_empty() {
                        sink.phrase(text.to_string(), cx);
                    }
                    sink.finish(cx);
                }
                Err(error) => sink.error(
                    SpeechError::recognizer(anyhow::anyhow!(error.to_string())),
                    cx,
                ),
            });
        })
        .detach();
    }
}
