use relay_core::{ContextSource, EnvironmentSnapshot};

use crate::{finalize, CaptureOptions, ContextError, ContextProvider};

/// Manual context for when there is no system perception permission: injected from
/// environment variables, handy for CI and tests.
#[derive(Debug, Clone, Default)]
pub struct ManualContextProvider {
    pub app_name: String,
    pub window_title: String,
    pub selected_files: Vec<std::path::PathBuf>,
}

impl ManualContextProvider {
    pub fn new(app_name: impl Into<String>, window_title: impl Into<String>) -> Self {
        Self {
            app_name: app_name.into(),
            window_title: window_title.into(),
            selected_files: Vec::new(),
        }
    }

    pub fn from_env() -> Self {
        Self {
            app_name: std::env::var("RELAY_CONTEXT_APP").unwrap_or_default(),
            window_title: std::env::var("RELAY_CONTEXT_WINDOW").unwrap_or_default(),
            selected_files: Vec::new(),
        }
    }
}

impl ContextProvider for ManualContextProvider {
    fn capture(&self, options: &CaptureOptions) -> Result<EnvironmentSnapshot, ContextError> {
        let snapshot = EnvironmentSnapshot {
            app_name: self.app_name.clone(),
            window_title: self.window_title.clone(),
            selected_files: self.selected_files.clone(),
            source: ContextSource::Manual,
            ..Default::default()
        };
        Ok(finalize(snapshot, options))
    }

    fn name(&self) -> &'static str {
        "manual"
    }
}
