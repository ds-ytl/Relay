use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result};
use relay_context::{default_provider, ContextProvider};
use relay_openai::{LanguageModel, MockModel, MockStep, OpenAiClient, OpenAiConfig, WebTools};
use relay_runtime::RuntimeConfig;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    /// Global hotkey, e.g. ctrl+alt+space / cmd+shift+space.
    pub hotkey: String,
    pub model: String,
    pub base_url: String,
    /// Model API key; when empty, RELAY_OPENAI_API_KEY / OPENAI_API_KEY is read from the env.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    pub include_clipboard: bool,
    pub include_screenshot: bool,
    /// Grab the current window screenshot into the first context before understanding (on by default: one round trip fewer).
    pub prefetch_screen: bool,
    pub require_confirmation: bool,
    pub auto_open: bool,
    /// Enable macOS sandbox-exec isolation; off by default, so scripts may touch any path and use the network.
    pub sandbox: bool,
    pub max_rounds: u32,
    pub timeout_secs: u64,
    /// Model context window (tokens); 0 means no accounting and no compaction.
    pub context_window: u32,
    pub allowed_dirs: Vec<PathBuf>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            hotkey: "ctrl+alt+space".into(),
            model: std::env::var("RELAY_OPENAI_MODEL").unwrap_or_else(|_| "deepseek-flash".into()),
            base_url: std::env::var("RELAY_OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".into()),
            api_key: None,
            include_clipboard: true,
            include_screenshot: false,
            prefetch_screen: true,
            require_confirmation: true,
            auto_open: false,
            sandbox: false,
            max_rounds: 24,
            timeout_secs: 300,
            context_window: 256_000,
            allowed_dirs: Vec::new(),
        }
    }
}

impl AppConfig {
    pub fn home() -> PathBuf {
        std::env::var("RELAY_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                dirs::home_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join(".relay")
            })
    }

    pub fn config_path() -> PathBuf {
        Self::home().join("config.toml")
    }

    pub fn load() -> Self {
        let path = Self::config_path();
        let mut config: AppConfig = match std::fs::read_to_string(&path) {
            Ok(raw) => toml::from_str(&raw).unwrap_or_else(|err| {
                tracing::warn!(%err, path = %path.display(), "failed to parse the config file, using defaults");
                AppConfig::default()
            }),
            Err(_) => AppConfig::default(),
        };
        config.normalize();
        config
    }

    /// Write the default config on first run so the user can edit it right away.
    pub fn ensure_config_file() -> Result<()> {
        let path = Self::config_path();
        if path.exists() {
            return Ok(());
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let config = AppConfig::default();
        std::fs::write(&path, toml::to_string_pretty(&config)?)
            .with_context(|| format!("write {}", path.display()))?;
        Ok(())
    }

    fn normalize(&mut self) {
        if self.hotkey.trim().is_empty() {
            self.hotkey = AppConfig::default().hotkey;
        }
        if self.max_rounds == 0 {
            self.max_rounds = 24;
        }
        if self.timeout_secs == 0 {
            self.timeout_secs = 300;
        }
    }

    pub fn runtime_config(&self) -> RuntimeConfig {
        let data_dir = Self::home();
        let mut config = RuntimeConfig::new(data_dir);
        config.include_clipboard = self.include_clipboard;
        config.include_screenshot = self.include_screenshot;
        config.prefetch_screen = self.prefetch_screen;
        // No separate vision model: images (screenshots, invoices) go to the same model.
        config.vision_model = None;
        config.policy.require_confirmation = self.require_confirmation;
        config.policy.auto_open = self.auto_open;
        config.policy.sandbox = self.sandbox;
        config.policy.max_rounds = self.max_rounds;
        config.policy.timeout_secs = self.timeout_secs;
        config.max_rounds = self.max_rounds;
        config.context_window = self.context_window;
        // Search / fetch reuse the model key and base URL: `{base_url}/tools/{search,fetch}`.
        config.web_tools = self
            .openai_config()
            .map(|openai| Arc::new(WebTools::new(openai.api_key, openai.base_url)));
        for dir in &self.allowed_dirs {
            if dir.exists() {
                config.policy.allow_dir(dir);
            }
        }
        // Projects under the home directory are not allowed by default; only ~/.relay itself is, extended when workspaces are created.
        config.policy.allow_dir(&config.data_dir);
        config
    }

    pub fn context_provider(&self) -> Arc<dyn ContextProvider> {
        Arc::from(default_provider())
    }

    /// The environment variable wins, then api_key from the config file.
    pub fn openai_config(&self) -> Option<OpenAiConfig> {
        let api_key = OpenAiConfig::api_key_from_env().or_else(|| {
            self.api_key
                .as_deref()
                .map(str::trim)
                .filter(|key| !key.is_empty())
                .map(str::to_string)
        })?;
        let mut config = OpenAiConfig::with_key(api_key);
        config.model = self.model.clone();
        config.base_url = self.base_url.clone();
        Some(config)
    }

    /// Whether the key is set: returns the source for doctor to show.
    pub fn api_key_source(&self) -> &'static str {
        if OpenAiConfig::api_key_from_env().is_some() {
            "env var"
        } else if self
            .api_key
            .as_deref()
            .map(str::trim)
            .is_some_and(|key| !key.is_empty())
        {
            "config file"
        } else {
            "not set"
        }
    }

    /// Use the Responses API when a key exists; otherwise fall back to the demo model (the UI says so).
    pub fn build_model(&self) -> (Arc<dyn LanguageModel>, bool) {
        match self.openai_config() {
            Some(config) => match OpenAiClient::new(config) {
                Ok(client) => (Arc::new(client), false),
                Err(err) => {
                    tracing::error!(%err, "failed to initialize the OpenAI client, falling back to the demo model");
                    (demo_model(), true)
                }
            },
            None => (demo_model(), true),
        }
    }
}

/// Offline demo model: walks the understand -> execute -> verify -> deliver loop with no external calls.
pub fn demo_model() -> Arc<dyn LanguageModel> {
    // The demo script uses a structured bash tool call: code blocks in the reply are never executed.
    let script = r#"set -e
cat > "$RELAY_WORKSPACE_DIR/relay-demo.md" <<'EOF'
# Relay demo task

Hand off the work. Keep your flow.

- Context aware: read the current window
- Auto routing: matched a workspace
- Background run: handled by the Bash engine
- Delivery: artifacts written into the workspace

Set api_key in ~/.relay/config.toml (or RELAY_OPENAI_API_KEY) to run real tasks.
EOF
relay-state done --summary "demo file created" --artifact relay-demo.md"#;
    Arc::new(MockModel::scripted([
        MockStep::Text("TITLE: Demo task\nSUMMARY: No API key configured, so Relay wrote a short note with the demo model.\nSTEPS:\n- Write relay-demo.md into the workspace\n- Report the done state\nREFERENCES:\n- none\nDELIVERY: notify_only".into()),
        MockStep::call("bash", serde_json::json!({ "command": script })),
        MockStep::Text("VERDICT: PASS\nSUMMARY: demo file created\nCHECKS:\n- relay-demo.md exists and is complete\nDELIVER: relay-demo.md".into()),
    ]))
}

/// Log retention in days: older files are pruned at startup and every 6 hours.
const LOG_RETENTION_DAYS: u64 = 7;
/// File-count fallback for daily rotation: keep one extra file across day boundaries.
const LOG_KEEP_FILES: usize = 8;
/// Guard for non-blocking writes; it must live as long as the process or tail logs are lost.
static LOG_GUARD: OnceLock<tracing_appender::non_blocking::WorkerGuard> = OnceLock::new();

/// Log lines use the local time zone so debugging needs no UTC math.
struct LocalTimer;

impl tracing_subscriber::fmt::time::FormatTime for LocalTimer {
    fn format_time(
        &self,
        writer: &mut tracing_subscriber::fmt::format::Writer<'_>,
    ) -> std::fmt::Result {
        write!(
            writer,
            "{}",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f")
        )
    }
}

/// Init logging: `~/.relay/logs/relay.<date>.log` rotates daily, keeping the last week.
///
/// Default filter `relay=info,relay_runtime=info,relay_openai=info,warn`,
/// overridable with `RUST_LOG` or `RELAY_LOG` (e.g. `RELAY_LOG=relay=debug`).
pub fn init_tracing() {
    use tracing_subscriber::layer::SubscriberExt as _;
    use tracing_subscriber::util::SubscriberInitExt as _;

    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .or_else(|_| tracing_subscriber::EnvFilter::try_from_env("RELAY_LOG"))
        .unwrap_or_else(|_| {
            tracing_subscriber::EnvFilter::new(
                "relay=info,relay_runtime=info,relay_openai=info,warn",
            )
        });

    let dir = AppConfig::home().join("logs");
    if let Err(err) = std::fs::create_dir_all(&dir) {
        eprintln!(
            "failed to create the log directory ({}): {err}",
            dir.display()
        );
    }
    let removed = prune_logs(&dir, retention_cutoff());
    spawn_log_pruner(dir.clone());

    let appender = tracing_appender::rolling::Builder::new()
        .rotation(tracing_appender::rolling::Rotation::DAILY)
        .filename_prefix("relay")
        .filename_suffix("log")
        .max_log_files(LOG_KEEP_FILES)
        .build(&dir);

    let registry = tracing_subscriber::registry().with(filter);
    // Also log to the terminal when run from one; under launchd write files only to avoid doubling stdout.
    let console = std::io::stderr().is_terminal();
    match appender {
        Ok(appender) => {
            let (writer, guard) = tracing_appender::non_blocking(appender);
            let _ = LOG_GUARD.set(guard);
            let console_layer = console.then(|| {
                tracing_subscriber::fmt::layer()
                    .with_timer(LocalTimer)
                    .with_target(false)
                    .with_ansi(true)
            });
            let _ = registry
                .with(console_layer)
                .with(
                    tracing_subscriber::fmt::layer()
                        .with_timer(LocalTimer)
                        .with_writer(writer)
                        .with_ansi(false)
                        .with_target(false),
                )
                .try_init();
        }
        Err(err) => {
            eprintln!("failed to initialize log files, writing to the terminal only: {err}");
            let _ = registry
                .with(
                    tracing_subscriber::fmt::layer()
                        .with_timer(LocalTimer)
                        .with_target(false)
                        .with_ansi(console),
                )
                .try_init();
        }
    }

    tracing::info!(
        dir = %dir.display(),
        retention_days = LOG_RETENTION_DAYS,
        pruned = removed,
        "log sink ready"
    );
}

fn retention_cutoff() -> SystemTime {
    SystemTime::now() - Duration::from_secs(LOG_RETENTION_DAYS * 24 * 60 * 60)
}

/// Delete outdated relay logs (a week old by default) and return how many were removed.
///
/// Only `relay*.log` files are touched; nothing else in the directory.
fn prune_logs(dir: &Path, cutoff: SystemTime) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    let mut removed = 0;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.starts_with("relay") || !name.ends_with(".log") {
            continue;
        }
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        if !metadata.is_file() {
            continue;
        }
        let modified = metadata.modified().unwrap_or_else(|_| SystemTime::now());
        if modified < cutoff && std::fs::remove_file(entry.path()).is_ok() {
            removed += 1;
        }
    }
    removed
}

/// The process lives across days, so prune every 6 hours on top of the startup pass.
fn spawn_log_pruner(dir: PathBuf) {
    std::thread::Builder::new()
        .name("relay-log-pruner".into())
        .spawn(move || loop {
            std::thread::sleep(Duration::from_secs(6 * 60 * 60));
            let removed = prune_logs(&dir, retention_cutoff());
            if removed > 0 {
                tracing::info!(pruned = removed, "pruned expired logs");
            }
        })
        .ok();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{File, FileTimes};
    use std::io::Write;

    fn touch(path: &Path, modified: SystemTime) {
        let file = File::options().write(true).open(path).unwrap();
        file.set_times(FileTimes::new().set_modified(modified))
            .unwrap();
    }

    #[test]
    fn prune_logs_removes_only_expired_relay_logs() {
        let dir = std::env::temp_dir().join(format!("relay-logs-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();

        let old = dir.join("relay.2026-01-01.log");
        let fresh = dir.join("relay.2026-11-25.log");
        let other = dir.join("notes.txt");
        for path in [&old, &fresh, &other] {
            let mut file = File::create(path).unwrap();
            writeln!(file, "x").unwrap();
        }

        let ten_days_ago = SystemTime::now() - Duration::from_secs(10 * 24 * 60 * 60);
        touch(&old, ten_days_ago);

        let removed = prune_logs(&dir, retention_cutoff());

        assert_eq!(removed, 1, "only the expired file should be removed");
        assert!(!old.exists(), "logs older than a week must be deleted");
        assert!(fresh.exists(), "today's log must be kept");
        assert!(other.exists(), "non-log files must not be removed");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn prune_logs_keeps_everything_when_nothing_expired() {
        let dir = std::env::temp_dir().join(format!("relay-logs-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let fresh = dir.join("relay.2026-11-25.log");
        File::create(&fresh).unwrap();

        assert_eq!(prune_logs(&dir, retention_cutoff()), 0);
        assert!(fresh.exists());

        std::fs::remove_dir_all(&dir).ok();
    }
}
