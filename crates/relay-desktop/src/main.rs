//! Relay desktop client entry point.
//!
//! - `relay`                      tray-resident desktop panel, woken by hotkey
//! - `relay run "task" [--mock]`  run a single task headless
//! - `relay tasks`                list recent tasks
//! - `relay workspaces`           list workspaces
//! - `relay doctor`               check the local environment

#[cfg(feature = "gui")]
mod app;
mod config;
#[cfg(feature = "gui")]
mod conversation;
mod hotkey;
#[cfg(feature = "gui")]
mod instance;
#[cfg(feature = "gui")]
mod snip;
#[cfg(feature = "gui")]
mod tray;
#[cfg(feature = "gui")]
mod voice;

use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use config::{demo_model, init_tracing, AppConfig};
use relay_core::{TaskId, TaskState};
use relay_openai::LanguageModel;
use relay_runtime::{Runtime, RuntimeConfig, TaskRequest};
use relay_storage::Store;

fn main() {
    init_tracing();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match run(args) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("relay: {err:#}");
            1
        }
    };
    std::process::exit(code);
}

fn run(args: Vec<String>) -> Result<i32> {
    let config = AppConfig::load();
    match args.first().map(String::as_str) {
        None => {
            #[cfg(feature = "gui")]
            {
                app::run(config)?;
                Ok(0)
            }
            #[cfg(not(feature = "gui"))]
            {
                let _ = config;
                bail!("this build has no UI; use `relay run \"task\"`");
            }
        }
        Some("run") | Some("ask") => {
            let rest = &args[1..];
            let mock = rest.iter().any(|a| a == "--mock");
            let confirm = rest.iter().any(|a| a == "--confirm");
            let instruction = rest
                .iter()
                .filter(|a| !a.starts_with("--"))
                .cloned()
                .collect::<Vec<_>>()
                .join(" ");
            if instruction.trim().is_empty() {
                bail!("usage: relay run \"task to complete\" [--mock]");
            }
            cmd_run(config, instruction, mock, !confirm)
        }
        Some("tasks") => cmd_tasks(config),
        Some("workspaces") => cmd_workspaces(config),
        Some("doctor") => cmd_doctor(config),
        Some("--help") | Some("-h") | Some("help") => {
            print_help();
            Ok(0)
        }
        Some(other) => {
            print_help();
            bail!("unknown command: {other}");
        }
    }
}

fn print_help() {
    println!(
        "Relay · Hand off the work. Keep your flow.\n\n\
         usage:\n  relay                       start the desktop panel (hotkey)\n  \
         relay run \"task\" [--mock]   run a single task headless\n  \
         relay tasks                 list recent tasks\n  \
         relay workspaces            list workspaces\n  \
         relay doctor                check the runtime environment\n\n\
         config: ~/.relay/config.toml (model / base_url / api_key)\n  \
         env: RELAY_OPENAI_API_KEY (or OPENAI_API_KEY), RELAY_OPENAI_MODEL, RELAY_OPENAI_BASE_URL, RELAY_HOME"
    );
}

fn build_runtime(config: &AppConfig, force_mock: bool) -> Result<(Runtime, bool)> {
    config
        .runtime_config()
        .ensure_dirs()
        .context("initialize the data directory")?;
    AppConfig::ensure_config_file()?;
    let rt_config: RuntimeConfig = config.runtime_config();
    let store = Arc::new(Store::open(rt_config.db_path()).context("open SQLite")?);
    let (model, demo): (Arc<dyn LanguageModel>, bool) = if force_mock {
        (demo_model(), true)
    } else {
        config.build_model()
    };
    let runtime = Runtime::new(store, model, config.context_provider(), rt_config)?;
    Ok((runtime, demo))
}

fn cmd_run(config: AppConfig, instruction: String, mock: bool, auto_confirm: bool) -> Result<i32> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let (runtime, demo) = build_runtime(&config, mock)?;
    if demo {
        eprintln!("note: no API key configured, running on the demo model. Add api_key to ~/.relay/config.toml to run real tasks.");
    }

    rt.block_on(async move {
        let mut events = runtime.subscribe();
        let mut understanding_events = runtime.subscribe_understanding();
        let task_id = runtime
            .submit(TaskRequest {
                instruction,
                supplement: None,
                pinned_workspace: None,
                pinned_session: None,
                auto_confirm,
                direct: false,
                attachments: Vec::new(),
            })
            .await?;

        let printer = tokio::spawn(async move {
            loop {
                tokio::select! {
                    event = events.recv() => match event {
                        Ok(event) => println!("[{}] {}", short_id(event.task_id), event.summary()),
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                        Err(_) => break,
                    },
                    // Understanding trail: show the user what it is checking.
                    trail = understanding_events.recv() => {
                        if let Ok(event) = trail {
                            println!("  · understanding {}", event.summary);
                        }
                    }
                }
            }
        });

        let state = wait_terminal(&runtime, task_id).await?;
        printer.abort();
        let task = runtime.task(task_id)?.context("task missing")?;
        println!("\nState: {}", state.label());
        if let Some(report) = &task.report {
            println!("Result: {}", report.summary);
            for check in &report.checks {
                println!("  - {check}");
            }
        }
        if let Some(error) = &task.error {
            println!("Error: {error}");
        }
        let artifacts = runtime.artifacts(task_id)?;
        if !artifacts.is_empty() {
            println!("Artifacts:");
            for artifact in &artifacts {
                println!("  - {}", artifact.path.display());
            }
        }
        println!("Task ID: {task_id}");
        Ok(match state {
            TaskState::Succeeded => 0,
            _ => 1,
        })
    })
}

async fn wait_terminal(runtime: &Runtime, task_id: TaskId) -> Result<TaskState> {
    loop {
        let Some(task) = runtime.task(task_id)? else {
            bail!("task not found: {task_id}");
        };
        if task.state.is_terminal() {
            return Ok(task.state);
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

fn cmd_tasks(config: AppConfig) -> Result<i32> {
    let (runtime, _) = build_runtime(&config, true)?;
    let tasks = runtime.recent_tasks(20)?;
    if tasks.is_empty() {
        println!("No tasks yet. Summon Relay with the hotkey, or run relay run \"task\".");
        return Ok(0);
    }
    println!("{:<10} {:<28} {:<16} State", "Task", "Title", "Workspace");
    for view in tasks {
        let title: String = view.task.title.chars().take(28).collect();
        println!(
            "{:<10} {:<28} {:<16} {}",
            short_id(view.task.id),
            title,
            view.workspace_name,
            view.task.state.label()
        );
    }
    Ok(0)
}

fn cmd_workspaces(config: AppConfig) -> Result<i32> {
    let (runtime, _) = build_runtime(&config, true)?;
    let workspaces = runtime.workspaces()?;
    if workspaces.is_empty() {
        println!("No workspaces yet; the first task creates one automatically.");
        return Ok(0);
    }
    for workspace in workspaces {
        println!(
            "{}  ·  {}  ·  {}",
            workspace.name,
            workspace.root_dir.display(),
            workspace.keywords.join(", ")
        );
    }
    Ok(0)
}

fn cmd_doctor(config: AppConfig) -> Result<i32> {
    println!("Relay environment check\n");
    let home = AppConfig::home();
    println!("Data directory: {}", home.display());
    println!(
        "Config file: {}",
        if AppConfig::config_path().exists() {
            format!("{} (exists)", AppConfig::config_path().display())
        } else {
            format!(
                "{} (created on first run)",
                AppConfig::config_path().display()
            )
        }
    );
    let key_source = config.api_key_source();
    println!("Model: {} (base URL: {})", config.model, config.base_url);
    println!(
        "Web tools: {}",
        match config.openai_config() {
            Some(_) => format!(
                "search + fetch via {}/tools",
                config.base_url.trim_end_matches('/')
            ),
            None => "disabled (no API key)".to_string(),
        }
    );
    println!(
        "API Key: {}",
        if key_source == "not set" {
            "not set, using the demo model (put api_key in the config file or set RELAY_OPENAI_API_KEY)".to_string()
        } else {
            format!("configured (source: {key_source})")
        }
    );
    println!("Hotkey: {}", config.hotkey);
    println!(
        "Bash: {}",
        which("bash").unwrap_or_else(|| "not found".into())
    );
    println!(
        "Python: {}",
        which("python3")
            .or_else(|| which("python"))
            .unwrap_or_else(|| "not found (business calculations will be limited)".into())
    );
    println!(
        "Execution isolation: {}",
        if relay_sandbox::is_supported() {
            "sandbox-exec available (macOS system-level isolation)"
        } else {
            "unavailable (falls back to path and command policy here)"
        }
    );
    println!(
        "Tray / global hotkey / screenshot / clipboard: {} ({}-{})",
        "tray-icon + global-hotkey + xcap + arboard",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    println!(
        "Disk: {}",
        match std::fs::create_dir_all(&home) {
            Ok(_) => "writable",
            Err(_) => "not writable",
        }
    );
    Ok(0)
}

fn which(name: &str) -> Option<String> {
    // Windows executables carry an extension; try both.
    let mut names = vec![name.to_string()];
    if cfg!(windows) && !name.ends_with(".exe") {
        names.push(format!("{name}.exe"));
    }
    for dir in std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()) {
        for candidate_name in &names {
            let candidate = dir.join(candidate_name);
            if candidate.is_file() {
                return Some(candidate.display().to_string());
            }
        }
    }
    None
}

fn short_id(id: TaskId) -> String {
    id.to_string().chars().take(8).collect()
}
