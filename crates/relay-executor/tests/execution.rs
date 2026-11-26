use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use relay_core::{ControlPhase, ExecutionPolicy};
use relay_executor::{BashExecutor, ExecutionRequest, ExecutorError};

fn setup() -> (tempfile::TempDir, BashExecutor, PathBuf) {
    let temp = tempfile::tempdir().unwrap();
    let task_dir = temp.path().join("task");
    std::fs::create_dir_all(&task_dir).unwrap();
    let mut policy = ExecutionPolicy::default();
    policy.allow_dir(temp.path());
    policy.sandbox = false;
    policy.timeout_secs = 5;
    (temp, BashExecutor::new(policy), task_dir)
}

#[tokio::test]
async fn runs_bash_and_collects_output() {
    let (_temp, executor, task_dir) = setup();
    let result = executor
        .execute(ExecutionRequest {
            script: "echo hello; echo oops >&2; exit 3".into(),
            cwd: task_dir.clone(),
            task_dir: task_dir.clone(),
            env: vec![],
            timeout: Duration::from_secs(5),
            turn: 1,
            cancel: None,
        })
        .await
        .unwrap();

    assert_eq!(result.exit_code, Some(3));
    assert!(result.stdout.contains("hello"));
    assert!(result.stderr.contains("oops"));
    assert!(!result.timed_out);
}

#[tokio::test]
async fn kills_long_running_script() {
    let (_temp, executor, task_dir) = setup();
    let result = executor
        .execute(ExecutionRequest {
            script: "sleep 30".into(),
            cwd: task_dir.clone(),
            task_dir: task_dir.clone(),
            env: vec![],
            timeout: Duration::from_millis(300),
            turn: 1,
            cancel: None,
        })
        .await
        .unwrap();

    assert!(result.timed_out);
    assert!(result.duration_ms < 5_000);
}

#[tokio::test]
async fn reads_state_protocol_and_artifacts() {
    let (_temp, executor, task_dir) = setup();
    let script = r#"
set -e
printf 'report' > "$RELAY_TASK_DIR/report.md"
mkdir -p "$RELAY_TASK_DIR/.relay"
printf '{"phase":"done","summary":"Done","artifacts":["report.md"]}' > "$RELAY_STATE_FILE"
"#;
    let result = executor
        .execute(ExecutionRequest {
            script: script.into(),
            cwd: task_dir.clone(),
            task_dir: task_dir.clone(),
            env: vec![],
            timeout: Duration::from_secs(10),
            turn: 1,
            cancel: None,
        })
        .await
        .unwrap();

    assert!(result.succeeded(), "{:?}", result.stderr);
    let control = result.control.expect("state");
    assert_eq!(control.phase, ControlPhase::Done);
    assert_eq!(control.summary, "Done");
    assert!(task_dir.join("report.md").exists());
}

#[tokio::test]
async fn refuses_paths_outside_policy() {
    let (_temp, executor, _task_dir) = setup();
    let outsider = tempfile::tempdir().unwrap();
    let result = executor
        .execute(ExecutionRequest {
            script: "echo nope".into(),
            cwd: outsider.path().to_path_buf(),
            task_dir: outsider.path().to_path_buf(),
            env: vec![],
            timeout: Duration::from_secs(1),
            turn: 1,
            cancel: None,
        })
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn cancel_kills_a_running_script_immediately() {
    let (_temp, executor, task_dir) = setup();
    let flag = Arc::new(AtomicBool::new(false));
    let trigger = flag.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(200)).await;
        trigger.store(true, Ordering::SeqCst);
    });

    let started = Instant::now();
    let result = executor
        .execute(ExecutionRequest {
            script: "sleep 30".into(),
            cwd: task_dir.clone(),
            task_dir: task_dir.clone(),
            env: vec![],
            timeout: Duration::from_secs(30),
            turn: 1,
            cancel: Some(flag),
        })
        .await;

    assert!(matches!(result, Err(ExecutorError::Cancelled)));
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "stopping should kill the process immediately; took {:?}",
        started.elapsed()
    );
}

#[tokio::test]
async fn already_cancelled_request_never_spawns() {
    let (_temp, executor, task_dir) = setup();
    let result = executor
        .execute(ExecutionRequest {
            script: "echo should-not-run".into(),
            cwd: task_dir.clone(),
            task_dir: task_dir.clone(),
            env: vec![],
            timeout: Duration::from_secs(5),
            turn: 1,
            cancel: Some(Arc::new(AtomicBool::new(true))),
        })
        .await;
    assert!(matches!(result, Err(ExecutorError::Cancelled)));
}
