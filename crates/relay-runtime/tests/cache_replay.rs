//! Reopening must send the same bytes: every request after a restart keeps the recorded
//! prefix verbatim, so the provider's token cache still hits.

use std::sync::Arc;
use std::time::{Duration, Instant};

use relay_context::ManualContextProvider;
use relay_core::TaskState;
use relay_openai::{MockModel, MockStep};
use relay_runtime::{Runtime, RuntimeConfig, TaskRequest};
use relay_storage::Store;

fn plan_step(reuse: Option<(&str, &str)>) -> MockStep {
    match reuse {
        None => MockStep::call(
            "submit_plan",
            serde_json::json!({
                "workspace_mode": "create",
                "workspace_name": "Deliverable",
                "workspace_keywords": ["deliverable", "out.txt"],
                "session_mode": "create",
                "session_title": "Generate the deliverable",
                "title": "Generate the deliverable",
                "summary": "Write out.txt",
                "steps": ["Write out.txt"],
                "references": [],
                "delivery": "notify_only",
                "confidence": 0.9,
                "reason": "mock"
            }),
        ),
        Some((workspace, session)) => MockStep::call(
            "submit_plan",
            serde_json::json!({
                "workspace_mode": "reuse",
                "workspace_id": workspace,
                "session_mode": "reuse",
                "session_id": session,
                "title": "Continue the work",
                "summary": "Continue",
                "steps": ["Continue"],
                "references": [],
                "delivery": "notify_only",
                "confidence": 0.9,
                "reason": "mock"
            }),
        ),
    }
}

fn bash(command: &str) -> MockStep {
    MockStep::call("bash", serde_json::json!({ "command": command }))
}

fn result(phase: &str, summary: &str) -> MockStep {
    MockStep::call(
        "submit_result",
        serde_json::json!({ "phase": phase, "summary": summary }),
    )
}

/// Input of the last executor request the model received: the bytes a restart has to replay.
fn last_executor_input(model: &MockModel) -> serde_json::Value {
    model
        .payloads()
        .into_iter()
        .rfind(|payload| {
            payload["instructions"]
                .as_str()
                .is_some_and(|text| text.contains("Relay Executor Agent"))
        })
        .expect("an executor request")["input"]
        .clone()
}

fn config(dir: &std::path::Path) -> RuntimeConfig {
    let mut config = RuntimeConfig::new(dir.join("data"));
    config.policy.allow_dir(dir);
    config.policy.sandbox = false;
    config.policy.auto_open = false;
    config.notifications = false;
    config.auto_confirm = true;
    config
}

async fn wait_for(
    runtime: &Runtime,
    task: relay_core::TaskId,
    predicate: impl Fn(TaskState) -> bool,
) -> TaskState {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let state = runtime.task(task).unwrap().unwrap().state;
        if predicate(state) {
            return state;
        }
        assert!(Instant::now() < deadline, "timed out: {state}");
        tokio::time::sleep(Duration::from_millis(60)).await;
    }
}

#[tokio::test]
async fn restart_mid_task_replays_the_same_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let config = config(dir.path());

    let before = Arc::new(MockModel::scripted([
        plan_step(None),
        bash("echo one"),
        bash("echo two"),
        result("continue", "first turn done"),
        bash("echo three"),
        result("paused", "waiting for review"),
    ]));
    let (task_id, last_input) = {
        let runtime = Runtime::new(
            Arc::new(Store::open(config.db_path()).unwrap()),
            before.clone(),
            Arc::new(ManualContextProvider::default()),
            config.clone(),
        )
        .unwrap();
        let task = runtime
            .submit(TaskRequest::new("Generate a deliverable file"))
            .await
            .unwrap();
        assert_eq!(
            wait_for(&runtime, task, |s| s == TaskState::Paused).await,
            TaskState::Paused
        );
        (task, before.payloads().pop().unwrap()["input"].clone())
    };

    let after = Arc::new(MockModel::scripted([
        result("done", "finished"),
        MockStep::Text("VERDICT: PASS\nSUMMARY: ok\nCHECKS:\n- out.txt\nDELIVER: out.txt".into()),
    ]));
    let runtime = Runtime::new(
        Arc::new(Store::open(config.db_path()).unwrap()),
        after.clone(),
        Arc::new(ManualContextProvider::default()),
        config.clone(),
    )
    .unwrap();
    runtime.resume(task_id).await.unwrap();
    assert_eq!(
        wait_for(&runtime, task_id, TaskState::is_terminal).await,
        TaskState::Succeeded
    );
    let payloads = after.payloads();
    let next = payloads[0]["input"].as_array().unwrap();
    let old = last_input.as_array().unwrap();
    assert_eq!(
        serde_json::to_string(old).unwrap(),
        serde_json::to_string(&next[..old.len()]).unwrap(),
        "reopened request must replay the previous input byte for byte"
    );
}

#[tokio::test]
async fn new_task_after_restart_replays_the_previous_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let config = config(dir.path());

    let before = Arc::new(MockModel::scripted([
        plan_step(None),
        bash("echo one"),
        result("done", "generated"),
        MockStep::Text("VERDICT: PASS\nSUMMARY: ok\nCHECKS:\n- out.txt\nDELIVER: out.txt".into()),
    ]));
    let (workspace_id, session_id, last_input) = {
        let runtime = Runtime::new(
            Arc::new(Store::open(config.db_path()).unwrap()),
            before.clone(),
            Arc::new(ManualContextProvider::default()),
            config.clone(),
        )
        .unwrap();
        let task = runtime
            .submit(TaskRequest::new("Generate a deliverable file"))
            .await
            .unwrap();
        assert_eq!(
            wait_for(&runtime, task, TaskState::is_terminal).await,
            TaskState::Succeeded
        );
        let stored = runtime.task(task).unwrap().unwrap();
        (
            stored.workspace_id,
            stored.session_id,
            last_executor_input(&before),
        )
    };

    let after = Arc::new(MockModel::scripted([
        plan_step(Some((&workspace_id.to_string(), &session_id.to_string()))),
        bash("echo two"),
        result("done", "continued"),
        MockStep::Text("VERDICT: PASS\nSUMMARY: ok\nCHECKS:\n- out.txt\nDELIVER: out.txt".into()),
    ]));
    let runtime = Runtime::new(
        Arc::new(Store::open(config.db_path()).unwrap()),
        after.clone(),
        Arc::new(ManualContextProvider::default()),
        config.clone(),
    )
    .unwrap();
    let task = runtime
        .submit(TaskRequest {
            pinned_session: Some(session_id),
            pinned_workspace: Some(workspace_id),
            ..TaskRequest::new("Continue the work")
        })
        .await
        .unwrap();
    assert_eq!(
        wait_for(&runtime, task, TaskState::is_terminal).await,
        TaskState::Succeeded
    );
    let payloads = after.payloads();
    let next = payloads
        .iter()
        .filter(|p| {
            p["instructions"]
                .as_str()
                .is_some_and(|i| i.contains("Relay Executor Agent"))
        })
        .map(|p| p["input"].as_array().unwrap().clone())
        .next()
        .expect("continuation request");
    let old = last_input.as_array().unwrap();
    assert_eq!(
        serde_json::to_string(old).unwrap(),
        serde_json::to_string(&next[..old.len()]).unwrap(),
        "a continuation after restart must replay the previous task's input byte for byte"
    );
}

#[tokio::test]
async fn restart_between_dispatch_and_checkpoint_still_replays_the_recorded_input() {
    let dir = tempfile::tempdir().unwrap();
    let config = config(dir.path());

    let before = Arc::new(MockModel::scripted([
        plan_step(None),
        bash("echo one"),
        result("paused", "waiting for review"),
    ]));
    let (task_id, recorded_input) = {
        let runtime = Runtime::new(
            Arc::new(Store::open(config.db_path()).unwrap()),
            before.clone(),
            Arc::new(ManualContextProvider::default()),
            config.clone(),
        )
        .unwrap();
        let task = runtime
            .submit(TaskRequest::new("Generate a deliverable file"))
            .await
            .unwrap();
        assert_eq!(
            wait_for(&runtime, task, |s| s == TaskState::Paused).await,
            TaskState::Paused
        );
        (task, before.payloads().pop().unwrap()["input"].clone())
    };

    // Simulate a crash between "request recorded" and "checkpoint written": the request survives,
    // the checkpoint row does not.
    let db = rusqlite::Connection::open(config.db_path()).unwrap();
    let removed = db
        .execute(
            "DELETE FROM executor_checkpoints WHERE task_id=?1",
            [task_id.to_string()],
        )
        .unwrap();
    assert_eq!(
        removed, 1,
        "the checkpoint row should be the only one removed"
    );
    drop(db);

    let after = Arc::new(MockModel::scripted([
        result("done", "finished"),
        MockStep::Text("VERDICT: PASS\nSUMMARY: ok\nCHECKS:\n- out.txt\nDELIVER: out.txt".into()),
    ]));
    let runtime = Runtime::new(
        Arc::new(Store::open(config.db_path()).unwrap()),
        after.clone(),
        Arc::new(ManualContextProvider::default()),
        config.clone(),
    )
    .unwrap();
    runtime.resume(task_id).await.unwrap();
    assert_eq!(
        wait_for(&runtime, task_id, TaskState::is_terminal).await,
        TaskState::Succeeded
    );

    let payloads = after.payloads();
    let next = payloads[0]["input"].as_array().unwrap();
    let old = recorded_input.as_array().unwrap();
    assert_eq!(
        serde_json::to_string(old).unwrap(),
        serde_json::to_string(&next[..old.len()]).unwrap(),
        "a recorded request must still be replayed verbatim when its checkpoint is missing"
    );
    // The model has to learn that the round was cut off; otherwise it assumes its own last
    // message and tool results survived the restart.
    let appended = serde_json::to_string(&next[old.len()..]).unwrap();
    assert!(
        appended.contains("Interrupted round"),
        "the replay must explain the cut-off round: {appended}"
    );
}
