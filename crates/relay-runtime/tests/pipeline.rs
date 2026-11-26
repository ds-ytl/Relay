use std::sync::Arc;
use std::time::{Duration, Instant};

use relay_context::ManualContextProvider;
use relay_core::{MessageRole, TaskState, Verdict};
use relay_openai::{MockModel, MockStep};
use relay_runtime::{Confirmation, Runtime, RuntimeConfig, TaskRequest};
use relay_storage::Store;

fn plan_text() -> &'static str {
    "TITLE: Generate the deliverable\nSUMMARY: Write out.txt into the workspace and deliver it.\nSTEPS:\n- Write out.txt\n- Verify the file contents\nREFERENCES:\n- none\nDELIVERY: notify_only"
}

/// One executor turn: a structured bash tool call (code fences in the reply are no longer executed).
fn exec_step() -> MockStep {
    MockStep::call(
        "bash",
        serde_json::json!({
            "command": "set -e\nprintf 'hello relay' > \"$RELAY_WORKSPACE_DIR/out.txt\"\nrelay-state done --summary \"out.txt generated\" --artifact out.txt"
        }),
    )
}

fn verify_text() -> &'static str {
    "VERDICT: PASS\nSUMMARY: out.txt exists and its contents are correct\nCHECKS:\n- out.txt exists\nDELIVER: out.txt"
}

fn step(text: &str) -> MockStep {
    MockStep::Text(text.to_string())
}

/// The understanding agent's final move: submit_plan (create a workspace + create a session).
fn plan_step() -> MockStep {
    MockStep::call(
        "submit_plan",
        serde_json::json!({
            "workspace_mode": "create",
            "workspace_name": "Deliverable",
            "workspace_keywords": ["deliverable", "out.txt"],
            "session_mode": "create",
            "session_title": "Generate the deliverable",
            "title": "Generate the deliverable",
            "summary": "Write out.txt into the workspace and deliver it.",
            "steps": ["Write out.txt", "Verify the file contents"],
            "references": [],
            "delivery": "notify_only",
            "confidence": 0.9,
            "reason": "the mock script matched the same item"
        }),
    )
}

/// submit_plan that reuses an existing workspace and session.
fn reuse_plan_step(workspace_id: &str, session_id: &str) -> MockStep {
    MockStep::call(
        "submit_plan",
        serde_json::json!({
            "workspace_mode": "reuse",
            "workspace_id": workspace_id,
            "session_mode": "reuse",
            "session_id": session_id,
            "title": "Generate the deliverable",
            "summary": "Continue generating in the same session.",
            "steps": ["Write out.txt", "Verify the file contents"],
            "references": [],
            "delivery": "notify_only",
            "confidence": 0.9,
            "reason": "a continuation of the same session"
        }),
    )
}

fn runtime_with(
    model: Arc<dyn relay_openai::LanguageModel>,
    auto_confirm: bool,
) -> (tempfile::TempDir, Runtime) {
    let temp = tempfile::tempdir().unwrap();
    let mut config = RuntimeConfig::new(temp.path().join("data"));
    config.policy.allow_dir(temp.path());
    config.policy.sandbox = false;
    config.policy.auto_open = false;
    config.notifications = false;
    config.auto_confirm = auto_confirm;
    let store = Arc::new(Store::open(config.db_path()).unwrap());
    let context = Arc::new(ManualContextProvider {
        selected_files: vec![temp.path().join("source-file.csv")],
        ..ManualContextProvider::new("Visual Studio Code", "relay — src/main.rs")
    });
    let runtime = Runtime::new(store, model, context, config).unwrap();
    (temp, runtime)
}

async fn wait_for(
    runtime: &Runtime,
    task_id: relay_core::TaskId,
    predicate: impl Fn(TaskState) -> bool,
) -> TaskState {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let state = runtime.task(task_id).unwrap().unwrap().state;
        if predicate(state) {
            return state;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for the task state: {state}"
        );
        tokio::time::sleep(Duration::from_millis(80)).await;
    }
}

#[tokio::test]
async fn end_to_end_task_succeeds_with_artifacts() {
    let model = Arc::new(MockModel::scripted([
        plan_step(),
        exec_step(),
        step(verify_text()),
    ]));
    let (temp, runtime) = runtime_with(model.clone(), true);

    let task_id = runtime
        .submit(TaskRequest::new("Generate a deliverable file"))
        .await
        .unwrap();
    let state = wait_for(&runtime, task_id, TaskState::is_terminal).await;
    assert_eq!(state, TaskState::Succeeded);

    let task = runtime.task(task_id).unwrap().unwrap();
    assert_eq!(task.title, "Generate the deliverable");
    assert_eq!(task.report.unwrap().verdict, Some(Verdict::Pass));

    let artifacts = runtime.artifacts(task_id).unwrap();
    assert!(
        artifacts.iter().any(|a| a.path.ends_with("out.txt")),
        "{artifacts:?}"
    );
    assert!(temp.path().join("data/workspaces").exists());

    let views = runtime.recent_tasks(5).unwrap();
    assert_eq!(views.len(), 1);
    assert!(!views[0].workspace_name.is_empty());

    // Context usage is persisted with the task: the understanding, execution and verification calls all count against the same task.
    assert!(task.usage.input_tokens > 0, "{:?}", task.usage);
    assert!(task.usage.output_tokens > 0);
    assert!(task.usage.peak_input_tokens > 0);
    // Badge semantics: the size of the executor's main request must not be overwritten by a small verifier request.
    assert!(task.usage.context_tokens > 0, "{:?}", task.usage);
    assert_eq!(task.usage.occupancy(), task.usage.context_tokens);
    assert!(task.usage.cached_tokens > 0, "cache hits must be counted");
    assert!(task.usage.cache_ratio() > 0.0);

    // Both execution records and the state protocol are persisted.
    let steps = runtime.steps(task_id).unwrap();
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].phase.unwrap().as_str(), "done");
    assert!(
        model.remaining() == 0,
        "every scripted response should be consumed"
    );
}

#[tokio::test]
async fn executor_streams_assistant_deltas_without_persisting_them() {
    let model = Arc::new(MockModel::scripted([
        plan_step(),
        // Process reply (plain text) → structured bash call: deltas come from the reply text, no longer from a code fence.
        step("First I will write hello relay into the workspace and report the done state."),
        exec_step(),
        step(verify_text()),
    ]));
    let (_temp, runtime) = runtime_with(model, true);
    let mut events = runtime.subscribe();

    let task_id = runtime
        .submit(TaskRequest::new("Generate a deliverable file"))
        .await
        .unwrap();
    let state = wait_for(&runtime, task_id, TaskState::is_terminal).await;
    assert_eq!(state, TaskState::Succeeded);

    let mut streamed = String::new();
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_millis(100), events.recv()).await {
            Ok(Ok(event)) => {
                if let relay_core::RuntimeEventKind::AssistantDelta { delta, .. } = event.kind {
                    streamed.push_str(&delta);
                }
            }
            Ok(Err(_)) => break,
            Err(_) if !streamed.is_empty() => break,
            Err(_) => {}
        }
    }
    assert!(
        streamed.contains("hello relay"),
        "executor output should be broadcast as deltas: {streamed:?}"
    );

    // Deltas are ephemeral: not every token may be written into task_events.
    let persisted = runtime.events(task_id, 200).unwrap();
    assert!(
        !persisted.iter().any(|event| matches!(
            event.kind,
            relay_core::RuntimeEventKind::AssistantDelta { .. }
        )),
        "deltas must not be persisted"
    );

    // Whole messages, however, are persisted in real order: user → plan → tool → reply.
    let task = runtime.task(task_id).unwrap().unwrap();
    let messages = runtime.messages(task.session_id).unwrap();
    let roles: Vec<_> = messages.iter().map(|message| message.role).collect();
    assert_eq!(roles.first(), Some(&MessageRole::User));
    assert!(roles.contains(&MessageRole::Plan), "{roles:?}");
    assert!(roles.contains(&MessageRole::Tool), "{roles:?}");
    assert!(roles.contains(&MessageRole::Assistant), "{roles:?}");
    let assistant: String = messages
        .iter()
        .filter(|message| message.role == MessageRole::Assistant)
        .map(|message| message.content.clone())
        .collect();
    assert!(assistant.contains("hello relay"), "{assistant:?}");
    // Tool messages must carry the execution result, or history replay cannot show the output.
    let tool = messages
        .iter()
        .find(|message| message.role == MessageRole::Tool)
        .unwrap();
    assert!(tool.meta.as_ref().map(|meta| meta.done).unwrap_or(false));
}

#[tokio::test]
async fn waits_for_user_confirmation_before_executing() {
    let model = Arc::new(MockModel::scripted([
        plan_step(),
        exec_step(),
        step(verify_text()),
    ]));
    let (_temp, runtime) = runtime_with(model, false);

    let task_id = runtime
        .submit(TaskRequest::new("Generate a deliverable file"))
        .await
        .unwrap();
    wait_for(&runtime, task_id, |s| s == TaskState::AwaitingConfirmation).await;

    runtime.confirm(task_id, Confirmation::Approve).unwrap();
    let state = wait_for(&runtime, task_id, TaskState::is_terminal).await;
    assert_eq!(state, TaskState::Succeeded);
}

/// The user edits the plan by hand on the confirmation card: the edit is persisted and execution uses it.
#[tokio::test]
async fn manual_plan_edit_is_kept_and_used() {
    let model = Arc::new(MockModel::scripted([
        plan_step(),
        exec_step(),
        step(verify_text()),
    ]));
    let (_temp, runtime) = runtime_with(model.clone(), false);

    let task_id = runtime
        .submit(TaskRequest::new("Generate a deliverable file"))
        .await
        .unwrap();
    wait_for(&runtime, task_id, |s| s == TaskState::AwaitingConfirmation).await;

    let mut plan = runtime.task(task_id).unwrap().unwrap().plan.unwrap();
    plan.title = "Manually edited title".into();
    plan.summary =
        "Manually edited description: only write out.txt, leave every other file alone".into();
    plan.steps = vec![
        "Write out.txt".into(),
        "Check the contents before delivering".into(),
    ];
    plan.references = vec!["Checked attachment A.csv".into()];
    runtime.update_plan(task_id, plan).unwrap();

    let stored = runtime.task(task_id).unwrap().unwrap();
    assert_eq!(
        stored.title, "Manually edited title",
        "the title should be updated too"
    );
    assert_eq!(stored.plan.as_ref().unwrap().steps.len(), 2);

    runtime.confirm(task_id, Confirmation::Approve).unwrap();
    let state = wait_for(&runtime, task_id, TaskState::is_terminal).await;
    assert_eq!(state, TaskState::Succeeded);

    let done = runtime.task(task_id).unwrap().unwrap();
    assert_eq!(
        done.title, "Manually edited title",
        "still the manual version after execution"
    );
    assert_eq!(
        done.plan.as_ref().unwrap().summary,
        "Manually edited description: only write out.txt, leave every other file alone"
    );

    for role in ["# Relay Executor Agent", "# Relay Verifier Agent"] {
        let sent = model
            .requests()
            .into_iter()
            .find(|r| r.starts_with(role))
            .unwrap();
        assert!(sent.contains(
            "Manually edited description: only write out.txt, leave every other file alone"
        ));
        assert!(sent.contains("Checked attachment A.csv"));
        assert!(sent.contains("source-file.csv"));
    }

    // A finished task no longer accepts plan edits.
    let late = done.plan.clone().unwrap();
    assert!(runtime.update_plan(task_id, late).is_err());
}

/// The user says how to change it; after re-understanding, the plan changes and returns to the confirmation card.
#[tokio::test]
async fn user_feedback_triggers_a_revision() {
    let model = Arc::new(MockModel::scripted([
        plan_step(),
        MockStep::call(
            "submit_plan",
            serde_json::json!({
                "workspace_mode": "create",
                "workspace_name": "Deliverable",
                "workspace_keywords": ["deliverable", "checklist"],
                "session_mode": "create",
                "title": "Emit a checklist only",
                "summary": "As the user asked, only produce a checklist and leave the original files untouched.",
                "steps": ["Read the materials", "Produce the checklist"],
                "references": [],
                "delivery": "notify_only",
                "confidence": 0.8,
                "reason": "the user asked for a checklist only"
            }),
        ),
        exec_step(),
        step(verify_text()),
    ]));
    let (_temp, runtime) = runtime_with(model, false);

    let task_id = runtime
        .submit(TaskRequest::new("Generate a deliverable file"))
        .await
        .unwrap();
    wait_for(&runtime, task_id, |s| s == TaskState::AwaitingConfirmation).await;

    runtime
        .confirm(
            task_id,
            Confirmation::Revise("Only output a checklist, leave the original files alone".into()),
        )
        .unwrap();
    wait_for(&runtime, task_id, |_| {
        runtime
            .task(task_id)
            .ok()
            .flatten()
            .and_then(|task| task.plan)
            .map(|plan| plan.title == "Emit a checklist only")
            .unwrap_or(false)
    })
    .await;

    runtime.confirm(task_id, Confirmation::Approve).unwrap();
    let state = wait_for(&runtime, task_id, TaskState::is_terminal).await;
    assert_eq!(state, TaskState::Succeeded);
}

#[tokio::test]
async fn cancellation_stops_the_task() {
    let model = Arc::new(MockModel::scripted([
        step(plan_text()),
        MockStep::call("bash", serde_json::json!({ "command": "sleep 8" })),
        step(verify_text()),
    ]));
    let (_temp, runtime) = runtime_with(model, true);

    let task_id = runtime
        .submit(TaskRequest::new("Run a slow task"))
        .await
        .unwrap();
    wait_for(&runtime, task_id, |s| s == TaskState::Executing).await;
    runtime.cancel(task_id).unwrap();
    let state = wait_for(&runtime, task_id, TaskState::is_terminal).await;
    assert_eq!(state, TaskState::Cancelled);
}

#[tokio::test]
async fn model_failure_marks_task_failed() {
    let model = Arc::new(MockModel::new(Vec::<String>::new()));
    let (_temp, runtime) = runtime_with(model, true);
    let task_id = runtime.submit(TaskRequest::new("Any task")).await.unwrap();
    let state = wait_for(&runtime, task_id, TaskState::is_terminal).await;
    assert_eq!(state, TaskState::Failed);
    assert!(runtime.task(task_id).unwrap().unwrap().error.is_some());
}

#[tokio::test]
async fn understanding_reuses_session_and_tasks_run_serially() {
    // Once the first task finishes, the second task's understanding phase explicitly reuses its workspace and session.
    let model = Arc::new(MockModel::scripted([
        plan_step(),
        exec_step(),
        step(verify_text()),
    ]));
    let (_temp, runtime) = runtime_with(model.clone(), true);

    let first = runtime
        .submit(TaskRequest::new("Generate a deliverable file"))
        .await
        .unwrap();
    assert_eq!(
        wait_for(&runtime, first, TaskState::is_terminal).await,
        TaskState::Succeeded
    );
    let first_task = runtime.task(first).unwrap().unwrap();

    // Second task: understanding carries a reuse instruction, so it must land in the first task's session.
    model.push_step(reuse_plan_step(
        &first_task.workspace_id.to_string(),
        &first_task.session_id.to_string(),
    ));
    model.push_step(exec_step());
    model.push_step(step(verify_text()));

    let second = runtime
        .submit(TaskRequest::new("Generate a deliverable file"))
        .await
        .unwrap();
    let second_task = runtime.task(second).unwrap().unwrap();
    assert_eq!(
        first_task.session_id, second_task.session_id,
        "reusing a session in the understanding result must land in the same session"
    );
    assert_eq!(first_task.workspace_id, second_task.workspace_id);

    assert_eq!(
        wait_for(&runtime, second, TaskState::is_terminal).await,
        TaskState::Succeeded
    );
    let second_done = runtime.task(second).unwrap().unwrap();
    assert!(
        second_done.started_at >= first_task.finished_at,
        "tasks in one session must run serially: {:?} vs {:?}",
        second_done.started_at,
        first_task.finished_at
    );
    assert_eq!(
        runtime.recent_tasks(10).unwrap().len(),
        2,
        "both tasks should be persisted"
    );
}

#[tokio::test]
async fn confirmation_card_carries_permissions_and_references() {
    let model = Arc::new(MockModel::scripted([
        plan_step(),
        exec_step(),
        step(verify_text()),
    ]));
    let (_temp, runtime) = runtime_with(model, false);
    let mut events = runtime.subscribe();

    let task_id = runtime
        .submit(TaskRequest::new("Generate a deliverable file"))
        .await
        .unwrap();

    let mut seen = None;
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        if let Ok(event) = tokio::time::timeout(Duration::from_millis(200), events.recv()).await {
            if let Ok(event) = event {
                if let relay_core::RuntimeEventKind::ConfirmationRequested {
                    permissions,
                    references,
                    ..
                } = &event.kind
                {
                    seen = Some((permissions.clone(), references.clone()));
                    break;
                }
            }
        }
    }
    let (permissions, references) =
        seen.expect("a confirmation event carrying a permission summary should be emitted");
    assert!(permissions.contains("network blocked"), "{permissions}");
    assert!(permissions.contains("authorized dirs"), "{permissions}");
    assert!(
        references.iter().any(|entry| entry.contains("materials")),
        "a new workspace should show its directory layout: {references:?}"
    );

    runtime.confirm(task_id, Confirmation::Approve).unwrap();
    let state = wait_for(&runtime, task_id, TaskState::is_terminal).await;
    assert_eq!(state, TaskState::Succeeded);
}

#[tokio::test]
async fn restart_marks_interrupted_tasks_failed_but_keeps_paused() {
    use chrono::Utc;
    use relay_core::{DeliveryMode, EnvironmentSnapshot, SessionId, Task, TaskId, WorkspaceId};

    let temp = tempfile::tempdir().unwrap();
    let mut config = RuntimeConfig::new(temp.path().join("data"));
    config.policy.sandbox = false;
    config.notifications = false;
    config.ensure_dirs().unwrap();
    let store = Arc::new(Store::open(config.db_path()).unwrap());

    let build = |state: TaskState, title: &str| Task {
        id: TaskId::new(),
        workspace_id: WorkspaceId::new(),
        session_id: SessionId::new(),
        title: title.into(),
        instruction: title.into(),
        supplement: None,
        context: EnvironmentSnapshot::default(),
        usage: relay_core::ContextUsage::default(),
        plan: None,
        state,
        report: None,
        delivery: DeliveryMode::NotifyOnly,
        error: None,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        started_at: Some(Utc::now()),
        finished_at: None,
    };
    let interrupted = build(TaskState::Executing, "Running before the restart");
    let paused = build(TaskState::Paused, "Task the user paused");
    store.save_task(&interrupted).unwrap();
    store.save_task(&paused).unwrap();

    let runtime = Runtime::new(
        store,
        Arc::new(MockModel::new(Vec::<String>::new())),
        Arc::new(ManualContextProvider::new("Visual Studio Code", "relay")),
        config,
    )
    .unwrap();

    let recovered = runtime.task(interrupted.id).unwrap().unwrap();
    assert_eq!(recovered.state, TaskState::Failed);
    assert!(recovered.error.unwrap().contains("restart"));
    assert!(recovered.report.unwrap().verdict.is_some());

    let kept = runtime.task(paused.id).unwrap().unwrap();
    assert_eq!(
        kept.state,
        TaskState::Paused,
        "paused tasks should be kept so they can resume"
    );
}

#[tokio::test]
async fn resume_continues_a_paused_task_from_its_plan() {
    use chrono::Utc;
    use relay_core::{
        DeliveryMode, EnvironmentSnapshot, Plan, Session, SessionId, Task, TaskId, Workspace,
        WorkspaceId,
    };

    let model = Arc::new(MockModel::scripted([exec_step(), step(verify_text())]));
    let (temp, runtime) = runtime_with(model, true);
    let store = runtime.store();

    let workspace = Workspace {
        id: WorkspaceId::new(),
        name: "Resume test".into(),
        root_dir: temp.path().join("resume-workspace"),
        rules: String::new(),
        keywords: Vec::new(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
    };
    std::fs::create_dir_all(&workspace.root_dir).unwrap();
    store.save_workspace(&workspace).unwrap();

    let session = Session {
        id: SessionId::new(),
        workspace_id: workspace.id,
        title: "Resume session".into(),
        summary: String::new(),
        created_at: Utc::now(),
        last_active_at: Utc::now(),
        last_context: None,
    };
    store.save_session(&session).unwrap();

    let task = Task {
        id: TaskId::new(),
        workspace_id: workspace.id,
        session_id: session.id,
        title: "Continue after the pause".into(),
        instruction: "Generate the deliverable file".into(),
        supplement: None,
        context: EnvironmentSnapshot::default(),
        usage: relay_core::ContextUsage::default(),
        plan: Some(Plan {
            title: "Generate the deliverable".into(),
            summary: "Write out.txt".into(),
            steps: vec!["Write out.txt".into()],
            references: Vec::new(),
            delivery: DeliveryMode::NotifyOnly,
            raw: String::new(),
        }),
        state: TaskState::Paused,
        report: None,
        delivery: DeliveryMode::NotifyOnly,
        error: None,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        started_at: Some(Utc::now()),
        finished_at: None,
    };
    store.save_task(&task).unwrap();

    runtime.resume(task.id).await.unwrap();
    let state = wait_for(&runtime, task.id, TaskState::is_terminal).await;
    assert_eq!(
        state,
        TaskState::Succeeded,
        "{:?}",
        runtime.task(task.id).unwrap()
    );
}

#[tokio::test]
async fn workspace_keeps_layout_and_syncs_rules_file() {
    let model = Arc::new(MockModel::scripted([
        plan_step(),
        exec_step(),
        step(verify_text()),
        // Second task: understanding reuses the workspace by name, so the rules file should be re-synced.
        plan_step(),
        exec_step(),
        step(verify_text()),
    ]));
    let (_temp, runtime) = runtime_with(model, true);

    let first = runtime
        .submit(TaskRequest::new("Generate a deliverable file"))
        .await
        .unwrap();
    assert_eq!(
        wait_for(&runtime, first, TaskState::is_terminal).await,
        TaskState::Succeeded
    );

    let workspace = runtime.workspaces().unwrap().remove(0);
    for dir in ["materials", "outputs"] {
        assert!(
            workspace.root_dir.join(dir).is_dir(),
            "missing the {dir} directory"
        );
    }
    let rules_path = workspace.root_dir.join("rules.md");
    let rules = std::fs::read_to_string(&rules_path).unwrap();
    assert!(rules.contains("Workspace Rules"), "{rules}");

    // After the user edits the rules file, the next task should read the new rules.
    std::fs::write(
        &rules_path,
        "# Workspace Rules\n\n- Only write to outputs/\n",
    )
    .unwrap();
    let second = runtime
        .submit(TaskRequest::new("Generate another deliverable file"))
        .await
        .unwrap();
    assert_eq!(
        wait_for(&runtime, second, TaskState::is_terminal).await,
        TaskState::Succeeded
    );

    let synced = runtime.workspaces().unwrap().remove(0);
    assert!(
        synced.rules.contains("Only write to outputs/"),
        "{:?}",
        synced.rules
    );

    // The events and steps the control center relies on must be persisted and render into a readable log.
    assert!(!runtime.events(first, 20).unwrap().is_empty());
    assert!(!runtime.steps(first).unwrap().is_empty());
    let log = runtime.task_log(first).unwrap();
    assert!(log.iter().any(|line| line.contains("turn 1")), "{log:?}");
    assert!(log.iter().any(|line| line.contains("events:")), "{log:?}");
    assert!(
        log.iter().any(|line| line.contains("$ ")),
        "the log should contain the script's first line"
    );
}

#[tokio::test]
async fn context_budget_summarizes_history_with_model() {
    // Eight big outputs in one turn plus a tiny window: model summarization must fire repeatedly.
    let payload = "y".repeat(3_500);
    let mut script = vec![plan_step()];
    for _ in 0..8 {
        script.push(MockStep::call(
            "bash",
            serde_json::json!({ "command": format!("printf '%s' '{payload}'") }),
        ));
    }
    script.push(MockStep::call(
        "submit_result",
        serde_json::json!({"phase":"continue","summary":"continue in the next turn"}),
    ));
    script.push(MockStep::call(
        "submit_result",
        serde_json::json!({ "phase": "done", "summary": "all checks done" }),
    ));
    script.push(step(verify_text()));
    let model = Arc::new(MockModel::scripted(script));

    let temp = tempfile::tempdir().unwrap();
    let mut config = RuntimeConfig::new(temp.path().join("data"));
    config.policy.allow_dir(temp.path());
    config.policy.sandbox = false;
    config.policy.auto_open = false;
    config.notifications = false;
    config.auto_confirm = true;
    // Deliberately tiny window: compaction should kick in after a few big outputs instead of waiting for a real server overflow.
    // Room is reserved for the full tool definitions and output, yet what remains still cannot hold eight tool results.
    config.context_window = 12_000;
    let store = Arc::new(Store::open(config.db_path()).unwrap());
    let context = Arc::new(ManualContextProvider::new("Visual Studio Code", "relay"));
    let runtime = Runtime::new(store, model.clone(), context, config).unwrap();

    let task_id = runtime
        .submit(TaskRequest::new("Check the data"))
        .await
        .unwrap();
    let state = wait_for(&runtime, task_id, TaskState::is_terminal).await;
    let task = runtime.task(task_id).unwrap().unwrap();
    assert_eq!(state, TaskState::Succeeded, "{:?}", task.error);
    assert!(
        task.usage.compactions > 0,
        "compaction should be triggered: {:?}",
        task.usage
    );

    // The summarizer model really was called (first line of prompts/compact.md).
    let compact_prompt = include_str!("../../../prompts/compact.md");
    let compact_first_line = compact_prompt.lines().next().unwrap();
    let summarized = model
        .requests()
        .iter()
        .any(|request| request.contains(compact_first_line));
    assert!(
        summarized,
        "crossing the threshold should call the model to summarize"
    );

    // The main request after compaction carries the summary block instead of the raw history.
    let compacted = model
        .requests()
        .iter()
        .any(|request| request.contains(relay_runtime::budget::SUMMARY_HEADING));
    assert!(
        compacted,
        "later requests should carry the model's summary block"
    );

    let next_round = model
        .requests()
        .into_iter()
        .find(|r| r.starts_with("# Relay Executor Agent") && r.contains("2 / 24"))
        .expect("the next turn should be reached");
    assert!(
        next_round.contains(relay_runtime::budget::SUMMARY_HEADING) && next_round.contains("out.txt"),
        "the compacted request should carry the summary and still mention the artifact: {next_round}"
    );
    // No deterministic trimming: raw tool output is never rewritten into placeholder text.
    assert!(!model
        .requests()
        .iter()
        .any(|request| request.contains("[compacted]")));
}

#[tokio::test]
async fn executor_requests_keep_a_stable_prefix_for_cache() {
    // Cache hits follow prefixes: each request in a turn must be the previous text plus the newly appended tool entries.
    let mut script = vec![plan_step()];
    for command in ["echo one", "echo two", "echo three"] {
        script.push(MockStep::call(
            "bash",
            serde_json::json!({ "command": command }),
        ));
    }
    script.push(MockStep::call(
        "submit_result",
        serde_json::json!({ "phase": "done", "summary": "checks complete" }),
    ));
    script.push(step(verify_text()));
    let model = Arc::new(MockModel::scripted(script));
    let (_temp, runtime) = runtime_with(model.clone(), true);

    let task_id = runtime
        .submit(TaskRequest::new("Check the data"))
        .await
        .unwrap();
    let state = wait_for(&runtime, task_id, TaskState::is_terminal).await;
    assert_eq!(state, TaskState::Succeeded);

    let executor: Vec<String> = model
        .requests()
        .into_iter()
        .filter(|request| request.starts_with("# Relay Executor Agent"))
        .collect();
    assert!(
        executor.len() >= 3,
        "there should be several executor requests: {}",
        executor.len()
    );
    for pair in executor.windows(2) {
        assert!(
            pair[1].starts_with(&pair[0]),
            "later requests in a turn must fully contain the previous input, otherwise every cache hit is lost"
        );
    }
}

#[tokio::test]
async fn cancel_stops_a_running_task_quickly() {
    // The UI "Stop" button calls Runtime::cancel: it must cut off a running bash call.
    let model = Arc::new(MockModel::scripted([
        plan_step(),
        MockStep::call("bash", serde_json::json!({ "command": "sleep 30" })),
        step(verify_text()),
    ]));
    let (_temp, runtime) = runtime_with(model, true);

    let task_id = runtime
        .submit(TaskRequest::new("Long-running task"))
        .await
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let task = runtime.task(task_id).unwrap().unwrap();
        if task.state == TaskState::Executing {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the task never entered execution: {}",
            task.state
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    let started = Instant::now();
    runtime.cancel(task_id).unwrap();
    let state = wait_for(&runtime, task_id, TaskState::is_terminal).await;
    assert_eq!(state, TaskState::Cancelled);
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "stop should take effect immediately; it took {:?}",
        started.elapsed()
    );
}

/// Several commands in one turn: across turns every one of them must be remembered, not just the last output.
#[tokio::test]
async fn every_command_of_a_round_stays_in_transcript() {
    let model = Arc::new(MockModel::scripted([
        plan_step(),
        MockStep::call("bash", serde_json::json!({ "command": "printf 'AAA'" })),
        MockStep::call("bash", serde_json::json!({ "command": "printf 'BBB'" })),
        MockStep::call(
            "submit_result",
            serde_json::json!({ "phase": "continue", "summary": "continue to the next step" }),
        ),
        MockStep::call("bash", serde_json::json!({ "command": "printf 'CCC'" })),
        MockStep::call(
            "submit_result",
            serde_json::json!({ "phase": "done", "summary": "all three commands finished" }),
        ),
        step("VERDICT: PASS\nSUMMARY: checked\nCHECKS:\n- none"),
    ]));
    let (_temp, runtime) = runtime_with(model.clone(), true);
    let task_id = runtime
        .submit(TaskRequest::new("Check the data"))
        .await
        .unwrap();
    let state = wait_for(&runtime, task_id, TaskState::is_terminal).await;
    assert_eq!(state, TaskState::Succeeded);

    let requests = model.requests();
    let turn2 = requests
        .iter()
        .find(|request| request.contains("2 / 24"))
        .expect("the second turn should be reached");
    assert!(
        turn2.contains("AAA"),
        "the first command of the previous turn must not be lost: {turn2}"
    );
    assert!(
        turn2.contains("BBB"),
        "the last command of the previous turn must be there too: {turn2}"
    );
    assert!(
        turn2.contains("function_call_output"),
        "execution history should carry over between turns: {turn2}"
    );
}

/// Continuing a session: a new task must see what was said and produced earlier.
#[tokio::test]
async fn continuation_task_receives_earlier_conversation() {
    let model = Arc::new(MockModel::scripted([
        plan_step(),
        exec_step(),
        step(verify_text()),
        // Continuation task: a direct conclusion plus submit_result.
        step("out.txt was already generated last turn, so there is nothing to redo."),
        MockStep::call(
            "submit_result",
            serde_json::json!({
                "phase": "done",
                "summary": "reuse the out.txt generated last turn",
                "artifacts": ["out.txt"],
                "delivery": "notify_only"
            }),
        ),
    ]));
    let (_temp, runtime) = runtime_with(model.clone(), true);
    let first_id = runtime
        .submit(TaskRequest::new("Generate a deliverable file"))
        .await
        .unwrap();
    assert_eq!(
        wait_for(&runtime, first_id, TaskState::is_terminal).await,
        TaskState::Succeeded
    );
    let first = runtime.task(first_id).unwrap().unwrap();

    let config = runtime.config().clone();
    drop(runtime);
    let reopened = Arc::new(Store::open(config.db_path()).unwrap());
    let runtime = Runtime::new(
        reopened,
        model.clone(),
        Arc::new(ManualContextProvider::new("reopened", "another window")),
        config,
    )
    .unwrap();
    let mut request = TaskRequest::new("Continuing: does out.txt still need another line?");
    request.pinned_workspace = Some(first.workspace_id);
    request.pinned_session = Some(first.session_id);
    request.direct = true;
    request.auto_confirm = true;
    let second_id = runtime.submit(request).await.unwrap();
    assert_eq!(
        wait_for(&runtime, second_id, TaskState::is_terminal).await,
        TaskState::Succeeded
    );

    let wire: Vec<_> = model
        .payloads()
        .into_iter()
        .filter(|body| {
            body["instructions"]
                .as_str()
                .unwrap()
                .starts_with("# Relay Executor Agent")
        })
        .collect();
    let before = &wire[0];
    let after = &wire[1];
    assert_eq!(before["model"], after["model"]);
    assert_eq!(before["instructions"], after["instructions"]);
    assert_eq!(before["tools"], after["tools"]);
    let old_inputs = before["input"].as_array().unwrap();
    let new_inputs = after["input"].as_array().unwrap();
    assert_eq!(
        serde_json::to_vec(old_inputs).unwrap(),
        serde_json::to_vec(&new_inputs[..old_inputs.len()]).unwrap()
    );
    assert!(new_inputs[old_inputs.len()..]
        .iter()
        .any(|item| item["type"] == "function_call_output"));

    let requests = model.requests();
    let continued = requests
        .iter()
        .find(|request| request.contains("## Conversation history"))
        .unwrap_or_else(|| {
            panic!("a continuation task should carry the conversation history: {requests:#?}")
        });
    assert!(
        continued
            .to_lowercase()
            .contains("user: generate a deliverable file"),
        "the user's earlier wording should be kept: {continued}"
    );
    assert!(
        continued.contains("out.txt"),
        "the earlier conclusion and artifacts should be kept: {continued}"
    );
    // The conversation history is a cache prefix: it must come before the task description.
    assert!(continued.contains("task state: Completed"));
    assert!(continued.contains("recorded artifact: "));
    let history_at = continued.find("## Conversation history").unwrap();
    let task_at = continued.rfind("## Task").unwrap();
    assert!(
        history_at < task_at,
        "conversation history must come before the task description: {continued}"
    );
}

#[tokio::test]
async fn queued_message_stays_out_of_earlier_task_and_failure_reaches_next_task() {
    let model = Arc::new(MockModel::scripted([
        plan_step(),
        MockStep::call(
            "submit_result",
            serde_json::json!({"phase":"failed","summary":"the raw data is missing the date column"}),
        ),
        step(verify_text()),
        MockStep::call(
            "submit_result",
            serde_json::json!({"phase":"done","summary":"the reason has been explained"}),
        ),
    ]));
    let (_temp, runtime) = runtime_with(model.clone(), false);
    let first = runtime
        .submit(TaskRequest::new("Generate the report"))
        .await
        .unwrap();
    wait_for(&runtime, first, |s| s == TaskState::AwaitingConfirmation).await;
    let session = runtime.task(first).unwrap().unwrap().session_id;
    let mut followup = TaskRequest::new("request meant only for the next task");
    followup.direct = true;
    followup.auto_confirm = true;
    followup.pinned_session = Some(session);
    let second = runtime.submit(followup).await.unwrap();
    runtime.confirm(first, Confirmation::Approve).unwrap();
    assert_eq!(
        wait_for(&runtime, first, TaskState::is_terminal).await,
        TaskState::Failed
    );
    assert_eq!(
        wait_for(&runtime, second, TaskState::is_terminal).await,
        TaskState::Succeeded
    );
    let runs: Vec<_> = model
        .requests()
        .into_iter()
        .filter(|r| r.starts_with("# Relay Executor Agent"))
        .collect();
    assert_eq!(runs.len(), 2);
    assert!(!runs[0].contains("request meant only for the next task"));
    assert!(runs[1].contains("task state: Failed"));
    assert!(runs[1].contains("failure reason: the raw data is missing the date column"));
}

/// Simulate a provider rejecting the next request after the tool has already run.
#[tokio::test]
async fn provider_overflow_retries_inference_without_repeating_tool() {
    use relay_openai::{CompletionRequest, InputItem, LanguageModel, ModelError, ModelOutput};

    struct RejectToolHistory(MockModel);
    #[async_trait::async_trait]
    impl LanguageModel for RejectToolHistory {
        fn name(&self) -> &str {
            "window-test"
        }
        async fn complete(&self, request: CompletionRequest) -> relay_openai::Result<ModelOutput> {
            if request.instructions.starts_with("# Relay Executor Agent")
                && request
                    .input
                    .iter()
                    .any(|item| matches!(item, InputItem::Raw(_)))
            {
                return Err(ModelError::Api {
                    status: 400,
                    body: "context_length_exceeded".into(),
                });
            }
            self.0.complete(request).await
        }
    }

    let model = Arc::new(RejectToolHistory(MockModel::scripted([
        plan_step(),
        MockStep::call("bash", serde_json::json!({"command":"echo executed"})),
        MockStep::call(
            "submit_result",
            serde_json::json!({"phase":"done", "summary":"checked"}),
        ),
        step(verify_text()),
    ])));
    let (_temp, runtime) = runtime_with(model.clone(), true);
    let task = runtime
        .submit(TaskRequest::new("wrap up after checking"))
        .await
        .unwrap();
    assert_eq!(
        wait_for(&runtime, task, TaskState::is_terminal).await,
        TaskState::Succeeded
    );
    assert_eq!(runtime.steps(task).unwrap().len(), 1);
    let retried = model
        .0
        .requests()
        .into_iter()
        .find(|s| {
            s.starts_with("# Relay Executor Agent")
                && s.contains(relay_runtime::budget::SUMMARY_HEADING)
        })
        .unwrap();
    assert!(retried.contains("wrap up after checking"));
}

#[test]
fn paused_task_reopens_with_its_exact_last_input() {
    let model = Arc::new(MockModel::scripted([
        plan_step(),
        MockStep::call(
            "submit_result",
            serde_json::json!({"phase":"paused","summary":"waiting for review"}),
        ),
    ]));
    // Drop the entire async runtime so no worker, session gate or model state survives.
    let (_temp, config, task, last_input) =
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let (temp, runtime) = runtime_with(model.clone(), true);
            let task = runtime
                .submit(TaskRequest::new("Review and then finish"))
                .await
                .unwrap();
            wait_for(&runtime, task, |s| s == TaskState::Paused).await;
            let input = model.payloads().pop().unwrap()["input"].clone();
            (temp, runtime.config().clone(), task, input)
        });
    drop(model);
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let model = Arc::new(MockModel::scripted([
            MockStep::call(
                "submit_result",
                serde_json::json!({"phase":"done","summary":"review complete"}),
            ),
            step(verify_text()),
        ]));
        let reopened = Runtime::new(
            Arc::new(Store::open(config.db_path()).unwrap()),
            model.clone(),
            Arc::new(ManualContextProvider::default()),
            config,
        )
        .unwrap();
        reopened.resume(task).await.unwrap();
        assert_eq!(
            wait_for(&reopened, task, TaskState::is_terminal).await,
            TaskState::Succeeded
        );
        let requests = model.payloads();
        let next = requests[0]["input"].as_array().unwrap();
        let old = last_input.as_array().unwrap();
        assert_eq!(
            serde_json::to_vec(old).unwrap(),
            serde_json::to_vec(&next[..old.len()]).unwrap()
        );
        assert!(next[old.len()..]
            .iter()
            .any(|v| v["type"] == "function_call_output"));
    });
}
