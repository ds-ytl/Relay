//! Executor agent: several tools (bash / read_file / list_files / read_image) plus `submit_result` to wrap up;
//! the old Markdown + relay-state protocol is still accepted for compatibility.

use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::Utc;
use relay_context::ManualContextProvider;
use relay_core::{Session, SessionId, TaskState, Workspace, WorkspaceId};
use relay_openai::{MockModel, MockStep};
use relay_runtime::{Runtime, RuntimeConfig, TaskRequest};
use relay_storage::Store;
use serde_json::json;

fn step(text: &str) -> MockStep {
    MockStep::Text(text.to_string())
}

fn bash(command: &str) -> MockStep {
    MockStep::call(
        "bash",
        json!({ "command": command, "description": "test command" }),
    )
}

fn plan_step() -> MockStep {
    MockStep::call(
        "submit_plan",
        json!({
            "workspace_mode": "create",
            "workspace_name": "executor agent test",
            "session_mode": "create",
            "session_title": "Generate out.txt",
            "title": "Generate out.txt",
            "summary": "Write and verify out.txt.",
            "steps": ["Write out.txt", "Verify the contents"],
            "references": [],
            "delivery": "notify_only",
            "confidence": 0.9,
            "reason": "mock"
        }),
    )
}

fn verify_text() -> &'static str {
    "VERDICT: PASS\nSUMMARY: ok\nCHECKS:\n- out.txt exists\nDELIVER: out.txt"
}

fn runtime_with(model: Arc<MockModel>, temp: &tempfile::TempDir) -> Runtime {
    let mut config = RuntimeConfig::new(temp.path().join("data"));
    config.policy.allow_dir(temp.path());
    config.policy.sandbox = false;
    config.policy.auto_open = false;
    config.notifications = false;
    config.auto_confirm = true;
    let store = Arc::new(Store::open(config.db_path()).unwrap());
    let context = Arc::new(ManualContextProvider::new(
        "Visual Studio Code",
        "relay — executor",
    ));
    Runtime::new(store, model, context, config).unwrap()
}

async fn wait_terminal(runtime: &Runtime, task_id: relay_core::TaskId) -> TaskState {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let state = runtime.task(task_id).unwrap().unwrap().state;
        if state.is_terminal() {
            return state;
        }
        assert!(Instant::now() < deadline, "timed out: {state}");
        tokio::time::sleep(Duration::from_millis(80)).await;
    }
}

#[tokio::test]
async fn multiple_bash_calls_in_one_round() {
    let temp = tempfile::tempdir().unwrap();
    let model = Arc::new(MockModel::scripted([
        plan_step(),
        bash(r#"printf 'hi' > "$RELAY_WORKSPACE_DIR/out.txt""#),
        bash(
            r#"test -f "$RELAY_WORKSPACE_DIR/out.txt" && relay-state done --summary "out.txt generated" --artifact out.txt"#,
        ),
        step(verify_text()),
    ]));
    let runtime = runtime_with(model.clone(), &temp);

    let task_id = runtime
        .submit(TaskRequest::new("Write out.txt and verify it"))
        .await
        .unwrap();
    let state = wait_terminal(&runtime, task_id).await;
    let task = runtime.task(task_id).unwrap().unwrap();
    assert_eq!(
        state,
        TaskState::Succeeded,
        "error={:?} report={:?} steps={:?}",
        task.error,
        task.report,
        runtime
            .steps(task_id)
            .unwrap()
            .iter()
            .map(|s| (&s.script, s.exit_code, &s.stderr))
            .collect::<Vec<_>>()
    );

    // Both bash calls should be recorded as execution steps.
    let steps = runtime.steps(task_id).unwrap();
    assert_eq!(
        steps.len(),
        2,
        "both bash calls in one turn must be recorded: {steps:?}"
    );
    assert!(steps[0].script.contains("out.txt"));
    assert!(steps[1].script.contains("relay-state done"));

    // The second call must receive the first one's feedback (tool results flow back).
    let requests = model.requests();
    assert!(
        requests
            .iter()
            .any(|request| request.contains("function_call_output")),
        "bash results must flow back to the model"
    );
    // Executor turns carry the bash tool, understanding turns carry submit_plan.
    let tools = model.tool_names();
    assert!(tools
        .iter()
        .any(|names| names.contains(&"bash".to_string())));
    // Desktop tools are open to executor turns by default, not hidden behind a feature flag.
    for tool in ["computer_accessibility", "computer_input"] {
        assert!(
            tools
                .iter()
                .any(|names| names.iter().any(|name| name == tool)),
            "executor turns should expose the {tool} tool: {tools:#?}"
        );
    }
}

#[tokio::test]
async fn markdown_without_tools_does_not_count_as_script() {
    let temp = tempfile::tempdir().unwrap();
    let model = Arc::new(MockModel::scripted([
        plan_step(),
        // First round: markdown only, no command at all → should move on to the next round instead of failing.
        step("## This round\n\nI confirmed the target file first and will start writing shortly."),
        bash(r#"relay-state done --summary \"no file to write\""#),
        step(verify_text()),
    ]));
    let runtime = runtime_with(model, &temp);

    let task_id = runtime
        .submit(TaskRequest::new("Just take a look"))
        .await
        .unwrap();
    let state = wait_terminal(&runtime, task_id).await;
    let task = runtime.task(task_id).unwrap().unwrap();
    let steps = runtime.steps(task_id).unwrap();
    // There is no out.txt here and the mock verifier returns PASS; the point is that a markdown-only turn is not a failure.
    assert_eq!(
        state,
        TaskState::Succeeded,
        "error={:?} steps={:?} requests={}",
        task.error,
        steps
            .iter()
            .map(|s| (&s.script, s.exit_code, &s.stderr, s.phase))
            .collect::<Vec<_>>(),
        runtime.steps(task_id).map(|s| s.len()).unwrap_or(0)
    );

    let steps = runtime.steps(task_id).unwrap();
    assert_eq!(
        steps.len(),
        1,
        "a markdown-only turn should not produce execution steps"
    );
    assert!(steps[0].script.contains("relay-state done"));
}

#[tokio::test]
async fn unknown_tool_is_reported_back_to_the_model() {
    let temp = tempfile::tempdir().unwrap();
    let model = Arc::new(MockModel::scripted([
        plan_step(),
        MockStep::call("read_secret", json!({ "path": "/root/.ssh/id_rsa" })),
        bash(r#"relay-state done --summary "wrap up after the unknown tool" --artifact out.txt""#),
        step(verify_text()),
    ]));
    let runtime = runtime_with(model.clone(), &temp);

    let task_id = runtime
        .submit(TaskRequest::new("Probe an unknown tool"))
        .await
        .unwrap();
    wait_terminal(&runtime, task_id).await;
    let requests = model.requests();
    assert!(
        requests
            .iter()
            .any(|request| request.contains("unknown tool: read_secret")),
        "unknown tools should be rejected explicitly: {requests:#?}"
    );
}

#[tokio::test]
async fn submit_result_finishes_round_and_verifier_uses_tools() {
    let temp = tempfile::tempdir().unwrap();
    let model = Arc::new(MockModel::scripted([
        plan_step(),
        bash(r#"printf 'hi' > "$RELAY_WORKSPACE_DIR/out.txt""#),
        // Executor turn: read-only checks first, then wrap up with submit_result (no more relay-state).
        MockStep::call("list_files", json!({})),
        MockStep::call(
            "submit_result",
            json!({
                "phase": "done",
                "summary": "out.txt generated",
                "artifacts": ["out.txt"],
                "delivery": "notify_only"
            }),
        ),
        // Verifier turn: read-only checks of the artifacts, then a structured verdict.
        MockStep::call("list_files", json!({})),
        MockStep::call(
            "submit_verification",
            json!({
                "verdict": "pass",
                "summary": "out.txt was generated",
                "checks": ["out.txt exists", "contents are hi"],
                "deliver": "out.txt"
            }),
        ),
    ]));
    let runtime = runtime_with(model.clone(), &temp);

    let task_id = runtime
        .submit(TaskRequest::new("Write out.txt and verify the result"))
        .await
        .unwrap();
    let state = wait_terminal(&runtime, task_id).await;
    let task = runtime.task(task_id).unwrap().unwrap();
    assert_eq!(
        state,
        TaskState::Succeeded,
        "error={:?} report={:?}",
        task.error,
        task.report
    );

    let requests = model.requests();
    assert!(
        requests
            .iter()
            .any(|request| request.contains("submit_result")),
        "executor turns should expose submit_result: {requests:#?}"
    );
    let tool_names = model.tool_names();
    assert!(
        tool_names
            .iter()
            .any(|names| names.iter().any(|name| name == "submit_result")),
        "executor turns should carry the submit_result tool: {tool_names:#?}"
    );
    assert!(
        tool_names
            .iter()
            .any(|names| names.iter().any(|name| name == "submit_verification")),
        "the verifier turn should carry the submit_verification tool: {tool_names:#?}"
    );
}

#[tokio::test]
async fn session_message_runs_executor_directly() {
    let temp = tempfile::tempdir().unwrap();
    let mut config = RuntimeConfig::new(temp.path().join("data"));
    config.policy.allow_dir(temp.path());
    config.policy.sandbox = false;
    config.policy.auto_open = false;
    config.notifications = false;
    config.auto_confirm = true;
    let store = Arc::new(Store::open(config.db_path()).unwrap());

    let workspace = Workspace {
        id: WorkspaceId::new(),
        name: "Session direct".into(),
        root_dir: temp.path().join("ws"),
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
        title: "Conversation".into(),
        summary: String::new(),
        created_at: Utc::now(),
        last_active_at: Utc::now(),
        last_context: None,
    };
    store.save_session(&session).unwrap();

    // Only the executor's two steps: act, then submit_result; no understanding or verification calls.
    let model = Arc::new(MockModel::scripted([
        bash(r#"printf 'hi' > "$RELAY_WORKSPACE_DIR/out.txt""#),
        MockStep::call(
            "submit_result",
            json!({
                "phase": "done",
                "summary": "out.txt generated",
                "artifacts": ["out.txt"]
            }),
        ),
    ]));
    let context = Arc::new(ManualContextProvider::new(
        "Visual Studio Code",
        "relay — chat",
    ));
    let runtime = Runtime::new(store, model.clone(), context, config).unwrap();

    let task_id = runtime
        .submit(TaskRequest {
            attachments: Vec::new(),
            instruction: "Write out.txt".into(),
            supplement: None,
            pinned_workspace: Some(workspace.id),
            pinned_session: Some(session.id),
            auto_confirm: true,
            direct: true,
        })
        .await
        .unwrap();
    let state = wait_terminal(&runtime, task_id).await;
    assert_eq!(
        state,
        TaskState::Succeeded,
        "{:?}",
        runtime.task(task_id).unwrap()
    );

    let names = model.tool_names();
    assert!(
        !names
            .iter()
            .any(|set| set.iter().any(|name| name == "submit_plan")),
        "a direct session must not call the understanding agent: {names:#?}"
    );
    assert!(
        !names
            .iter()
            .any(|set| set.iter().any(|name| name == "submit_verification")),
        "a direct session must not call the verifier agent: {names:#?}"
    );
    assert!(
        names
            .iter()
            .any(|set| set.iter().any(|name| name == "submit_result")),
        "the turn should go straight to the executor agent: {names:#?}"
    );
}

/// When the executor decides to pause and wait for more input, the turn must end and release the session lock,
/// otherwise the next message stays stuck in the queue.
#[tokio::test]
async fn paused_turn_ends_and_next_message_runs() {
    let temp = tempfile::tempdir().unwrap();
    let mut config = RuntimeConfig::new(temp.path().join("data"));
    config.policy.allow_dir(temp.path());
    config.policy.sandbox = false;
    config.policy.auto_open = false;
    config.notifications = false;
    config.auto_confirm = true;
    let store = Arc::new(Store::open(config.db_path()).unwrap());

    let workspace = Workspace {
        id: WorkspaceId::new(),
        name: "Session direct".into(),
        root_dir: temp.path().join("ws"),
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
        title: "Conversation".into(),
        summary: String::new(),
        created_at: Utc::now(),
        last_active_at: Utc::now(),
        last_context: None,
    };
    store.save_session(&session).unwrap();

    let model = Arc::new(MockModel::scripted([
        // First round: there is nothing to do, so the executor submits a pause.
        MockStep::call(
            "submit_result",
            json!({ "phase": "paused", "summary": "waiting for you to add requirements", "artifacts": [] }),
        ),
        // Second round: the same session continues and completes normally.
        bash(r#"printf 'hi' > "$RELAY_WORKSPACE_DIR/out.txt""#),
        MockStep::call(
            "submit_result",
            json!({ "phase": "done", "summary": "out.txt generated", "artifacts": ["out.txt"] }),
        ),
    ]));
    let context = Arc::new(ManualContextProvider::new(
        "Visual Studio Code",
        "relay — chat",
    ));
    let runtime = Runtime::new(store, model, context, config).unwrap();

    let request = |text: &str| TaskRequest {
        attachments: Vec::new(),
        instruction: text.into(),
        supplement: None,
        pinned_workspace: Some(workspace.id),
        pinned_session: Some(session.id),
        auto_confirm: true,
        direct: true,
    };

    let first = runtime.submit(request("Hello")).await.unwrap();
    assert_eq!(wait_terminal(&runtime, first).await, TaskState::Succeeded);

    let second = runtime.submit(request("Write out.txt")).await.unwrap();
    assert_eq!(wait_terminal(&runtime, second).await, TaskState::Succeeded);
    assert!(temp.path().join("ws/out.txt").exists());
}

/// A bash fence inside the reply is no longer executed as a fallback: without a structured tool call the turn just ends.
#[tokio::test]
async fn bash_fence_in_reply_is_not_executed() {
    let temp = tempfile::tempdir().unwrap();
    let model = Arc::new(MockModel::scripted([
        plan_step(),
        step("Here is how I would write it:\n```bash\nprintf 'pwned' > \"$RELAY_WORKSPACE_DIR/fence.txt\"\n```"),
        bash(r#"relay-state done --summary "the fence in the reply was not executed""#),
        step(verify_text()),
    ]));
    let runtime = runtime_with(model, &temp);
    let task_id = runtime
        .submit(TaskRequest::new("Write a file"))
        .await
        .unwrap();
    let state = wait_terminal(&runtime, task_id).await;
    let steps = runtime.steps(task_id).unwrap();
    assert!(
        !steps.iter().any(|step| step.script.contains("fence.txt")),
        "a code fence in the reply must not be executed: {:?}",
        steps.iter().map(|step| &step.script).collect::<Vec<_>>()
    );
    let task = runtime.task(task_id).unwrap().unwrap();
    assert_eq!(state, TaskState::Succeeded, "error={:?}", task.error);
}

#[tokio::test]
async fn search_and_fetch_tools_use_the_configured_endpoint() {
    use std::sync::Mutex;

    use relay_openai::WebTools;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    let temp = tempfile::tempdir().unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let captured: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let captured_clone = captured.clone();

    tokio::spawn(async move {
        for _ in 0..2 {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buffer = vec![0u8; 64 * 1024];
            let read = socket.read(&mut buffer).await.unwrap();
            let request = String::from_utf8_lossy(&buffer[..read]).to_string();
            let body = if request.contains("/tools/search") {
                r#"{"results":[{"rerank_score":2.2,"title":"Search hit","url":"https://example.com","snippet":"s","summary":"summary text","content":"","site_name":"Example","site_icon":null,"date":null,"authority":3}]}"#
            } else {
                r##"{"title":"Fetched page","content":"# Hello\n\nbody text"}"##
            };
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\nconnection: close\r\ncontent-length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            socket.write_all(response.as_bytes()).await.unwrap();
            captured_clone.lock().unwrap().push(request);
        }
    });

    let model = Arc::new(MockModel::scripted([
        plan_step(),
        MockStep::call("search", json!({"query": "relay web tools", "count": 1})),
        MockStep::call("fetch", json!({"url": "https://example.com"})),
        MockStep::call(
            "submit_result",
            json!({"phase": "done", "summary": "Read the search results and the page."}),
        ),
        step(verify_text()),
    ]));

    let mut config = RuntimeConfig::new(temp.path().join("data"));
    config.policy.allow_dir(temp.path());
    config.notifications = false;
    config.auto_confirm = true;
    config.web_tools = Some(Arc::new(WebTools::new(
        "test-key",
        format!("http://{addr}/v1"),
    )));
    let store = Arc::new(Store::open(config.db_path()).unwrap());
    let context = Arc::new(ManualContextProvider::new("Safari", "relay — web tools"));
    let runtime = Runtime::new(store, model.clone(), context, config).unwrap();

    let task_id = runtime
        .submit(TaskRequest::new(
            "Look up relay web tools and read the first result",
        ))
        .await
        .unwrap();
    let state = wait_terminal(&runtime, task_id).await;
    let task = runtime.task(task_id).unwrap().unwrap();
    assert_eq!(state, TaskState::Succeeded, "error={:?}", task.error);

    let requests = captured.lock().unwrap().clone();
    assert_eq!(
        requests.len(),
        2,
        "both web tools must be called: {requests:?}"
    );
    assert!(
        requests[0].starts_with("POST /v1/tools/search "),
        "{}",
        requests[0]
    );
    assert!(requests[0].contains("relay web tools"), "{}", requests[0]);
    assert!(
        requests[0].contains("authorization: Bearer test-key")
            || requests[0].contains("Authorization: Bearer test-key"),
        "{}",
        requests[0]
    );
    assert!(
        requests[1].starts_with("POST /v1/tools/fetch "),
        "{}",
        requests[1]
    );

    let output = model.requests().join("\n");
    assert!(
        output.contains("Search hit"),
        "search results must flow back"
    );
    assert!(
        output.contains("Fetched page"),
        "fetched page must flow back"
    );
}
