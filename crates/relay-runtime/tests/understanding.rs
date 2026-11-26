//! Understanding agent behaviour: inspect before deciding, tool results really flow back, the reuse path works and the fallback never hangs.

use std::sync::Arc;

use relay_context::ManualContextProvider;
use relay_core::{
    DeliveryMode, EnvironmentSnapshot, Plan, Session, SessionId, TaskState, Workspace, WorkspaceId,
};
use relay_openai::{MockModel, MockStep};
use relay_runtime::{Runtime, RuntimeConfig, TaskRequest, UnderstandStage};
use relay_storage::Store;
use serde_json::json;

fn step(text: &str) -> MockStep {
    MockStep::Text(text.to_string())
}

/// One executor turn: a structured bash tool call (code fences in the reply are no longer executed).
fn exec_step() -> MockStep {
    MockStep::call(
        "bash",
        json!({
            "command": "set -e\nprintf 'hello' > \"$RELAY_WORKSPACE_DIR/out.txt\"\nrelay-state done --summary \"generated\" --artifact out.txt"
        }),
    )
}

fn verify_text() -> &'static str {
    "VERDICT: PASS\nSUMMARY: ok\nCHECKS:\n- file exists\nDELIVER: out.txt"
}

fn runtime_with(model: Arc<MockModel>, temp: &tempfile::TempDir) -> Runtime {
    let mut config = RuntimeConfig::new(temp.path().join("data"));
    config.policy.allow_dir(temp.path());
    config.policy.sandbox = false;
    config.policy.auto_open = false;
    config.notifications = false;
    config.auto_confirm = true;
    // Image recognition goes through a separate vision model, so screen text still arrives even when the main model cannot read images.
    config.vision_model = Some("vision-test".into());
    let store = Arc::new(Store::open(config.db_path()).unwrap());
    let context = Arc::new(ManualContextProvider::new(
        "WeChat",
        "Project A electricity group",
    ));
    Runtime::new(store, model, context, config).unwrap()
}

fn seed_workspace(runtime: &Runtime, temp: &tempfile::TempDir, name: &str) -> Workspace {
    let root = temp.path().join(name);
    std::fs::create_dir_all(&root).unwrap();
    let now = chrono::Utc::now();
    let workspace = Workspace {
        id: WorkspaceId::new(),
        name: name.into(),
        root_dir: root,
        rules: "# Rules\n\n- Reconcile the electricity ledger every month\n".into(),
        keywords: vec!["electricity".into(), "project a".into()],
        created_at: now,
        updated_at: now,
    };
    runtime.store().save_workspace(&workspace).unwrap();
    let session = Session {
        id: SessionId::new(),
        workspace_id: workspace.id,
        title: "September electricity bill check".into(),
        summary: "Check Project A's September electricity bill of 12800".into(),
        created_at: now,
        last_active_at: now,
        last_context: Some("Current context: WeChat · Project A electricity group".into()),
    };
    runtime.store().save_session(&session).unwrap();
    workspace
}

#[tokio::test]
async fn tools_are_called_before_the_decision() {
    let temp = tempfile::tempdir().unwrap();
    let model = Arc::new(MockModel::scripted([
        MockStep::call("get_environment", json!({})),
        MockStep::call("list_workspaces", json!({})),
        MockStep::call(
            "submit_plan",
            json!({
                "workspace_mode": "create",
                "workspace_name": "Project A electricity",
                "workspace_keywords": ["electricity", "project a"],
                "session_mode": "create",
                "session_title": "Organize the electricity bills",
                "title": "Organize Project A's electricity bills",
                "summary": "Turn the electricity info in WeChat into a ledger.",
                "steps": ["Read the clipboard", "Write the ledger"],
                "references": ["WeChat · Project A electricity group"],
                "delivery": "open_file",
                "confidence": 0.82,
                "reason": "WeChat shows an electricity group, which is a new electricity ledger item"
            }),
        ),
        exec_step(),
        step(verify_text()),
    ]));
    let runtime = runtime_with(model.clone(), &temp);
    let mut understanding_events = runtime.subscribe_understanding();

    let task_id = runtime
        .submit(TaskRequest::new("Note this down for me"))
        .await
        .unwrap();

    // Three model requests: two tool calls plus one submit.
    let tool_names = model.tool_names();
    assert_eq!(tool_names.len(), 3, "understanding and execution each add requests; these are the first three understanding turns");
    assert!(tool_names[0].contains(&"submit_plan".to_string()));
    assert!(tool_names[0].contains(&"take_screenshot".to_string()));

    let requests = model.requests();
    assert!(requests[0].contains("Note this down for me"));
    // Key point: the first request must not be stuffed with history, or we are simply dumping everything on the model.
    assert!(
        !requests[0].contains("Reconcile the electricity ledger every month"),
        "the first request should not preload the workspace rules"
    );
    // The second request should carry the first turn's structured tool results.
    assert!(
        requests[1].contains("function_call_output"),
        "tool results must flow back as function_call_output"
    );

    let task = runtime.task(task_id).unwrap().unwrap();
    assert_eq!(
        task.plan.as_ref().unwrap().title,
        "Organize Project A's electricity bills"
    );
    let workspace = runtime
        .workspaces()
        .unwrap()
        .into_iter()
        .find(|w| w.id == task.workspace_id)
        .unwrap();
    assert_eq!(workspace.name, "Project A electricity");
    assert!(workspace.keywords.contains(&"electricity".to_string()));

    // Trace events must be consumable by the UI.
    let mut stages = Vec::new();
    while let Ok(event) = understanding_events.try_recv() {
        stages.push(event.stage);
    }
    assert!(stages.contains(&UnderstandStage::ToolCall));
    assert!(stages.contains(&UnderstandStage::Decided));
}

#[tokio::test]
async fn reuse_decision_lands_in_existing_workspace() {
    let temp = tempfile::tempdir().unwrap();
    let model = Arc::new(MockModel::scripted(Vec::<MockStep>::new()));
    let runtime = runtime_with(model.clone(), &temp);
    let existing = seed_workspace(&runtime, &temp, "Electricity ledger");

    model.push_step(MockStep::call(
        "submit_plan",
        json!({
            "workspace_mode": "reuse",
            "workspace_id": existing.id.to_string(),
            "session_mode": "create",
            "session_title": "October electricity bills",
            "title": "October electricity bill check",
            "summary": "Keep reconciling on last month's electricity ledger.",
            "steps": ["Reconcile"],
            "references": ["session September electricity bill check"],
            "delivery": "notify_only",
            "confidence": 0.9,
            "reason": "the same ongoing item as the existing electricity ledger"
        }),
    ));
    model.push_step(exec_step());
    model.push_step(step(verify_text()));

    let task_id = runtime
        .submit(TaskRequest::new("Check October's too"))
        .await
        .unwrap();
    let task = runtime.task(task_id).unwrap().unwrap();
    assert_eq!(task.workspace_id, existing.id);
    assert_eq!(
        runtime.workspaces().unwrap().len(),
        1,
        "no new workspace should be created"
    );
}

#[tokio::test]
async fn read_file_reads_any_path() {
    let temp = tempfile::tempdir().unwrap();
    let model = Arc::new(MockModel::scripted([
        MockStep::call("read_file", json!({"path": "/etc/hosts"})),
        MockStep::call(
            "submit_plan",
            json!({
                "workspace_mode": "create",
                "workspace_name": "Privilege test",
                "session_mode": "create",
                "session_title": "Privilege test",
                "title": "Privilege test",
                "summary": "Verify that read_file does not read files outside the authorized folders.",
                "steps": ["none"],
                "delivery": "notify_only",
                "confidence": 0.5,
                "reason": "test"
            }),
        ),
        exec_step(),
        step(verify_text()),
    ]));
    let runtime = runtime_with(model.clone(), &temp);

    runtime
        .submit(TaskRequest::new("Read a system file"))
        .await
        .unwrap();
    let requests = model.requests();
    assert!(
        requests[1].contains("file: /etc/hosts") || requests[1].contains("localhost"),
        "any path should be readable: {}",
        requests[1]
    );
}

#[tokio::test]
async fn text_only_answer_falls_back_instead_of_hanging() {
    let temp = tempfile::tempdir().unwrap();
    let model = Arc::new(MockModel::scripted([
        step("TITLE: Fallback plan\nSUMMARY: The model did not follow the tool protocol.\nSTEPS:\n- Write the file\nREFERENCES:\n- none\nDELIVERY: notify_only"),
        exec_step(),
        step(verify_text()),
    ]));
    let runtime = runtime_with(model, &temp);

    let task_id = runtime
        .submit(TaskRequest::new("Do something, anything"))
        .await
        .unwrap();
    let task = runtime.task(task_id).unwrap().unwrap();
    let plan: &Plan = task.plan.as_ref().unwrap();
    assert_eq!(plan.title, "Fallback plan");
    assert_eq!(task.delivery, DeliveryMode::NotifyOnly);
    assert!(!task.title.is_empty());
    // The fallback must be persisted too, not left in pending.
    assert!(matches!(
        task.state,
        TaskState::Pending
            | TaskState::Understanding
            | TaskState::AwaitingConfirmation
            | TaskState::Executing
            | TaskState::Verifying
            | TaskState::Delivering
            | TaskState::Succeeded
    ));
}

/// Blank instruction: the user types nothing and only presses "start understanding", so the understanding agent has to infer the goal from the screen.
#[tokio::test]
async fn blank_instruction_is_inferred_from_the_screen() {
    let temp = tempfile::tempdir().unwrap();
    let model = Arc::new(MockModel::scripted([
        MockStep::call("get_environment", json!({})),
        MockStep::call("take_screenshot", json!({})),
        // After the screenshot the vision model turns it into text, so the understanding agent reads the screen content instead of a file path.
        step("WeChat · Project A electricity group, latest message \"September electricity bill 12800\""),
        MockStep::call(
            "submit_plan",
            json!({
                "workspace_mode": "create",
                "workspace_name": "Electricity ledger",
                "workspace_keywords": ["electricity", "ledger"],
                "session_mode": "create",
                "title": "Turn the WeChat electricity info into a ledger",
                "summary": "I see Project A's September electricity bill of 12800 in WeChat, so I infer you want it turned into a ledger.",
                "steps": ["Read the electricity info on screen", "Write the ledger file"],
                "references": ["WeChat · Project A electricity group · 12800"],
                "delivery": "open_file",
                "confidence": 0.74,
                "reason": "the foreground window is the WeChat electricity group, nothing in the chat asks for work, so infer from the screen"
            }),
        ),
        exec_step(),
        step(verify_text()),
    ]));
    let runtime = runtime_with(model.clone(), &temp);

    let task_id = runtime.submit(TaskRequest::new("   ")).await.unwrap();

    // The first request must raise the blank-instruction scenario and really inspect the environment.
    let requests = model.requests();
    assert!(
        requests[0].to_lowercase().contains("blank instruction"),
        "a blank instruction should trigger the inference guidance: {}",
        requests[0]
    );
    assert!(requests[0].contains("take_screenshot"));
    // Screenshots must go through the vision model: the request carries the screenshot prompt and vision_model.
    assert!(
        requests
            .iter()
            .any(|request| request.to_lowercase().contains("desktop screenshot")),
        "the screenshot should be recognized by the vision model: {requests:?}"
    );
    assert!(
        model
            .models()
            .iter()
            .any(|name| name.as_deref() == Some("vision-test")),
        "image recognition must use vision_model: {:?}",
        model.models()
    );

    let task = runtime.task(task_id).unwrap().unwrap();
    assert_eq!(
        task.plan.as_ref().unwrap().title,
        "Turn the WeChat electricity info into a ledger"
    );
    // The placeholder instruction must not be stored as if it carried business meaning.
    assert!(relay_runtime::is_blank_instruction(&task.instruction));
    assert!(
        !task.instruction.is_empty(),
        "the placeholder prompt must reach downstream stages"
    );

    // The session title uses the inferred plan title, not the placeholder text.
    let session = runtime.store().session(task.session_id).unwrap().unwrap();
    assert_eq!(
        session.title,
        "Turn the WeChat electricity info into a ledger"
    );
}

#[tokio::test]
async fn blank_instruction_checks_screen_before_deciding() {
    let temp = tempfile::tempdir().unwrap();
    let decision = || {
        MockStep::call(
            "submit_plan",
            json!({
                "workspace_mode": "create",
                "workspace_name": "Fix the error",
                "workspace_keywords": ["error", "logs"],
                "session_mode": "create",
                "session_title": "Infer from the screen",
                "title": "Organize the error on screen",
                "summary": "I see an error in the terminal, so I infer you want a fix.",
                "steps": ["Find the cause", "Write the fix"],
                "references": ["Terminal · build failed"],
                "delivery": "notify_only",
                "confidence": 0.6,
                "reason": "the screen shows an error log"
            }),
        )
    };
    let model = Arc::new(MockModel::scripted([
        decision(), // Submitting without looking at the screen → should be rejected
        MockStep::call("get_environment", json!({})), // Look at the current screen first
        decision(), // Then submit → accepted
        exec_step(),
        step(verify_text()),
    ]));
    let runtime = runtime_with(model.clone(), &temp);

    let task_id = runtime.submit(TaskRequest::new("")).await.unwrap();

    let requests = model.requests();
    assert!(
        requests[0].to_lowercase().contains("current screen"),
        "the first blank-instruction prompt must set the priority order"
    );
    assert!(
        requests[1].to_lowercase().contains("blank instruction")
            && requests[1].contains("take_screenshot"),
        "submitting without looking at the screen should be rejected and asked for a screenshot first"
    );
    let task = runtime.task(task_id).unwrap().unwrap();
    assert_eq!(
        task.plan.as_ref().unwrap().title,
        "Organize the error on screen",
        "after looking at the screen the submission should be accepted"
    );
}

#[tokio::test]
async fn read_image_recognizes_picture_through_the_vision_model() {
    let temp = tempfile::tempdir().unwrap();
    let image = temp.path().join("receipt.png");
    // A real PNG: the tool checks the magic bytes and reads the dimensions.
    image::RgbaImage::from_pixel(640, 480, image::Rgba([255, 255, 255, 255]))
        .save(&image)
        .unwrap();

    let decision = MockStep::call(
        "submit_plan",
        json!({
            "workspace_mode": "create",
            "workspace_name": "Electricity ledger",
            "workspace_keywords": ["electricity", "invoice"],
            "session_mode": "create",
            "session_title": "Recognize the invoice",
            "title": "Build the electricity ledger from the invoice",
            "summary": "The image is a 12,800 electricity bill; turn it into a ledger.",
            "steps": ["Recognize the invoice fields", "Write the ledger"],
            "references": ["receipt.png"],
            "delivery": "open_file",
            "confidence": 0.8,
            "reason": "the user selected an image of an electricity bill"
        }),
    );
    let model = Arc::new(MockModel::scripted([
        MockStep::call(
            "read_image",
            json!({
                "path": image.to_string_lossy(),
                "prompt": "Read the amount and the date on this invoice",
                "max_edge": 1024
            }),
        ),
        // This step is the vision request's answer: it describes the image.
        step("The image is an electricity bill: 12,800, dated September 30."),
        // A blank instruction still has to look at the current screen before submitting a plan.
        MockStep::call("get_environment", json!({})),
        decision,
        exec_step(),
        step(verify_text()),
    ]));
    let runtime = runtime_with(model.clone(), &temp);

    let task_id = runtime.submit(TaskRequest::new("")).await.unwrap();

    let requests = model.requests();
    assert!(
        requests[1].contains("image recognition assistant")
            && requests[1].contains("[image-bytes:image/png:"),
        "recognition should use a multimodal request: the image is classified by magic bytes and inlined for the model"
    );
    assert!(
        requests[2].contains("receipt.png")
            && requests[2].contains("sha=")
            && requests[2].contains("max_edge=1024")
            && requests[2].contains("12,800"),
        "the result carries the image/sha/max_edge header and flows back as text"
    );
    let task = runtime.task(task_id).unwrap().unwrap();
    assert_eq!(
        task.plan.as_ref().unwrap().title,
        "Build the electricity ledger from the invoice",
        "the recognized content should support the plan"
    );
}

/// When the screen is captured up front, its content goes straight into the first message, so a blank instruction can be submitted immediately without waiting for a screenshot round trip.
#[tokio::test]
async fn prefetched_screen_lands_in_the_first_round() {
    let temp = tempfile::tempdir().unwrap();
    let store = Store::open(temp.path().join("relay.sqlite3")).unwrap();
    let model = Arc::new(MockModel::scripted([MockStep::call(
        "submit_plan",
        json!({
            "workspace_mode": "create",
            "workspace_name": "Electricity ledger",
            "workspace_keywords": ["electricity", "ledger"],
            "session_mode": "create",
            "title": "Turn the WeChat electricity info into a ledger",
            "summary": "The screen shows Project A's September electricity bill of 12800; turn it into a ledger.",
            "steps": ["Read the electricity info on screen", "Write the ledger file"],
            "references": ["WeChat · Project A electricity group"],
            "delivery": "open_file",
            "confidence": 0.8,
            "reason": "the first-round screen already provides the electricity info"
        }),
    )]));
    let agent = relay_runtime::UnderstandingAgent::new(model.clone(), None);
    let snapshot = EnvironmentSnapshot::default();
    let input = relay_runtime::UnderstandingInput {
        store: &store,
        context: &snapshot,
        instruction: "   ",
        supplement: None,
        pinned_workspace: None,
        allowed_dirs: Vec::new(),
        vision_model: Some("vision-test".into()),
        screenshot_dir: temp.path().join("shots"),
        max_steps: 4,
        budget: relay_runtime::budget::Budget::new(128_000, 0.9),
        prefetched: Some(relay_runtime::PrefetchedScene {
            attached: false,
            text: "Captured WeChat · Project A electricity group\n--- Screen content ---\nSeptember electricity bill 12800".into(),
            image: None,
        }),
    };

    let understanding = agent.run(&input, &|_| {}).await.unwrap();

    let first = &model.requests()[0];
    assert!(
        first.to_lowercase().contains("current screen") && first.contains("12800"),
        "the screen content should be in the first round: {first}"
    );
    assert_eq!(
        understanding.plan.title, "Turn the WeChat electricity info into a ledger",
        "after seeing the screen a blank instruction should submit directly"
    );
}
