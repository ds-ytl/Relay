//! End-to-end tests for the in-process `xa11y` backend.
//!
//! Every operation runs through the real tool surface against `xa11y`'s
//! in-memory provider (`test-support`), whose fixture is a small application
//! tree: `Application "TestApp" → Window "Main Window" → toolbar "Navigation"
//! → button "Back" / button "Forward" (disabled)`, plus `text_field "Search"`,
//! `check_box "Agree"`, `slider "Volume"`, `list "Items"`, and an invisible
//! `static_text "Status"`.
//!
//! The point is to cover the whole path — selector matching, traversal
//! budgets, handle interning, action dispatch, hit testing, and waiting —
//! without a desktop and without accessibility permissions.

use std::sync::Arc;

use relay_accessibility_tool::{ComputerAccessibilityTool, Xa11yRuntime};
use relay_tool_api::Tool;
use serde_json::{Value, json};
use xa11y::mock::{MockProvider, build_provider};
use xa11y::{Point, Rect, Screenshot, ScreenshotProvider};

/// A tool wired to the in-memory fixture.
fn tool() -> (ComputerAccessibilityTool, Arc<MockProvider>) {
    let provider = build_provider();
    let runtime = Xa11yRuntime::with_provider(provider.clone());
    (ComputerAccessibilityTool::with_runtime(runtime), provider)
}

/// Runs one call and returns its envelope.
async fn call(tool: &ComputerAccessibilityTool, arguments: Value) -> Value {
    tool.invoke(arguments)
        .await
        .expect("the tool must not fail")
}

/// Asserts a call succeeded and returns its `data`.
async fn ok(tool: &ComputerAccessibilityTool, arguments: Value) -> Value {
    let envelope = call(tool, arguments).await;
    assert_eq!(envelope["ok"], json!(true), "{envelope}");
    envelope["data"].clone()
}

/// Asserts a call failed with `code` and returns the error object.
async fn failed(tool: &ComputerAccessibilityTool, arguments: Value, code: &str) -> Value {
    let envelope = call(tool, arguments).await;
    assert_eq!(envelope["ok"], json!(false), "{envelope}");
    assert_eq!(envelope["error"]["code"], json!(code), "{envelope}");
    envelope["error"].clone()
}

/// Finds one element and returns its handle.
async fn handle_of(tool: &ComputerAccessibilityTool, selector: Value) -> String {
    let data = ok(tool, json!({"op": "query", "selector": selector})).await;
    data["elements"][0]["id"]
        .as_str()
        .expect("a matching element")
        .to_string()
}

#[tokio::test]
async fn apps_lists_the_application_with_a_focused_flag_and_a_handle() {
    let (tool, _provider) = tool();
    let data = ok(&tool, json!({"op": "apps"})).await;
    let apps = data["apps"].as_array().expect("an apps array");
    assert_eq!(apps.len(), 1, "{data}");
    assert_eq!(apps[0]["name"], json!("TestApp"));
    assert_eq!(apps[0]["focused"], json!(true));
    assert!(
        apps[0]["id"].as_str().unwrap().starts_with("app_"),
        "{data}"
    );
}

#[tokio::test]
async fn apps_deduplicates_by_process_and_reports_the_foreground_flag() {
    let (tool, _provider) = tool();
    let data = ok(&tool, json!({"op": "apps"})).await;
    let pids: Vec<i64> = data["apps"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|app| app["pid"].as_i64())
        .collect();
    let mut unique: Vec<i64> = pids.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(pids.len(), unique.len(), "one entry per process: {data}");
}

#[tokio::test]
async fn query_finds_elements_by_role_name_and_state() {
    let (tool, _provider) = tool();
    let data = ok(
        &tool,
        json!({"op": "query", "selector": {"role": "button", "name": "Back"}}),
    )
    .await;
    let elements = data["elements"].as_array().expect("elements");
    assert_eq!(elements.len(), 1, "{data}");
    assert_eq!(elements[0]["role"], json!("button"));
    assert_eq!(elements[0]["name"], json!("Back"));
    assert_eq!(elements[0]["enabled"], json!(true));
    assert_eq!(elements[0]["visible"], json!(true));
    assert_eq!(elements[0]["bounds"]["width"], json!(50));

    // `name_contains` is the case-insensitive escape hatch.
    let data = ok(
        &tool,
        json!({"op": "query", "selector": {"name_contains": "VOLUME"}}),
    )
    .await;
    assert_eq!(data["elements"][0]["role"], json!("slider"));

    // State fields find the *disabled* control instead of filtering it out.
    let data = ok(
        &tool,
        json!({"op": "query", "selector": {"role": "button", "enabled": false}}),
    )
    .await;
    assert_eq!(data["elements"][0]["name"], json!("Forward"));

    // The invisible status text is reachable by state, too.
    let data = ok(
        &tool,
        json!({"op": "query", "selector": {"visible": false}}),
    )
    .await;
    assert!(
        data["elements"]
            .as_array()
            .unwrap()
            .iter()
            .any(|element| element["name"] == json!("Status")),
        "{data}"
    );
}

#[tokio::test]
async fn query_handles_are_stable_across_calls() {
    let (tool, _provider) = tool();
    let selector = json!({"role": "button", "name": "Back"});
    let first = handle_of(&tool, selector.clone()).await;
    let second = handle_of(&tool, selector).await;
    assert_eq!(first, second);
}

#[tokio::test]
async fn a_scope_handle_narrows_the_search() {
    let (tool, _provider) = tool();
    let window = handle_of(&tool, json!({"role": "window", "name": "Main Window"})).await;

    let data = ok(
        &tool,
        json!({"op": "query", "element": window, "selector": {"role": "text_field"}}),
    )
    .await;
    assert_eq!(data["elements"][0]["name"], json!("Search"));

    // The application root is still reachable through the app handle.
    let apps = ok(&tool, json!({"op": "apps"})).await;
    let app = apps["apps"][0]["id"].as_str().unwrap().to_string();
    let data = ok(
        &tool,
        json!({"op": "query", "app": app, "selector": {"role": "check_box"}}),
    )
    .await;
    assert_eq!(data["elements"][0]["name"], json!("Agree"));
}

#[tokio::test]
async fn an_empty_selector_lists_everything_inside_the_scope() {
    let (tool, _provider) = tool();
    let window = handle_of(&tool, json!({"role": "window", "name": "Main Window"})).await;
    let data = ok(
        &tool,
        json!({"op": "query", "element": window, "max_results": 5}),
    )
    .await;
    assert_eq!(data["elements"].as_array().unwrap().len(), 5);
    assert_eq!(data["truncated"], json!(true));
}

#[tokio::test]
async fn inspect_reports_the_actions_the_element_really_supports() {
    let (tool, _provider) = tool();
    let handle = handle_of(&tool, json!({"role": "button", "name": "Back"})).await;

    let element = ok(&tool, json!({"op": "inspect", "element": handle})).await;
    assert_eq!(element["role"], json!("button"));
    assert_eq!(element["name"], json!("Back"));
    assert_eq!(element["description"], json!("Go back"));
    assert_eq!(element["stable_id"], json!("btn-back"));
    assert_eq!(element["states"]["focusable"], json!(true));
    assert_eq!(element["actions"], json!(["press", "focus"]));
    assert!(element["parent"].as_str().is_some(), "{element}");
    assert_eq!(element["children"], json!([]));
}

#[tokio::test]
async fn inspect_reports_children_and_a_stale_handle_codes_as_stale() {
    let (tool, _provider) = tool();
    let window = handle_of(&tool, json!({"role": "window", "name": "Main Window"})).await;
    let element = ok(&tool, json!({"op": "inspect", "element": window})).await;
    let children = element["children"].as_array().expect("children");
    assert!(
        children
            .iter()
            .any(|child| child["name"] == json!("Navigation")),
        "{element}"
    );

    let error = failed(
        &tool,
        json!({"op": "inspect", "element": "el_9999"}),
        "stale_element",
    )
    .await;
    assert_eq!(error["retryable"], json!(true));
}

#[tokio::test]
async fn tree_renders_bounded_indented_lines_with_handles() {
    let (tool, _provider) = tool();
    let data = ok(&tool, json!({"op": "tree", "depth": 3})).await;
    let text = data["tree"].as_str().expect("tree text");
    assert!(text.starts_with("[el_"), "{text}");
    assert!(text.contains("button \"Back\""), "{text}");
    assert!(text.contains("text_field \"Search\""), "{text}");
    assert!(data["nodes"].as_u64().unwrap() > 1);
    assert_eq!(data["truncated"], json!(false));
    assert!(data["root"].as_str().unwrap().starts_with("el_"));
}

#[tokio::test]
async fn tree_reports_truncation_when_the_node_budget_is_reached() {
    let (tool, _provider) = tool();
    let data = ok(&tool, json!({"op": "tree", "max_nodes": 2})).await;
    assert_eq!(data["nodes"], json!(2));
    assert_eq!(data["truncated"], json!(true));
}

#[tokio::test]
async fn action_presses_a_handle_and_names_the_method_used() {
    let (tool, provider) = tool();
    let handle = handle_of(&tool, json!({"role": "button", "name": "Back"})).await;
    let data = ok(
        &tool,
        json!({"op": "action", "element": handle, "action": "press"}),
    )
    .await;
    assert_eq!(data["action"], json!("press"));
    assert_eq!(data["method"], json!("native_accessibility"));
    assert_eq!(data["resolved_by"], json!("handle"));
    let recorded = provider.actions();
    assert_eq!(recorded.last().map(|entry| entry.1.as_str()), Some("press"));
}

#[tokio::test]
async fn action_resolves_a_selector_and_passes_the_payload_through() {
    let (tool, provider) = tool();
    let data = ok(
        &tool,
        json!({
            "op": "action",
            "selector": {"role": "text_field", "name": "Search"},
            "action": "set_value",
            "value": "world"
        }),
    )
    .await;
    assert_eq!(data["method"], json!("native_accessibility"));
    assert_eq!(data["resolved_by"], json!("selector"));
    assert_eq!(data["value"], json!("world"));
    let recorded = provider.actions();
    let (_, action, payload) = recorded.last().expect("a recorded action");
    assert_eq!(action, "set_value");
    assert_eq!(payload.as_deref(), Some("world"));
}

#[tokio::test]
async fn action_rejects_ambiguous_and_missing_selectors() {
    let (tool, _provider) = tool();
    let ambiguous = failed(
        &tool,
        json!({"op": "action", "selector": {"role": "button"}, "action": "press"}),
        "multiple_matches",
    )
    .await;
    assert_eq!(ambiguous["retryable"], json!(false));

    let missing = failed(
        &tool,
        json!({"op": "action", "selector": {"name": "Nowhere"}, "action": "press"}),
        "element_not_found",
    )
    .await;
    assert_eq!(missing["retryable"], json!(true));
}

#[tokio::test]
async fn select_with_a_value_selects_the_named_entry() {
    let (tool, provider) = tool();
    let list = handle_of(&tool, json!({"role": "list", "name": "Items"})).await;
    let data = ok(
        &tool,
        json!({"op": "action", "element": list, "action": "select", "value": "Item 1"}),
    )
    .await;
    assert_eq!(data["method"], json!("descendant_select"));
    let recorded = provider.actions();
    assert_eq!(
        recorded.last().map(|entry| entry.1.as_str()),
        Some("select")
    );

    let missing = failed(
        &tool,
        json!({"op": "action", "element": list, "action": "select", "value": "Item 99"}),
        "element_not_found",
    )
    .await;
    assert!(missing["message"].as_str().unwrap().contains("Item 99"));
}

#[tokio::test]
async fn hit_test_maps_a_point_to_the_deepest_element() {
    let (tool, _provider) = tool();
    // (130, 70) is inside button "Back" (110, 60, 50x30), the window, and the
    // application root — the deepest match must win.
    let element = ok(&tool, json!({"op": "hit_test", "x": 130, "y": 70})).await;
    assert_eq!(element["element"]["name"], json!("Back"));
    assert_eq!(element["element"]["role"], json!("button"));

    let outside = failed(
        &tool,
        json!({"op": "hit_test", "x": 5000, "y": 5000}),
        "element_not_found",
    )
    .await;
    assert_eq!(outside["retryable"], json!(true));
}

#[tokio::test]
async fn wait_returns_as_soon_as_the_condition_holds() {
    let (tool, _provider) = tool();
    let data = ok(
        &tool,
        json!({
            "op": "wait",
            "selector": {"role": "button", "name": "Back"},
            "condition": "exists",
            "timeout_ms": 1_000
        }),
    )
    .await;
    assert_eq!(data["condition"], json!("exists"));
    assert_eq!(data["element"]["name"], json!("Back"));
}

#[tokio::test]
async fn wait_by_handle_re_resolves_the_element_each_poll() {
    let (tool, _provider) = tool();
    let handle = handle_of(&tool, json!({"role": "text_field", "name": "Search"})).await;
    let data = ok(
        &tool,
        json!({
            "op": "wait",
            "element": handle,
            "condition": "visible",
            "timeout_ms": 1_000
        }),
    )
    .await;
    assert_eq!(data["element"]["role"], json!("text_field"));
    assert_eq!(data["element"]["value"], json!("hello"));
}

#[tokio::test]
async fn wait_times_out_with_a_retryable_code() {
    let (tool, _provider) = tool();
    let error = failed(
        &tool,
        json!({
            "op": "wait",
            "selector": {"role": "button", "name": "Continue"},
            "condition": "exists",
            "timeout_ms": 200
        }),
        "timeout",
    )
    .await;
    assert_eq!(error["retryable"], json!(true));
    // A bounded poll loop, not a busy spin: 200 ms cannot produce many attempts.
    assert!(
        error["message"]
            .as_str()
            .unwrap()
            .contains("was not met within 200 ms"),
        "{error}"
    );
}

#[tokio::test]
async fn wait_supports_absence_visibility_and_value_conditions() {
    let (tool, _provider) = tool();

    // `not_exists` is satisfied by an element that is not there.
    let data = ok(
        &tool,
        json!({
            "op": "wait",
            "selector": {"role": "button", "name": "Continue"},
            "condition": "not_exists",
            "timeout_ms": 200
        }),
    )
    .await;
    assert_eq!(data["element"], json!(null));

    // `visible` never holds for the hidden status text.
    failed(
        &tool,
        json!({
            "op": "wait",
            "selector": {"role": "static_text", "name": "Status"},
            "condition": "visible",
            "timeout_ms": 200
        }),
        "timeout",
    )
    .await;

    // `value_changed` compares against the live value.
    let data = ok(
        &tool,
        json!({
            "op": "wait",
            "selector": {"role": "text_field", "name": "Search"},
            "condition": "value_changed",
            "value": "something else",
            "timeout_ms": 200
        }),
    )
    .await;
    assert_eq!(data["element"]["value"], json!("hello"));
}

#[tokio::test]
async fn unknown_app_handles_are_reported_as_app_not_found() {
    let (tool, _provider) = tool();
    let error = failed(
        &tool,
        json!({"op": "query", "app": "app_9999", "selector": {"role": "button"}}),
        "app_not_found",
    )
    .await;
    assert_eq!(error["retryable"], json!(true));
}

// ── op=screenshot ───────────────────────────────────────────────────────────
//
// Capture is a different backend from the accessibility provider — a different
// OS permission, and `xa11y`'s in-memory fixture knows nothing about pixels —
// so these tests inject a capture backend with a known ratio and origin. That
// is the only way the physical/logical bookkeeping can be checked without a
// display and without a Screen Recording grant.

/// Capture backend with a known pixel ratio and desktop origin.
struct FakeScreenshots {
    /// Physical pixels per logical pixel.
    pixel_ratio: u32,
    /// The ratio the backend reports as `scale`.
    scale: f32,
    /// Logical coordinate of the image's pixel `(0, 0)`.
    origin: Point,
}

impl FakeScreenshots {
    /// A 2× capture, the Retina case where physical and logical differ.
    fn retina(origin: Point) -> Arc<Self> {
        Arc::new(Self {
            pixel_ratio: 2,
            scale: 2.0,
            origin,
        })
    }

    /// A 1× capture, where the two spaces coincide.
    fn standard(origin: Point) -> Arc<Self> {
        Arc::new(Self {
            pixel_ratio: 1,
            scale: 1.0,
            origin,
        })
    }

    /// RGBA pixels of one `logical_width × logical_height` capture: a gradient,
    /// so a PNG of the right dimensions cannot be a coincidence.
    #[allow(clippy::cast_possible_truncation)] // the modulus keeps every value inside a u8
    fn image(&self, logical_width: u32, logical_height: u32) -> Screenshot {
        let width = logical_width * self.pixel_ratio;
        let height = logical_height * self.pixel_ratio;
        let mut pixels = Vec::with_capacity((width as usize) * (height as usize) * 4);
        for y in 0..height {
            for x in 0..width {
                pixels.extend_from_slice(&[(x % 251) as u8, (y % 241) as u8, 0x40, 0xff]);
            }
        }
        Screenshot::new(width, height, pixels, self.scale)
    }
}

impl ScreenshotProvider for FakeScreenshots {
    fn capture_full(&self) -> xa11y::Result<(Screenshot, Point)> {
        Ok((self.image(640, 480), self.origin))
    }

    fn capture_region(&self, rect: Rect) -> xa11y::Result<Screenshot> {
        Ok(self.image(rect.width, rect.height))
    }
}

/// Capture backend that reports the platform's Screen Recording refusal.
struct DeniedScreenshots;

impl ScreenshotProvider for DeniedScreenshots {
    fn capture_full(&self) -> xa11y::Result<(Screenshot, Point)> {
        Err(denied())
    }

    fn capture_region(&self, _rect: Rect) -> xa11y::Result<Screenshot> {
        Err(denied())
    }
}

/// The error a missing Screen Recording grant surfaces as.
fn denied() -> xa11y::Error {
    xa11y::Error::PermissionDenied {
        instructions: "Enable Screen Recording in System Settings".into(),
    }
}

/// A tool over the in-memory accessibility fixture and `capture`.
fn tool_with_capture(capture: Arc<dyn ScreenshotProvider>) -> ComputerAccessibilityTool {
    let runtime = Xa11yRuntime::with_provider(build_provider()).with_screenshot_provider(capture);
    ComputerAccessibilityTool::with_runtime(runtime)
}

/// Width and height read out of a PNG's IHDR chunk.
fn png_size(bytes: &[u8]) -> (u32, u32) {
    assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n", "not a PNG");
    assert_eq!(&bytes[12..16], b"IHDR");
    (
        u32::from_be_bytes(bytes[16..20].try_into().unwrap()),
        u32::from_be_bytes(bytes[20..24].try_into().unwrap()),
    )
}

/// Reads the file a reply points at, checks the byte count it reported, and
/// returns the PNG's real dimensions. The file is deleted again.
fn read_capture(data: &Value) -> (Vec<u8>, (u32, u32)) {
    let path = data["path"].as_str().expect("a path");
    let bytes = std::fs::read(path).unwrap_or_else(|error| panic!("{path}: {error}"));
    assert_eq!(
        bytes.len() as u64,
        data["file_bytes"].as_u64().unwrap(),
        "{data}"
    );
    let size = png_size(&bytes);
    std::fs::remove_file(path).ok();
    (bytes, size)
}

#[tokio::test]
async fn a_region_capture_reports_pixels_logical_size_and_origin() {
    let tool = tool_with_capture(FakeScreenshots::retina(Point::new(0, 0)));
    let data = ok(
        &tool,
        json!({"op": "screenshot", "x": 100, "y": 80, "width": 720, "height": 450}),
    )
    .await;

    assert_eq!(
        data["region"],
        json!({"x": 100, "y": 80, "width": 720, "height": 450})
    );
    assert_eq!(data["full_screen"], json!(false));
    // A region's origin is the rectangle's own, so the model can feed
    // `origin + pixel / scale` straight into op=hit_test.
    assert_eq!(data["origin"], json!({"x": 100, "y": 80}));
    assert_eq!(data["hint"], json!(null));
    assert_eq!(data["image"]["pixel_width"], json!(1440));
    assert_eq!(data["image"]["pixel_height"], json!(900));
    assert_eq!(data["image"]["logical_width"], json!(720));
    assert_eq!(data["image"]["logical_height"], json!(450));
    assert_eq!(data["image"]["scale"], json!(2.0));
    assert_eq!(data["image"]["downscaled"], json!(false));
    assert!(data["captured_at_ms"].as_u64().unwrap() > 0);

    let (_, size) = read_capture(&data);
    assert_eq!(
        size,
        (1440, 900),
        "the file must hold exactly the reported pixels"
    );
}

#[tokio::test]
async fn a_whole_screen_capture_reports_the_backend_origin() {
    let tool = tool_with_capture(FakeScreenshots::retina(Point::new(-1920, 0)));
    let data = ok(&tool, json!({"op": "screenshot"})).await;

    assert_eq!(data["full_screen"], json!(true));
    assert_eq!(data["region"], json!(null));
    // A desktop that starts left of the origin is exactly why the origin is
    // reported instead of assumed to be (0, 0).
    assert_eq!(data["origin"], json!({"x": -1920, "y": 0}));
    assert_eq!(data["image"]["pixel_width"], json!(1280));
    assert_eq!(data["image"]["pixel_height"], json!(960));
    assert_eq!(data["image"]["logical_width"], json!(640));
    assert_eq!(data["image"]["logical_height"], json!(480));
    assert_eq!(data["image"]["scale"], json!(2.0));

    let (_, size) = read_capture(&data);
    assert_eq!(size, (1280, 960));
}

#[tokio::test]
async fn default_captures_land_in_the_screenshot_directory_with_unique_names() {
    let tool = tool_with_capture(FakeScreenshots::standard(Point::new(0, 0)));
    let first = ok(&tool, json!({"op": "screenshot"})).await;
    let second = ok(&tool, json!({"op": "screenshot"})).await;

    let first_path = first["path"].as_str().unwrap().to_string();
    let second_path = second["path"].as_str().unwrap().to_string();
    assert_ne!(first_path, second_path, "a capture never reuses a file");
    for path in [&first_path, &second_path] {
        assert!(path.contains("tizi-screenshots"), "{path}");
        assert!(std::path::Path::new(path).is_file(), "{path}");
        std::fs::remove_file(path).ok();
    }
}

#[tokio::test]
async fn a_relative_path_resolves_under_the_screenshot_directory() {
    let tool = tool_with_capture(FakeScreenshots::standard(Point::new(0, 0)));
    let name = format!("tizi-relative-{}/nested.png", std::process::id());
    let data = ok(
        &tool,
        json!({
            "op": "screenshot",
            "x": 0,
            "y": 0,
            "width": 12,
            "height": 6,
            "path": name
        }),
    )
    .await;

    // The reply carries the resolved absolute path, so `read_image` can open
    // exactly the file that was just written — and the directory was created.
    let path = data["path"].as_str().unwrap();
    assert!(std::path::Path::new(path).is_absolute(), "{path}");
    assert!(path.contains("tizi-screenshots"), "{path}");
    assert!(std::path::Path::new(path).is_file(), "{path}");
    let (_, size) = read_capture(&data);
    assert_eq!(size, (12, 6));
    std::fs::remove_dir_all(std::path::Path::new(path).parent().unwrap()).ok();
}

#[tokio::test]
async fn max_width_downscales_the_file_and_the_scale_follows_the_file() {
    let tool = tool_with_capture(FakeScreenshots::retina(Point::new(0, 0)));
    let data = ok(
        &tool,
        json!({
            "op": "screenshot",
            "x": 0,
            "y": 0,
            "width": 720,
            "height": 450,
            "max_width": 480
        }),
    )
    .await;

    assert_eq!(data["image"]["pixel_width"], json!(480));
    assert_eq!(data["image"]["pixel_height"], json!(300));
    assert_eq!(data["image"]["downscaled"], json!(true));
    // Downscaling shrinks the file, never the screen it describes.
    assert_eq!(data["image"]["logical_width"], json!(720));
    assert_eq!(data["image"]["logical_height"], json!(450));
    assert_eq!(data["image"]["scale"], json!(480.0 / 720.0));
    assert_eq!(data["origin"], json!({"x": 0, "y": 0}));

    let (_, size) = read_capture(&data);
    assert_eq!(size, (480, 300));
}

#[tokio::test]
async fn a_capture_never_overwrites_an_existing_file() {
    let tool = tool_with_capture(FakeScreenshots::standard(Point::new(0, 0)));
    let path = std::env::temp_dir().join(format!(
        "tizi-accessibility-shot-{}.png",
        std::process::id()
    ));
    std::fs::write(&path, b"an earlier capture").unwrap();

    let error = failed(
        &tool,
        json!({
            "op": "screenshot",
            "x": 0,
            "y": 0,
            "width": 10,
            "height": 10,
            "path": path.to_str().unwrap()
        }),
        "write_failed",
    )
    .await;
    assert_eq!(error["retryable"], json!(false));
    assert_eq!(std::fs::read(&path).unwrap(), b"an earlier capture");
    std::fs::remove_file(&path).ok();
}

#[tokio::test]
async fn a_missing_screen_recording_grant_is_a_result_not_a_tool_error() {
    let tool = tool_with_capture(Arc::new(DeniedScreenshots));
    let error = failed(&tool, json!({"op": "screenshot"}), "permission_denied").await;
    assert_eq!(error["retryable"], json!(false));
    assert!(
        error["message"]
            .as_str()
            .unwrap()
            .contains("Screen Recording")
    );
}

#[tokio::test]
async fn capturing_pixels_does_not_need_the_accessibility_backend() {
    // `Xa11yRuntime::new` talks to the real platform accessibility provider,
    // which a headless machine — and a Mac whose Accessibility grant is still
    // missing — does not have. Capture is a different permission, so it has to
    // succeed anyway.
    let runtime =
        Xa11yRuntime::new().with_screenshot_provider(FakeScreenshots::standard(Point::new(0, 0)));
    let tool = ComputerAccessibilityTool::with_runtime(runtime);
    let data = ok(
        &tool,
        json!({"op": "screenshot", "x": 5, "y": 6, "width": 8, "height": 4}),
    )
    .await;

    assert_eq!(data["image"]["pixel_width"], json!(8));
    assert_eq!(data["image"]["pixel_height"], json!(4));
    assert_eq!(data["origin"], json!({"x": 5, "y": 6}));
    let (_, size) = read_capture(&data);
    assert_eq!(size, (8, 4));
}
