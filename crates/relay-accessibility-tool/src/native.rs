//! In-process accessibility backend built on the `xa11y` crate.
//!
//! [`Xa11yRuntime`] implements the seven accessibility operations of
//! `computer_accessibility` directly against the platform accessibility API —
//! macOS `AXUIElement`, Windows UIA, Linux AT-SPI — through `xa11y`, the Rust
//! binding that normalizes them into one element model; `op=screenshot` uses
//! the same crate's capture backend. No helper process, plugin CLI, or
//! shell-out is involved: the server itself is the accessibility client.
//!
//! Three consequences are worth stating up front:
//!
//! * Every `xa11y` call is blocking, so each operation runs on `tokio`'s
//!   blocking pool. The async loop never stalls on a slow accessibility
//!   question, and a stuck platform call cannot wedge the server.
//! * The *server process* needs the OS accessibility permission (macOS:
//!   *Accessibility* in System Settings → Privacy & Security). Without it the
//!   backend surfaces `permission_denied`, which is the model's cue to stop
//!   retrying and tell the user what to grant.
//! * `op=screenshot` needs a *different* grant — Screen Recording on macOS —
//!   and never touches the element tree, so it is dispatched before the
//!   accessibility backend is resolved: a machine that holds only one of the
//!   two permissions still gets the operation that permission allows.
//!
//! `unsafe` is forbidden workspace-wide, which is why the FFI lives in
//! `xa11y` rather than here — this module only ever touches safe APIs.
//!
//! # Shape of an operation
//!
//! ```text
//! request  ──► resolve root (handle / app handle / focused app)
//!          ──► walk the tree under a node budget and a deadline
//!          ──► intern matches as el_71 handles
//!          ──► {"ok": true, "data": …, "meta": {op, elapsed_ms, …}}
//! ```
//!
//! Handles are minted and resolved by [`Registry`]; the walk is bounded by
//! [`Budget`] and [`Deadline`] so a 20 000-node Electron tree can never turn a
//! `query` into an unbounded traversal.

use std::{
    collections::VecDeque,
    io::Write,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, MutexGuard, PoisonError,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use serde_json::{Value, json};
use tracing::{debug, warn};
use xa11y::{
    App, ClickOptions, ClickTarget, Element, ElementData, Error as Xa11yError, Point, Provider,
    Rect, Role, Screenshot, ScreenshotProvider,
};

use crate::{
    AccessibilityRequest,
    envelope::{self, Failure, codes},
    registry::{MAX_TRACKED_APPS, Registry},
    request::{
        ActionRequest, ActionTarget, ElementAction, HitTestRequest, MAX_QUERY_RESULTS,
        MAX_TREE_DEPTH, MAX_TREE_NODES, QueryRequest, ScreenshotRequest, Selector, TreeRequest,
        WaitCondition, WaitRequest,
    },
};

/// Default budget for one call that is not itself a `wait`.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Extra budget granted on top of a `wait` caller timeout.
const GRACE_PERIOD: Duration = Duration::from_secs(5);

/// Wait budget used when `timeout_ms` is omitted.
const DEFAULT_WAIT_TIMEOUT: Duration = Duration::from_secs(30);

/// Interval between two polls of a `wait`.
const WAIT_POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Result cap used by `op=query` when `max_results` is omitted.
const DEFAULT_QUERY_RESULTS: u32 = 50;

/// Deepest level `op=query`, `op=wait`, and `op=hit_test` descend to.
const QUERY_MAX_DEPTH: u32 = 32;

/// Nodes one `op=query` / `op=hit_test` walk may materialize.
///
/// Measured on macOS: an AX traversal materializes roughly 1 700-3 000 nodes
/// per second, so this budget bounds a fruitless search at a few seconds. A
/// search that reaches it answers `truncated: true` — narrow the scope (a
/// window handle from `op=tree`) instead of scanning a whole Electron app from
/// the top, where the interesting elements are hundreds of nodes deep anyway.
const TRAVERSAL_NODE_BUDGET: usize = 6_000;

/// Nodes one `op=wait` re-resolution may materialize.
///
/// Deliberately smaller than a query's: a wait polls, so the per-poll cost is
/// what keeps a 30-second wait from consuming the whole budget. The watched
/// element is normally near a window root, which is where breadth-first
/// traversal looks first.
const REFRESH_NODE_BUDGET: usize = 2_000;

/// Depth used by `op=tree` when `depth` is omitted.
const DEFAULT_TREE_DEPTH: u32 = 4;

/// Node budget used by `op=tree` when `max_nodes` is omitted.
const DEFAULT_TREE_NODES: u32 = 200;

/// Cap on the rendered `op=tree` text, so a wide tree cannot flood the model.
const TREE_TEXT_CHAR_BUDGET: usize = 24_000;

/// Children listed by `op=inspect`.
const INSPECT_CHILD_LIMIT: usize = 50;

/// Applications `op=hit_test` walks when no foreground application is known.
const HIT_TEST_MAX_APPS: usize = 32;

/// Depth searched by `select` when it resolves an entry by name or value.
const SELECT_DESCENDANT_DEPTH: u32 = 8;

/// Nodes searched by `select` when it resolves an entry by name or value.
const SELECT_DESCENDANT_BUDGET: usize = 2_000;

/// Advice attached to a truncated listing or tree, so the model's next move is
/// "narrow the scope" rather than "retry the same call".
const SCOPE_HINT: &str = "the walk stopped at its node budget, so there may be more elements; \
                          narrow the scope (a window or group handle from op=tree, or an `app` \
                          handle), add selector fields, and query again";

/// `method` reported when the platform's own action ran.
const METHOD_NATIVE: &str = "native_accessibility";

/// `method` reported when `press` fell back to a synthesized mouse click.
const METHOD_MOUSE_FALLBACK: &str = "mouse_fallback";

/// `method` reported when `select` resolved and selected a descendant.
const METHOD_DESCENDANT: &str = "descendant_select";

/// `meta.mime` reported by `op=screenshot`.
const SCREENSHOT_MIME: &str = "image/png";

/// Directory under the system temp directory that receives `op=screenshot`
/// files when the caller does not name a path.
const SCREENSHOT_DIR_NAME: &str = "tizi-screenshots";

/// Counter that keeps two captures in the same millisecond from colliding.
static SCREENSHOT_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Attached to a whole-screen capture, whose image origin the umbrella API
/// cannot report: `xa11y::screenshot()` drops the backend's origin, and
/// assuming `(0, 0)` would misplace every pixel on a multi-display desktop.
const FULL_SCREEN_ORIGIN_HINT: &str = "origin is unknown for a whole-screen capture, because a \
     desktop need not start at (0, 0); capture a region (x, y, width, height) when an image pixel \
     has to be turned back into a screen coordinate";

/// Where the backend comes from.
#[derive(Clone)]
enum ProviderSource {
    /// The process-wide `xa11y` singleton, created on first use.
    Global,
    /// An injected provider (tests, or a host that already owns one).
    Fixed(Arc<dyn Provider>),
}

/// Where captured pixels come from.
///
/// Symmetrical to [`ProviderSource`], and for the same reason: capture is a
/// different OS permission from accessibility, and tests need a deterministic
/// desktop without one.
#[derive(Clone)]
enum ScreenshotSource {
    /// The process-wide `xa11y` capture backend, created on first use.
    Global,
    /// An injected backend (tests, or a host that already owns one).
    Fixed(Arc<dyn ScreenshotProvider>),
}

/// The in-process accessibility backend.
///
/// Cloneable and shareable: the handle registry is behind an `Arc<Mutex<_>>`,
/// and every operation serializes only the short critical sections that touch
/// it, never the traversal itself.
#[derive(Clone)]
pub struct Xa11yRuntime {
    provider: ProviderSource,
    screenshot: ScreenshotSource,
    registry: Arc<Mutex<Registry>>,
    timeout: Duration,
}

impl Xa11yRuntime {
    /// Creates a runtime over the process-wide `xa11y` provider singleton.
    #[must_use]
    pub fn new() -> Self {
        Self {
            provider: ProviderSource::Global,
            screenshot: ScreenshotSource::Global,
            registry: Arc::new(Mutex::new(Registry::new())),
            timeout: DEFAULT_TIMEOUT,
        }
    }

    /// Creates a runtime over an injected provider.
    ///
    /// Used by tests (which drive `xa11y`'s in-memory mock provider) and by any
    /// host that already owns a provider it wants the tool to share.
    #[must_use]
    pub fn with_provider(provider: Arc<dyn Provider>) -> Self {
        Self {
            provider: ProviderSource::Fixed(provider),
            screenshot: ScreenshotSource::Global,
            registry: Arc::new(Mutex::new(Registry::new())),
            timeout: DEFAULT_TIMEOUT,
        }
    }

    /// Replaces the screenshot backend, keeping everything else.
    ///
    /// Used by tests: `xa11y`'s in-memory provider knows nothing about
    /// pixels, so `op=screenshot` is exercised against a fake capture backend
    /// that reports a known scale and origin.
    #[must_use]
    pub fn with_screenshot_provider(mut self, provider: Arc<dyn ScreenshotProvider>) -> Self {
        self.screenshot = ScreenshotSource::Fixed(provider);
        self
    }

    /// Overrides the per-call timeout. A zero value is normalized to one
    /// second, so a call always gets at least one real attempt.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout.max(Duration::from_secs(1));
        self
    }

    /// Base per-call timeout, before per-operation adjustments.
    #[must_use]
    pub const fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Budget for one request: a `wait` may ask for more time than the base
    /// timeout, plus a grace period that covers the polling loop's own work.
    fn effective_timeout(&self, request: &AccessibilityRequest) -> Duration {
        match request {
            AccessibilityRequest::Wait(wait) => wait.timeout().map_or(self.timeout, |ms| {
                Duration::from_millis(ms).saturating_add(GRACE_PERIOD)
            }),
            _ => self.timeout,
        }
    }

    /// Resolves the provider this runtime talks to.
    ///
    /// # Errors
    ///
    /// Returns `permission_denied` when the OS withholds accessibility access
    /// and `backend_unavailable` when no accessibility service answers at all
    /// (a Linux session with no AT-SPI bus, for example). `xa11y` caches the
    /// failed construction, so granting macOS permission requires a restart of
    /// the server process.
    fn backend(&self) -> Result<Arc<dyn Provider>, Failure> {
        match &self.provider {
            ProviderSource::Fixed(provider) => Ok(Arc::clone(provider)),
            ProviderSource::Global => {
                xa11y::provider().map_err(|error| Failure::from_xa11y(&error))
            }
        }
    }

    /// Runs one operation to completion, returning its envelope.
    fn run(&self, request: &AccessibilityRequest) -> Value {
        let operation = request.op();
        let started = Instant::now();
        let deadline = Deadline::after(self.effective_timeout(request));
        // Capture and accessibility are separate OS permissions, and a capture
        // never reads the element tree: resolving the accessibility provider
        // first would fail `op=screenshot` on a machine that granted Screen
        // Recording and nothing else.
        let outcome = if let AccessibilityRequest::Screenshot(screenshot) = request {
            debug!(operation, "screenshot request dispatched");
            self.op_screenshot(screenshot)
        } else {
            let provider = match self.backend() {
                Ok(provider) => provider,
                Err(failure) => {
                    warn!(
                        operation,
                        code = failure.code,
                        message = %failure.message,
                        "accessibility backend unavailable"
                    );
                    return failure.to_value();
                }
            };
            debug!(operation, "accessibility request dispatched");
            self.dispatch(&provider, request, &deadline)
        };
        match outcome {
            Ok(outcome) => {
                let mut meta = outcome.meta.as_object().cloned().unwrap_or_default();
                meta.insert("op".to_string(), Value::String(operation.to_string()));
                meta.insert("elapsed_ms".to_string(), json!(elapsed_ms(started)));
                debug!(
                    operation,
                    elapsed_ms = elapsed_ms(started),
                    "accessibility operation finished"
                );
                envelope::success(&outcome.data, &Value::Object(meta))
            }
            Err(failure) => {
                debug!(
                    operation,
                    code = failure.code,
                    message = %failure.message,
                    "accessibility operation failed"
                );
                failure.to_value()
            }
        }
    }

    /// Dispatches one validated request to its operation.
    ///
    /// # Errors
    ///
    /// Propagates the operation's [`Failure`]; semantic failures are results,
    /// so they travel as envelopes rather than tool errors.
    fn dispatch(
        &self,
        provider: &Arc<dyn Provider>,
        request: &AccessibilityRequest,
        deadline: &Deadline,
    ) -> Result<Outcome, Failure> {
        match request {
            AccessibilityRequest::Apps(_) => op_apps(provider, &self.registry),
            AccessibilityRequest::Query(query) => {
                op_query(provider, &self.registry, query, deadline)
            }
            AccessibilityRequest::Inspect(inspect) => {
                op_inspect(provider, &self.registry, &inspect.element)
            }
            AccessibilityRequest::Tree(tree) => op_tree(provider, &self.registry, tree, deadline),
            AccessibilityRequest::Action(action) => {
                op_action(provider, &self.registry, action, deadline)
            }
            AccessibilityRequest::HitTest(hit) => {
                op_hit_test(provider, &self.registry, hit, deadline)
            }
            // [`Xa11yRuntime::run`] answers a screenshot *before* it resolves
            // the accessibility provider, so this arm is reached only when a
            // caller invokes `dispatch` directly; it needs no handle, tree, or
            // deadline.
            AccessibilityRequest::Screenshot(screenshot) => self.op_screenshot(screenshot),
            AccessibilityRequest::Wait(wait) => op_wait(provider, &self.registry, wait),
        }
    }
}

impl Default for Xa11yRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Xa11yRuntime {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Xa11yRuntime")
            .field(
                "provider",
                &match self.provider {
                    ProviderSource::Global => "xa11y singleton",
                    ProviderSource::Fixed(_) => "injected provider",
                },
            )
            .field(
                "screenshot",
                &match self.screenshot {
                    ScreenshotSource::Global => "xa11y capture singleton",
                    ScreenshotSource::Fixed(_) => "injected capture backend",
                },
            )
            .field("timeout", &self.timeout)
            .field("elements", &lock(&self.registry).element_count())
            .finish()
    }
}

/// One structured answer: the operation's `data` plus its own `meta` fields.
struct Outcome {
    data: Value,
    meta: Value,
}

/// Whether a walk should keep going or stop early on a hit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Flow {
    /// Keep traversing.
    Continue,
    /// The caller has everything it needs.
    Stop,
}

/// Result of one bounded walk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WalkStats {
    /// Nodes visited.
    visited: usize,
    /// Whether the node budget cut the walk short.
    truncated: bool,
}

/// Node allowance for one walk.
#[derive(Clone, Copy, Debug)]
struct Budget {
    remaining: usize,
    truncated: bool,
}

impl Budget {
    /// Creates a budget of `nodes` nodes.
    const fn new(nodes: usize) -> Self {
        Self {
            remaining: nodes,
            truncated: false,
        }
    }

    /// Takes one node from the budget; `false` means it is exhausted.
    fn take(&mut self) -> bool {
        if self.remaining == 0 {
            self.truncated = true;
            false
        } else {
            self.remaining -= 1;
            true
        }
    }
}

/// Cooperative deadline for one operation.
///
/// A blocking operation cannot be cancelled from the outside, so the deadline
/// is checked by the traversal and polling loops themselves: a `query` that
/// runs out of time stops with a `timeout` failure instead of holding a
/// blocking-pool thread for an unbounded period.
#[derive(Clone, Copy, Debug)]
struct Deadline {
    at: Instant,
}

impl Deadline {
    /// A deadline `budget` from now.
    fn after(budget: Duration) -> Self {
        Self {
            at: Instant::now() + budget,
        }
    }

    /// Whether the deadline has passed.
    fn expired(&self) -> bool {
        Instant::now() >= self.at
    }

    /// Time left before the deadline.
    fn remaining(&self) -> Duration {
        self.at.saturating_duration_since(Instant::now())
    }

    /// Fails with `timeout` once the deadline has passed.
    ///
    /// # Errors
    ///
    /// Returns `timeout`, which is retryable: the UI may answer faster next
    /// time, or the caller may narrow the request.
    fn check(&self, operation: &str) -> Result<(), Failure> {
        if self.expired() {
            Err(Failure::new(
                codes::TIMEOUT,
                format!(
                    "op={operation} exceeded its time budget; narrow the request (a selector, a \
                     smaller depth/max_nodes, a scope element handle) or retry"
                ),
            ))
        } else {
            Ok(())
        }
    }
}

/// Locks a registry mutex, recovering from poisoning.
///
/// A panic inside one operation must not make every later call fail: the
/// registry holds only snapshots and handles, so the value stays usable.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Milliseconds since `started`, saturating instead of wrapping.
fn elapsed_ms(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

// ── traversal ───────────────────────────────────────────────────────────────

/// Walks the accessibility subtree under `root` breadth-first.
///
/// `visit` sees every node, including the root, with its depth. A `Stop` from
/// `visit` ends the walk early; the node budget and the deadline end it with
/// `truncated`/`timeout` instead. A failure to enumerate the *root's* children
/// is propagated (it means the scope itself is unreadable); a failure deeper
/// down only skips that subtree, because one busy application must not fail a
/// query that other branches can answer.
fn visit<F>(
    provider: &Arc<dyn Provider>,
    root: &ElementData,
    max_depth: u32,
    budget: &mut Budget,
    deadline: &Deadline,
    operation: &str,
    mut visit: F,
) -> Result<WalkStats, Failure>
where
    F: FnMut(&ElementData, u32) -> Result<Flow, Failure>,
{
    let mut queue: VecDeque<(ElementData, u32)> = VecDeque::new();
    queue.push_back((root.clone(), 0));
    let mut visited = 0_usize;
    while let Some((data, depth)) = queue.pop_front() {
        if !budget.take() {
            break;
        }
        deadline.check(operation)?;
        visited += 1;
        if visit(&data, depth)? == Flow::Stop {
            break;
        }
        if depth >= max_depth {
            continue;
        }
        match provider.get_children(Some(&data)) {
            Ok(children) => queue.extend(children.into_iter().map(|child| (child, depth + 1))),
            Err(error) if visited == 1 => return Err(Failure::from_xa11y(&error)),
            Err(error) => debug!(
                operation,
                error = %error,
                "skipping a subtree whose children could not be enumerated"
            ),
        }
    }
    Ok(WalkStats {
        visited,
        truncated: budget.truncated,
    })
}

/// Resolves the root an operation starts from.
///
/// Precedence is explicit scope handle, then application handle, then the
/// focused application: the model chooses how narrow the search is, and gets a
/// clear `app_not_found` when nothing holds the foreground.
///
/// # Errors
///
/// Returns `stale_element` for an unknown element handle, `app_not_found` for
/// an unknown application handle or no foreground application, and whatever the
/// provider reports when the focused application cannot be read.
fn resolve_root(
    provider: &Arc<dyn Provider>,
    registry: &Mutex<Registry>,
    app: Option<&str>,
    element: Option<&str>,
) -> Result<ElementData, Failure> {
    if let Some(handle) = element {
        return lock(registry).element(handle);
    }
    if let Some(handle) = app {
        return lock(registry).app(handle);
    }
    provider.focused_app().map_err(|error| match error {
        Xa11yError::SelectorNotMatched { .. } => Failure::new(
            codes::APP_NOT_FOUND,
            "no application currently holds the foreground; pass `app` from op=apps, or focus a \
             window first",
        ),
        other => Failure::from_xa11y(&other),
    })
}

/// Interns one element snapshot, returning its handle.
fn intern_element(registry: &Mutex<Registry>, data: &ElementData) -> String {
    lock(registry).intern_element(data)
}

// ── op=apps ─────────────────────────────────────────────────────────────────

/// Lists the applications the backend can reach.
///
/// # Errors
///
/// Propagates enumeration failures (`permission_denied`, `backend_unavailable`)
/// and reports `app_not_found` when the backend knows of no application at all.
fn op_apps(provider: &Arc<dyn Provider>, registry: &Mutex<Registry>) -> Result<Outcome, Failure> {
    // `App::list_with` collapses platform registrations by pid and tags the
    // foreground application in one focus query, which is the library's own
    // notion of "the same application" and "the focused one". Rebuilding that
    // here would silently disagree with `xa11y` on macOS, where cheap
    // presentation data (title, bounds) cannot identify an application and the
    // per-snapshot handle is not a cross-enumeration key.
    let apps = App::list_with(Arc::clone(provider)).map_err(|error| Failure::from_xa11y(&error))?;
    let focused_known = apps.iter().any(App::is_foreground);
    let mut listed = Vec::new();
    for app in apps {
        let handle = lock(registry).intern_app(&app.data);
        listed.push(json!({
            "id": handle,
            "name": app.data.name.clone().unwrap_or_default(),
            "pid": app.data.pid,
            "focused": app.is_foreground(),
        }));
        if listed.len() >= MAX_TRACKED_APPS {
            break;
        }
    }
    if listed.is_empty() {
        return Err(Failure::new(
            codes::APP_NOT_FOUND,
            "the accessibility backend reports no running application",
        ));
    }
    Ok(Outcome {
        data: json!({ "apps": listed }),
        meta: json!({
            "count": listed.len(),
            // The foreground query answers "no application" on a desktop where
            // nothing is frontmost (and can refuse outright); saying so beats
            // letting the model read 30 unfocused apps as a real snapshot.
            "focused_known": focused_known,
        }),
    })
}

// ── op=query ────────────────────────────────────────────────────────────────

/// Finds elements by semantic selector.
///
/// # Errors
///
/// Propagates scope-resolution failures and `timeout`; an empty result is a
/// success with an empty `elements` array.
fn op_query(
    provider: &Arc<dyn Provider>,
    registry: &Mutex<Registry>,
    request: &QueryRequest,
    deadline: &Deadline,
) -> Result<Outcome, Failure> {
    let root = resolve_root(
        provider,
        registry,
        request.app.as_deref(),
        request.element.as_deref(),
    )?;
    let limit = request
        .max_results
        .unwrap_or(DEFAULT_QUERY_RESULTS)
        .clamp(1, MAX_QUERY_RESULTS) as usize;
    let (matches, stats) = collect_matches(
        provider,
        &root,
        request.selector.as_ref(),
        limit,
        deadline,
        "query",
    )?;
    let elements: Vec<Value> = matches
        .iter()
        .map(|data| element_summary(&intern_element(registry, data), data))
        .collect();
    let truncated = stats.truncated || matches.len() >= limit;
    let mut data = json!({
        "elements": elements,
        // The result cap and the node budget are both "there may be more".
        "truncated": truncated,
    });
    if let Some(object) = data.as_object_mut()
        && truncated
    {
        object.insert("hint".to_string(), json!(SCOPE_HINT));
    }
    Ok(Outcome {
        data,
        meta: json!({ "count": matches.len(), "visited": stats.visited }),
    })
}

/// Collects up to `limit` elements matching `selector` under `root`.
///
/// An absent selector matches every element, which is what makes
/// `query` usable as "show me what is inside this scope" — bounded by `limit`
/// and by the traversal budget.
///
/// # Errors
///
/// Propagates an unreadable scope or an expired deadline.
fn collect_matches(
    provider: &Arc<dyn Provider>,
    root: &ElementData,
    selector: Option<&Selector>,
    limit: usize,
    deadline: &Deadline,
    operation: &str,
) -> Result<(Vec<ElementData>, WalkStats), Failure> {
    let mut budget = Budget::new(TRAVERSAL_NODE_BUDGET);
    let mut matches: Vec<ElementData> = Vec::new();
    let stats = visit(
        provider,
        root,
        QUERY_MAX_DEPTH,
        &mut budget,
        deadline,
        operation,
        |data, _depth| {
            if selector.is_none_or(|selector| selector_matches(selector, data)) {
                matches.push(data.clone());
                if matches.len() >= limit {
                    return Ok(Flow::Stop);
                }
            }
            Ok(Flow::Continue)
        },
    )?;
    Ok((matches, stats))
}

/// Whether one element satisfies every field of `selector`.
///
/// Name matching is exact and case-sensitive; `name_contains` is the
/// case-insensitive substring escape hatch. The three state fields compare the
/// element's own state flags, so `{"enabled": false}` finds a disabled control
/// rather than silently filtering it out.
fn selector_matches(selector: &Selector, data: &ElementData) -> bool {
    selector
        .role
        .as_ref()
        .is_none_or(|role| role_matches(role, data))
        && selector
            .name
            .as_ref()
            .is_none_or(|name| data.name.as_deref() == Some(name.as_str()))
        && selector.name_contains.as_ref().is_none_or(|needle| {
            data.name
                .as_deref()
                .is_some_and(|name| contains_ignore_case(name, needle))
        })
        && selector
            .value
            .as_ref()
            .is_none_or(|value| data.value.as_deref() == Some(value.as_str()))
        && selector
            .description
            .as_ref()
            .is_none_or(|description| data.description.as_deref() == Some(description.as_str()))
        && selector
            .focused
            .is_none_or(|focused| data.states.focused == focused)
        && selector
            .enabled
            .is_none_or(|enabled| data.states.enabled == enabled)
        && selector
            .visible
            .is_none_or(|visible| data.states.visible == visible)
}

/// Whether `role` names this element's role.
///
/// Well-known roles compare through [`Role`] so `text_field`, `textfield`, and
/// the platform spelling all land on the same variant; anything the enum does
/// not know compares as the `snake_case` string the backend reported.
fn role_matches(role: &str, data: &ElementData) -> bool {
    let expected = role.trim().to_ascii_lowercase();
    Role::from_snake_case(&expected).map_or_else(
        || data.role.to_snake_case() == expected,
        |parsed| data.role == parsed,
    )
}

/// Case-insensitive substring test.
fn contains_ignore_case(haystack: &str, needle: &str) -> bool {
    haystack.to_lowercase().contains(&needle.to_lowercase())
}

// ── op=inspect ──────────────────────────────────────────────────────────────

/// Reads every property of one element, plus parent and child handles.
///
/// # Errors
///
/// Returns `stale_element` for a handle this runtime did not issue or has
/// already dropped.
fn op_inspect(
    provider: &Arc<dyn Provider>,
    registry: &Mutex<Registry>,
    handle: &str,
) -> Result<Outcome, Failure> {
    let data = lock(registry).element(handle)?;
    let mut detail = element_detail(handle, &data);
    let parent = provider
        .get_parent(&data)
        .ok()
        .flatten()
        .map(|parent| intern_element(registry, &parent));
    let (children, children_truncated) = match provider.get_children(Some(&data)) {
        Ok(children) => {
            let total = children.len();
            let listed = children
                .iter()
                .take(INSPECT_CHILD_LIMIT)
                .map(|child| {
                    json!({
                        "id": intern_element(registry, child),
                        "role": role_name(child),
                        "name": child.name.clone(),
                    })
                })
                .collect::<Vec<Value>>();
            (listed, total > INSPECT_CHILD_LIMIT)
        }
        Err(error) => {
            debug!(error = %error, "inspect could not enumerate children");
            (Vec::new(), false)
        }
    };
    if let Some(object) = detail.as_object_mut() {
        object.insert("parent".to_string(), json!(parent));
        object.insert("children".to_string(), json!(children));
        object.insert("children_truncated".to_string(), json!(children_truncated));
    }
    Ok(Outcome {
        data: detail,
        meta: json!(null),
    })
}

// ── op=tree ─────────────────────────────────────────────────────────────────

/// Renders a bounded, indented subtree with a handle on every line.
///
/// # Errors
///
/// Propagates scope-resolution failures and `timeout`.
fn op_tree(
    provider: &Arc<dyn Provider>,
    registry: &Mutex<Registry>,
    request: &TreeRequest,
    deadline: &Deadline,
) -> Result<Outcome, Failure> {
    let root = resolve_root(
        provider,
        registry,
        request.app.as_deref(),
        request.root.as_deref(),
    )?;
    let max_depth = request
        .depth
        .unwrap_or(DEFAULT_TREE_DEPTH)
        .min(MAX_TREE_DEPTH);
    let max_nodes = request
        .max_nodes
        .unwrap_or(DEFAULT_TREE_NODES)
        .min(MAX_TREE_NODES) as usize;
    let root_handle = intern_element(registry, &root);
    let mut writer = TreeWriter {
        provider,
        registry,
        deadline,
        max_depth,
        budget: Budget::new(max_nodes),
        chars: 0,
        nodes: 0,
        text: String::new(),
    };
    writer.visit(&root, 0)?;
    let truncated = writer.budget.truncated || writer.chars >= TREE_TEXT_CHAR_BUDGET;
    let mut data = json!({
        "root": root_handle,
        "tree": writer.text,
        "nodes": writer.nodes,
        "depth": max_depth,
        "truncated": truncated,
    });
    if let Some(object) = data.as_object_mut()
        && truncated
    {
        object.insert("hint".to_string(), json!(SCOPE_HINT));
    }
    Ok(Outcome {
        data,
        meta: json!({ "max_nodes": max_nodes }),
    })
}

/// Depth-first renderer for `op=tree`.
struct TreeWriter<'a> {
    provider: &'a Arc<dyn Provider>,
    registry: &'a Mutex<Registry>,
    deadline: &'a Deadline,
    max_depth: u32,
    budget: Budget,
    chars: usize,
    nodes: usize,
    text: String,
}

impl TreeWriter<'_> {
    /// Appends one element (and its subtree) to the rendering.
    ///
    /// # Errors
    ///
    /// Propagates an unreadable root or an expired deadline.
    fn visit(&mut self, data: &ElementData, depth: u32) -> Result<(), Failure> {
        if !self.budget.take() || self.chars >= TREE_TEXT_CHAR_BUDGET {
            return Ok(());
        }
        self.deadline.check("tree")?;
        self.nodes += 1;
        let line = format!(
            "[{}] {}",
            intern_element(self.registry, data),
            describe(data)
        );
        self.text.push_str(&"  ".repeat(depth as usize));
        self.chars += line.chars().count() + 2 * depth as usize + 1;
        self.text.push_str(&line);
        self.text.push('\n');
        if depth >= self.max_depth {
            return Ok(());
        }
        let children = match self.provider.get_children(Some(data)) {
            Ok(children) => children,
            Err(error) if depth == 0 => return Err(Failure::from_xa11y(&error)),
            Err(error) => {
                debug!(error = %error, "tree could not enumerate a branch");
                return Ok(());
            }
        };
        for child in &children {
            self.visit(child, depth + 1)?;
        }
        Ok(())
    }
}

/// One-line rendering of an element: `button "Submit" = "value"`.
fn describe(data: &ElementData) -> String {
    use std::fmt::Write as _;

    let mut text = role_name(data).to_string();
    if let Some(name) = data.name.as_deref().filter(|name| !name.is_empty()) {
        let _ = write!(text, " {name:?}");
    }
    if let Some(value) = data.value.as_deref().filter(|value| !value.is_empty()) {
        let _ = write!(text, " = {value:?}");
    }
    text
}

// ── op=action ───────────────────────────────────────────────────────────────

/// Performs one semantic action on an element handle or an inline selector.
///
/// The runtime owns the fallbacks: a `press` that the platform refuses falls
/// back to a synthesized click on the element's own bounds, and a `select` with
/// a value resolves the matching descendant. The reply names the path that was
/// actually used in `method`, so a surprising result can be debugged.
///
/// # Errors
///
/// `element_not_found` / `multiple_matches` when an inline selector does not
/// resolve uniquely, `stale_element` for an expired handle, and
/// `unsupported_action` when neither the native action nor its fallback works.
fn op_action(
    provider: &Arc<dyn Provider>,
    registry: &Mutex<Registry>,
    request: &ActionRequest,
    deadline: &Deadline,
) -> Result<Outcome, Failure> {
    let (data, resolved_by) = match &request.target {
        ActionTarget::Element(target) => (lock(registry).element(&target.element)?, "handle"),
        ActionTarget::Selector(target) => {
            let root = resolve_root(provider, registry, target.app.as_deref(), None)?;
            let (mut matches, _stats) = collect_matches(
                provider,
                &root,
                Some(&target.selector),
                2,
                deadline,
                "action",
            )?;
            match matches.len() {
                0 => {
                    return Err(Failure::new(
                        codes::ELEMENT_NOT_FOUND,
                        format!(
                            "no element matched the selector for action {:?}",
                            request.action.as_str()
                        ),
                    ));
                }
                1 => (matches.remove(0), "selector"),
                _ => {
                    return Err(Failure::new(
                        codes::MULTIPLE_MATCHES,
                        format!(
                            "the selector for action {:?} matched more than one element; add \
                             fields until it is unique, or query first and act on a handle",
                            request.action.as_str()
                        ),
                    ));
                }
            }
        }
    };
    let element = Element::new(data.clone(), Arc::clone(provider));
    let method = perform(provider, &element, request, deadline)
        .map_err(|failure| explain_action_failure(failure, &data))?;
    Ok(Outcome {
        data: json!({
            "action": request.action.as_str(),
            "element": intern_element(registry, &data),
            "method": method,
            "resolved_by": resolved_by,
            "value": request.value.clone(),
        }),
        meta: json!(null),
    })
}

/// Replaces a vague "the platform refused" with the element state that
/// explains it.
///
/// A disabled or hidden control is the most common reason an action comes back
/// unsupported, and both are recoverable — wait for `enabled` / `visible`, or
/// scroll the element into view — so the reply says which one it was instead of
/// leaving the model to guess.
fn explain_action_failure(failure: Failure, data: &ElementData) -> Failure {
    if failure.code != codes::UNSUPPORTED_ACTION {
        return failure;
    }
    if !data.states.enabled {
        return Failure::new(
            codes::NOT_ENABLED,
            format!(
                "{}; the element is disabled — wait for condition `enabled`, or act on a \
                 different element",
                failure.message
            ),
        );
    }
    if !data.states.visible {
        return Failure::new(
            codes::NOT_VISIBLE,
            format!(
                "{}; the element is not visible — scroll it into view or wait for condition \
                 `visible`",
                failure.message
            ),
        );
    }
    failure
}

/// Runs one action, applying the documented fallbacks.
///
/// # Errors
///
/// Returns the translated backend failure when the action — and, for `press`,
/// its coordinate fallback — cannot be performed.
fn perform(
    provider: &Arc<dyn Provider>,
    element: &Element,
    request: &ActionRequest,
    deadline: &Deadline,
) -> Result<&'static str, Failure> {
    let text = request.value.as_ref().map(action_text);
    match request.action {
        ElementAction::Press => press_with_fallback(element),
        ElementAction::Focus => call(element.focus()),
        ElementAction::SetValue => call(element.set_value(text.as_deref().unwrap_or_default())),
        ElementAction::TypeText => call(element.type_text(text.as_deref().unwrap_or_default())),
        ElementAction::Toggle => call(element.toggle()),
        ElementAction::Select => match text.as_deref() {
            Some(value) => select_descendant(provider, element, value, deadline),
            None => call(element.select()),
        },
        ElementAction::Expand => call(element.expand()),
        ElementAction::Collapse => call(element.collapse()),
        ElementAction::Increment => call(element.increment()),
        ElementAction::Decrement => call(element.decrement()),
        ElementAction::ScrollIntoView => call(element.scroll_into_view()),
    }
}

/// Maps a backend result to the method name reported to the model.
///
/// # Errors
///
/// Translates the backend failure into the tool's stable code set.
fn call(result: xa11y::Result<()>) -> Result<&'static str, Failure> {
    result
        .map(|()| METHOD_NATIVE)
        .map_err(|error| Failure::from_xa11y(&error))
}

/// Presses an element, falling back to a synthesized click.
///
/// Only a *missing capability* triggers the fallback: a permission failure or
/// a disappearing element is reported as-is, because clicking would either fail
/// the same way or act on something the user cannot see.
///
/// # Errors
///
/// Returns the original failure joined with the fallback's when both paths
/// fail, and the plain backend failure for anything that is not a missing
/// capability.
fn press_with_fallback(element: &Element) -> Result<&'static str, Failure> {
    match element.press() {
        Ok(()) => Ok(METHOD_NATIVE),
        Err(error) if is_missing_capability(&error) => {
            let original = Failure::from_xa11y(&error);
            match mouse_click(element) {
                Ok(()) => Ok(METHOD_MOUSE_FALLBACK),
                Err(fallback) => Err(Failure::new(
                    original.code,
                    format!(
                        "{}; coordinate fallback failed: {}",
                        original.message, fallback.message
                    ),
                )),
            }
        }
        Err(error) => Err(Failure::from_xa11y(&error)),
    }
}

/// Whether a failure means "this backend cannot do that", rather than "that
/// went wrong".
fn is_missing_capability(error: &Xa11yError) -> bool {
    matches!(
        error,
        Xa11yError::ActionNotSupported { .. }
            | Xa11yError::TextValueNotSupported
            | Xa11yError::Unsupported { .. }
            | Xa11yError::NoElementBounds
    )
}

/// Clicks the element's own anchor point through the input layer.
///
/// # Errors
///
/// Returns `unsupported` when the platform has no input backend (a Wayland
/// session without a portal grant), and the backend's failure otherwise.
fn mouse_click(element: &Element) -> Result<(), Failure> {
    let simulation = xa11y::input_sim().map_err(|error| Failure::from_xa11y(&error))?;
    simulation
        .mouse()
        .click_with(ClickTarget::Element(element), ClickOptions::new())
        .map_err(|error| Failure::from_xa11y(&error))
}

/// Selects the one descendant named or valued `value`.
///
/// # Errors
///
/// `element_not_found` when nothing matches, `multiple_matches` when several do
/// (the model must name the entry uniquely), and the backend failure otherwise.
fn select_descendant(
    provider: &Arc<dyn Provider>,
    element: &Element,
    value: &str,
    deadline: &Deadline,
) -> Result<&'static str, Failure> {
    let root = element.data().clone();
    let mut budget = Budget::new(SELECT_DESCENDANT_BUDGET);
    let mut matches: Vec<ElementData> = Vec::new();
    visit(
        provider,
        &root,
        SELECT_DESCENDANT_DEPTH,
        &mut budget,
        deadline,
        "action",
        |data, _depth| {
            if data.name.as_deref() == Some(value) || data.value.as_deref() == Some(value) {
                matches.push(data.clone());
                if matches.len() > 1 {
                    return Ok(Flow::Stop);
                }
            }
            Ok(Flow::Continue)
        },
    )?;
    match matches.len() {
        0 => Err(Failure::new(
            codes::ELEMENT_NOT_FOUND,
            format!("no entry inside the target element is named or valued {value:?}"),
        )),
        1 => call(Element::new(matches.remove(0), Arc::clone(provider)).select())
            .map(|_| METHOD_DESCENDANT),
        _ => Err(Failure::new(
            codes::MULTIPLE_MATCHES,
            format!("more than one entry matches {value:?}; scope the action to a single element"),
        )),
    }
}

/// Renders an action payload as the string the platform layer takes.
fn action_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

// ── op=hit_test ─────────────────────────────────────────────────────────────

/// Application roots to search, foreground application first.
///
/// The foreground application is listed first because a screenshot coordinate
/// almost always belongs to it; the rest of the desktop follows as a fallback,
/// capped by [`HIT_TEST_MAX_APPS`]. Application entries are deduplicated by
/// pid, the process-level identity `xa11y` itself uses.
///
/// # Errors
///
/// Returns the enumeration failure when no application could be listed at all.
/// A failing *foreground* query is tolerated: it costs the ordering hint, not
/// the search.
fn application_roots(provider: &Arc<dyn Provider>) -> Result<Vec<ElementData>, Failure> {
    let mut roots: Vec<ElementData> = Vec::new();
    if let Ok(focused) = provider.focused_app() {
        roots.push(focused);
    }
    match provider.list_apps() {
        Ok(apps) => {
            for app in apps {
                if app
                    .pid
                    .is_some_and(|pid| roots.iter().any(|root| root.pid == Some(pid)))
                {
                    continue;
                }
                roots.push(app);
                if roots.len() >= HIT_TEST_MAX_APPS {
                    break;
                }
            }
        }
        Err(error) if roots.is_empty() => return Err(Failure::from_xa11y(&error)),
        Err(error) => debug!(error = %error, "hit_test could not enumerate applications"),
    }
    Ok(roots)
}

/// Resolves screen coordinates to the deepest element containing the point.
///
/// # Errors
///
/// Returns `element_not_found` when no visible element covers the point, and
/// propagates backend failures.
fn op_hit_test(
    provider: &Arc<dyn Provider>,
    registry: &Mutex<Registry>,
    request: &HitTestRequest,
    deadline: &Deadline,
) -> Result<Outcome, Failure> {
    let (x, y) = (request.x, request.y);
    let roots = application_roots(provider)?;
    let mut budget = Budget::new(TRAVERSAL_NODE_BUDGET);
    let mut best: Option<Candidate> = None;
    let mut visited = 0_usize;
    for root in &roots {
        if budget.remaining == 0 {
            break;
        }
        let stats = visit(
            provider,
            root,
            QUERY_MAX_DEPTH,
            &mut budget,
            deadline,
            "hit_test",
            |data, depth| {
                consider(data, depth, x, y, &mut best);
                Ok(Flow::Continue)
            },
        )?;
        visited += stats.visited;
    }
    let Some(candidate) = best else {
        return Err(Failure::new(
            codes::ELEMENT_NOT_FOUND,
            format!("no visible element contains the screen point ({x}, {y})"),
        ));
    };
    Ok(Outcome {
        data: json!({
            "element": element_summary(&intern_element(registry, &candidate.data), &candidate.data),
        }),
        meta: json!({
            "roots": roots.len(),
            "visited": visited,
            "truncated": budget.truncated,
        }),
    })
}

/// Best hit so far: the deepest element containing the point, then the
/// smallest one — a button beats the window that contains it.
struct Candidate {
    depth: u32,
    area: u64,
    data: ElementData,
}

/// Offers one element as a hit-test candidate.
fn consider(data: &ElementData, depth: u32, x: f64, y: f64, best: &mut Option<Candidate>) {
    if !data.states.visible {
        return;
    }
    let Some(bounds) = data.bounds else {
        return;
    };
    if !rect_contains(&bounds, x, y) {
        return;
    }
    let area = u64::from(bounds.width) * u64::from(bounds.height);
    let better = best.as_ref().is_none_or(|current| {
        depth > current.depth || (depth == current.depth && area < current.area)
    });
    if better {
        *best = Some(Candidate {
            depth,
            area,
            data: data.clone(),
        });
    }
}

/// Whether a logical-coordinate point falls inside `rect`.
fn rect_contains(rect: &Rect, x: f64, y: f64) -> bool {
    let left = f64::from(rect.x);
    let top = f64::from(rect.y);
    x >= left && y >= top && x < left + f64::from(rect.width) && y < top + f64::from(rect.height)
}

// ── op=wait ─────────────────────────────────────────────────────────────────

/// Polls until an element reaches the requested state.
///
/// The wait budget is the caller's `timeout_ms` (or [`DEFAULT_WAIT_TIMEOUT`]).
/// Each poll re-resolves the target against the live tree — a handle is a hint,
/// not an identity — so `stale_element` cannot make a wait spin on a snapshot
/// that the application has already replaced.
///
/// # Errors
///
/// Returns `timeout` when the budget elapses, and propagates backend failures
/// that are not themselves the awaited condition.
fn op_wait(
    provider: &Arc<dyn Provider>,
    registry: &Mutex<Registry>,
    request: &WaitRequest,
) -> Result<Outcome, Failure> {
    let budget = request
        .timeout()
        .map_or(DEFAULT_WAIT_TIMEOUT, Duration::from_millis);
    let deadline = Deadline::after(budget);
    let started = Instant::now();
    let expected = request.value.as_ref().map(action_text);
    let mut attempts = 0_u64;
    loop {
        attempts += 1;
        let target = match resolve_wait_target(provider, registry, request, &deadline) {
            Ok(target) => target,
            Err(failure) if awaiting_absence(request.condition, &failure) => None,
            // A resolution that ran out of budget is the wait running out of
            // budget: report it as one timeout, not as a generic traversal
            // failure the model cannot place.
            Err(failure) if failure.code == codes::TIMEOUT => {
                return Err(wait_timeout(request.condition, budget, attempts));
            }
            Err(failure) => return Err(failure),
        };
        if wait_condition_met(request.condition, target.as_ref(), expected.as_deref()) {
            let element = target
                .as_ref()
                .map(|data| element_summary(&intern_element(registry, data), data));
            return Ok(Outcome {
                data: json!({
                    "condition": condition_name(request.condition),
                    "elapsed_ms": elapsed_ms(started),
                    "element": element,
                }),
                meta: json!({
                    "attempts": attempts,
                    "timeout_ms": u64::try_from(budget.as_millis()).unwrap_or(u64::MAX),
                }),
            });
        }
        if deadline.expired() {
            return Err(wait_timeout(request.condition, budget, attempts));
        }
        std::thread::sleep(WAIT_POLL_INTERVAL.min(deadline.remaining()));
    }
}

/// The single timeout failure a `wait` reports, whatever path noticed it.
fn wait_timeout(condition: WaitCondition, budget: Duration, attempts: u64) -> Failure {
    Failure::new(
        codes::TIMEOUT,
        format!(
            "condition {:?} was not met within {} ms ({attempts} poll(s)); re-observe with \
             op=query or op=tree, or raise timeout_ms",
            condition_name(condition),
            budget.as_millis()
        ),
    )
}

/// Whether a failed resolution *is* the awaited condition.
///
/// `not_exists` is satisfied by an application or element that has gone away,
/// which is exactly the failure a resolution reports.
fn awaiting_absence(condition: WaitCondition, failure: &Failure) -> bool {
    condition == WaitCondition::NotExists
        && matches!(
            failure.code,
            codes::APP_NOT_FOUND | codes::ELEMENT_NOT_FOUND | codes::STALE_ELEMENT
        )
}

/// Resolves what the wait is observing: a fresh snapshot of the watched
/// element, or `None` when it no longer exists.
///
/// # Errors
///
/// Propagates scope resolution and traversal failures.
fn resolve_wait_target(
    provider: &Arc<dyn Provider>,
    registry: &Mutex<Registry>,
    request: &WaitRequest,
    deadline: &Deadline,
) -> Result<Option<ElementData>, Failure> {
    if let Some(handle) = request.element.as_deref() {
        let snapshot = lock(registry).element(handle)?;
        return refresh_element(provider, &snapshot, deadline);
    }
    let selector = request
        .selector
        .as_ref()
        .expect("validated: wait takes an element or a selector");
    let root = resolve_root(provider, registry, request.app.as_deref(), None)?;
    let (matches, _stats) = collect_matches(provider, &root, Some(selector), 1, deadline, "wait")?;
    Ok(matches.into_iter().next())
}

/// Re-resolves a snapshot against the live tree.
///
/// The application is located by pid first — that keeps the search inside the
/// right process even when focus moved elsewhere — then the app's subtree is
/// searched for the same platform identity. `Ok(None)` means the element is
/// gone, which is a result, not an error.
///
/// # Errors
///
/// Propagates an expired deadline.
fn refresh_element(
    provider: &Arc<dyn Provider>,
    snapshot: &ElementData,
    deadline: &Deadline,
) -> Result<Option<ElementData>, Failure> {
    let wanted = Registry::identity(snapshot);
    let root = snapshot
        .pid
        .and_then(|pid| {
            let apps = provider.list_apps().ok()?;
            apps.into_iter().find(|app| app.pid == Some(pid))
        })
        .or_else(|| provider.focused_app().ok());
    let Some(root) = root else {
        return Ok(None);
    };
    let mut budget = Budget::new(REFRESH_NODE_BUDGET);
    let mut by_identity: Option<ElementData> = None;
    let mut by_presentation: Vec<ElementData> = Vec::new();
    visit(
        provider,
        &root,
        QUERY_MAX_DEPTH,
        &mut budget,
        deadline,
        "wait",
        |data, _depth| {
            if Registry::identity(data) == wanted {
                by_identity = Some(data.clone());
                return Ok(Flow::Stop);
            }
            if same_presentation(snapshot, data) {
                by_presentation.push(data.clone());
            }
            Ok(Flow::Continue)
        },
    )?;
    if let Some(found) = by_identity {
        return Ok(Some(found));
    }
    // No stable id (macOS usually has none) means the platform identity of this
    // snapshot cannot be recomputed; a *unique* presentation match is then the
    // best available answer. Ambiguity is reported as "gone" rather than
    // guessed, which keeps `not_exists` honest.
    if by_presentation.len() == 1 {
        return Ok(by_presentation.pop());
    }
    Ok(None)
}

/// Whether two snapshots describe the same control by presentation.
///
/// Role, name, and bounds are the triple a user would point at twice; `value`
/// is deliberately left out, because a wait for `value_changed` must still
/// recognize the element whose value it is watching.
fn same_presentation(a: &ElementData, b: &ElementData) -> bool {
    a.role == b.role && a.name == b.name && a.bounds == b.bounds
}

/// Whether the observed element now satisfies `condition`.
fn wait_condition_met(
    condition: WaitCondition,
    target: Option<&ElementData>,
    expected: Option<&str>,
) -> bool {
    match condition {
        WaitCondition::Exists => target.is_some(),
        WaitCondition::NotExists => target.is_none(),
        WaitCondition::Visible => target.is_some_and(|data| data.states.visible),
        WaitCondition::Enabled => target.is_some_and(|data| data.states.enabled),
        WaitCondition::Focused => target.is_some_and(|data| data.states.focused),
        WaitCondition::ValueChanged => target.is_some_and(|data| data.value.as_deref() != expected),
    }
}

/// Stable wire spelling of a wait condition.
const fn condition_name(condition: WaitCondition) -> &'static str {
    match condition {
        WaitCondition::Exists => "exists",
        WaitCondition::NotExists => "not_exists",
        WaitCondition::Visible => "visible",
        WaitCondition::Enabled => "enabled",
        WaitCondition::Focused => "focused",
        WaitCondition::ValueChanged => "value_changed",
    }
}

// ── screenshot ──────────────────────────────────────────────────────────────

impl Xa11yRuntime {
    /// `op=screenshot`: capture the whole screen (or one region), encode it as
    /// a PNG, write it, and report where it landed together with the
    /// pixel/logical relationship of what was written.
    ///
    /// The invariant this operation exists to keep: geometry arrives in
    /// **logical** pixels — the space `op=hit_test` and `computer_input` speak
    /// — and the file holds **physical** pixels, so the reply carries both
    /// dimensions, the ratio between them, and the logical coordinate of the
    /// image's pixel `(0, 0)`.
    fn op_screenshot(&self, request: &ScreenshotRequest) -> Result<Outcome, Failure> {
        let region = request.region().map(logical_rect);
        let (captured, origin) = self.capture(region)?;
        let captured_at_ms = unix_millis();
        // Logical size comes from the *capture*, before any downscale: the
        // region the caller asked for, or the desktop the backend reported at
        // its own scale. Taking it after a resize would shrink the screen.
        let (logical_width, logical_height) = region.map_or_else(
            || {
                (
                    pixels_to_logical(captured.width, captured.scale),
                    pixels_to_logical(captured.height, captured.scale),
                )
            },
            |rect| (rect.width, rect.height),
        );
        let captured_pixels = (captured.width, captured.height);
        let shot = fit_to_limits(captured, request)?;
        let png = shot.to_png().map_err(|error| Failure::from_xa11y(&error))?;
        let path = screenshot_path(request, captured_at_ms);
        write_new(&path, &png).map_err(|error| {
            Failure::new(
                codes::WRITE_FAILED,
                format!(
                    "could not write {} ({error}); an existing file is never overwritten, so \
                     choose another `path` or remove that file",
                    path.display()
                ),
            )
        })?;
        let data = json!({
            "path": path.display().to_string(),
            "file_bytes": png.len(),
            "image": {
                "pixel_width": shot.width,
                "pixel_height": shot.height,
                "logical_width": logical_width,
                "logical_height": logical_height,
                // Physical pixels per logical pixel *of this file*: downscaling
                // shows up here, so one formula describes every capture:
                // `logical = origin + pixel / scale`.
                "scale": f64::from(shot.width) / f64::from(logical_width),
                "downscaled": (shot.width, shot.height) != captured_pixels,
            },
            "region": match region {
                Some(rect) => json!({
                    "x": rect.x,
                    "y": rect.y,
                    "width": rect.width,
                    "height": rect.height,
                }),
                None => Value::Null,
            },
            "full_screen": region.is_none(),
            "origin": match origin {
                Some(point) => json!({"x": point.x, "y": point.y}),
                None => Value::Null,
            },
            "hint": match origin {
                Some(_) => Value::Null,
                None => json!(FULL_SCREEN_ORIGIN_HINT),
            },
            "captured_at_ms": captured_at_ms,
        });
        Ok(Outcome {
            data,
            meta: json!({ "mime": SCREENSHOT_MIME }),
        })
    }

    /// Captures the whole screen or one region, and reports the logical
    /// coordinate of the image's pixel `(0, 0)` when the backend knows it.
    ///
    /// A region's origin is exact by `ScreenshotProvider::capture_region`'s
    /// contract — the backend captures the pixels *at* that rectangle, so pixel
    /// `(0, 0)` sits at `(x, y)`. A whole-screen capture can report it only
    /// when the backend is injected: `xa11y::screenshot()` drops the origin it
    /// got from `capture_full`, and guessing `(0, 0)` would misplace every
    /// pixel on a desktop that does not start there.
    ///
    /// # Errors
    ///
    /// `permission_denied` when the OS withholds Screen Recording (a human must
    /// grant it), `backend_unavailable` when the capture call itself failed,
    /// `unsupported` where no capture backend exists at all.
    fn capture(&self, region: Option<Rect>) -> Result<(Screenshot, Option<Point>), Failure> {
        match (&self.screenshot, region) {
            (ScreenshotSource::Fixed(provider), Some(rect)) => provider
                .capture_region(rect)
                .map(|shot| (shot, Some(Point::new(rect.x, rect.y))))
                .map_err(|error| Failure::from_xa11y(&error)),
            (ScreenshotSource::Fixed(provider), None) => provider
                .capture_full()
                .map(|(shot, origin)| (shot, Some(origin)))
                .map_err(|error| Failure::from_xa11y(&error)),
            (ScreenshotSource::Global, Some(rect)) => xa11y::screenshot_region(rect)
                .map(|shot| (shot, Some(Point::new(rect.x, rect.y))))
                .map_err(|error| Failure::from_xa11y(&error)),
            (ScreenshotSource::Global, None) => xa11y::screenshot()
                .map(|shot| (shot, None))
                .map_err(|error| Failure::from_xa11y(&error)),
        }
    }
}

/// Rounds a validated logical region into the platform rectangle.
///
/// [`ScreenshotRequest::validate`] has already bounded every field to a finite
/// value in `1..=MAX_CAPTURE_EDGE`, so these casts cannot lose anything.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn logical_rect((x, y, width, height): (f64, f64, f64, f64)) -> Rect {
    Rect {
        x: x.round() as i32,
        y: y.round() as i32,
        width: width.round().max(1.0) as u32,
        height: height.round().max(1.0) as u32,
    }
}

/// Applies the caller's `max_width` / `max_height`.
///
/// The capture is otherwise returned untouched: the request level already
/// bounds a region's logical area, and quietly rescaling a capture nobody asked
/// to shrink would change the very ratio this reply reports.
fn fit_to_limits(shot: Screenshot, request: &ScreenshotRequest) -> Result<Screenshot, Failure> {
    match fit_target(shot.width, shot.height, request) {
        None => Ok(shot),
        Some((width, height)) => shot
            .resize(width, height)
            .map_err(|error| Failure::from_xa11y(&error)),
    }
}

/// The size `width` × `height` has to shrink to, preserving the aspect ratio,
/// or `None` when the caller's limits already allow it.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // floored to 1..=max, then a u32
fn fit_target(width: u32, height: u32, request: &ScreenshotRequest) -> Option<(u32, u32)> {
    if width == 0 || height == 0 || (request.max_width.is_none() && request.max_height.is_none()) {
        return None;
    }
    let (width_f, height_f) = (f64::from(width), f64::from(height));
    let mut factor = 1.0_f64;
    if let Some(max_width) = request.max_width {
        factor = factor.min(f64::from(max_width) / width_f);
    }
    if let Some(max_height) = request.max_height {
        factor = factor.min(f64::from(max_height) / height_f);
    }
    if factor >= 1.0 {
        return None;
    }
    let shrink = |pixels: f64| -> u32 { (pixels * factor).floor().max(1.0) as u32 };
    Some((shrink(width_f).min(width), shrink(height_f).min(height)))
}

/// A physical dimension expressed in logical pixels, never below 1.
///
/// Used for a whole-screen capture, whose logical extent only the backend's own
/// scale describes.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // rounded into 1..=pixels
fn pixels_to_logical(pixels: u32, scale: f32) -> u32 {
    let scale = f64::from(scale);
    if !scale.is_finite() || scale <= 0.0 {
        return pixels.max(1);
    }
    ((f64::from(pixels) / scale).round() as u32).max(1)
}

/// Where this capture is written: the caller's `path` (absolute as given,
/// relative under the default directory), or a fresh unique name there.
///
/// Uniqueness comes from the millisecond stamp plus a counter, so two captures
/// in the same millisecond cannot land on the same file and turn into a
/// `write_failed`.
fn screenshot_path(request: &ScreenshotRequest, captured_at_ms: u64) -> PathBuf {
    match request.path.as_deref().map(str::trim) {
        Some(path) if !path.is_empty() => {
            let path = Path::new(path);
            if path.is_absolute() {
                path.to_path_buf()
            } else {
                screenshot_dir().join(path)
            }
        }
        _ => {
            let sequence = SCREENSHOT_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            screenshot_dir().join(format!("screenshot-{captured_at_ms}-{sequence}.png"))
        }
    }
}

/// Default landing directory for captures: `<system temp>/tizi-screenshots`.
fn screenshot_dir() -> PathBuf {
    std::env::temp_dir().join(SCREENSHOT_DIR_NAME)
}

/// Writes a brand-new file, creating its directory first.
///
/// `create_new` rather than `create`: a capture that reused a name would
/// silently destroy whatever was already there, and the model would believe it
/// had just taken that picture.
fn write_new(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(bytes)?;
    file.flush()
}

/// Milliseconds since the Unix epoch, shared by the capture stamp and the
/// default file name.
fn unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| {
            u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX)
        })
}

// ── element rendering ───────────────────────────────────────────────────────

/// Stable `snake_case` role name.
fn role_name(data: &ElementData) -> &'static str {
    data.role.to_snake_case()
}

/// Small element rendering: enough to act on, cheap to repeat.
fn element_summary(handle: &str, data: &ElementData) -> Value {
    json!({
        "id": handle,
        "role": role_name(data),
        "name": data.name.clone(),
        "value": data.value.clone(),
        "bounds": bounds_value(data),
        "enabled": data.states.enabled,
        "visible": data.states.visible,
        "focused": data.states.focused,
    })
}

/// Full element rendering used by `op=inspect` and `op=hit_test`.
fn element_detail(handle: &str, data: &ElementData) -> Value {
    json!({
        "id": handle,
        "role": role_name(data),
        "name": data.name.clone(),
        "description": data.description.clone(),
        "value": data.value.clone(),
        "bounds": bounds_value(data),
        "enabled": data.states.enabled,
        "focused": data.states.focused,
        "visible": data.states.visible,
        "states": states_value(data),
        // The plan's key affordance: never let the model guess whether an
        // element supports an action — tell it.
        "actions": data.actions.clone(),
        "numeric_value": data.numeric_value,
        "min_value": data.min_value,
        "max_value": data.max_value,
        "stable_id": data.stable_id.clone(),
        "pid": data.pid,
    })
}

/// Bounds as a JSON object, or `null` for an element without geometry.
fn bounds_value(data: &ElementData) -> Value {
    data.bounds.map_or(Value::Null, |bounds| {
        json!({
            "x": bounds.x,
            "y": bounds.y,
            "width": bounds.width,
            "height": bounds.height,
        })
    })
}

/// Every state flag a backend reports, so "why did the action fail" is
/// answerable from one reply.
fn states_value(data: &ElementData) -> Value {
    let states = &data.states;
    json!({
        "active": states.active,
        "selected": states.selected,
        "editable": states.editable,
        "focusable": states.focusable,
        "modal": states.modal,
        "required": states.required,
        "busy": states.busy,
        "checked": states.checked.map(|checked| match checked {
            xa11y::Toggled::On => "on",
            xa11y::Toggled::Off => "off",
            xa11y::Toggled::Mixed => "mixed",
        }),
        "expanded": states.expanded,
        "minimized": states.minimized,
        "maximized": states.maximized,
        "fullscreen": states.fullscreen,
    })
}

impl crate::AccessibilityRuntime for Xa11yRuntime {
    fn execute(&self, request: AccessibilityRequest) -> crate::ToolFuture<'_> {
        Box::pin(async move {
            let runtime = self.clone();
            match tokio::task::spawn_blocking(move || runtime.run(&request)).await {
                Ok(value) => Ok(value),
                Err(join_error) => {
                    warn!(error = %join_error, "accessibility worker failed");
                    Ok(envelope::internal_error(format!(
                        "the accessibility worker stopped unexpectedly: {join_error}"
                    )))
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn element(name: Option<&str>, value: Option<&str>, role: Role) -> ElementData {
        let mut data = ElementData::for_role(role);
        data.name = name.map(str::to_string);
        data.value = value.map(str::to_string);
        data
    }

    fn selector(json: Value) -> Selector {
        serde_json::from_value(json).expect("selector must parse")
    }

    #[test]
    fn an_empty_selector_matches_every_element() {
        let empty = Selector::default();
        assert!(selector_matches(
            &empty,
            &element(Some("anything"), None, Role::Button)
        ));
    }

    #[test]
    fn name_matching_is_exact_while_name_contains_ignores_case() {
        let exact = selector(json!({"name": "Submit"}));
        assert!(selector_matches(
            &exact,
            &element(Some("Submit"), None, Role::Button)
        ));
        assert!(!selector_matches(
            &exact,
            &element(Some("Submit now"), None, Role::Button)
        ));

        let partial = selector(json!({"name_contains": "subm"}));
        assert!(selector_matches(
            &partial,
            &element(Some("Submit now"), None, Role::Button)
        ));
        assert!(!selector_matches(
            &partial,
            &element(Some("Cancel"), None, Role::Button)
        ));
    }

    #[test]
    fn roles_compare_by_normalized_name() {
        let button = selector(json!({"role": "button"}));
        assert!(selector_matches(
            &button,
            &element(None, None, Role::Button)
        ));
        assert!(!selector_matches(
            &button,
            &element(None, None, Role::TextField)
        ));

        let text_field = selector(json!({"role": "text_field"}));
        assert!(selector_matches(
            &text_field,
            &element(None, None, Role::TextField)
        ));
    }

    #[test]
    fn state_fields_match_both_polarities() {
        let mut disabled = element(None, None, Role::Button);
        disabled.states.enabled = false;
        let mut hidden = element(None, None, Role::StaticText);
        hidden.states.visible = false;

        assert!(selector_matches(
            &selector(json!({"enabled": false})),
            &disabled
        ));
        assert!(!selector_matches(
            &selector(json!({"enabled": true})),
            &disabled
        ));
        assert!(selector_matches(
            &selector(json!({"visible": false})),
            &hidden
        ));
    }

    #[test]
    fn value_and_description_are_exact_matches() {
        let mut data = element(Some("Search"), Some("hello"), Role::TextField);
        data.description = Some("Search field".into());
        assert!(selector_matches(
            &selector(json!({"value": "hello", "description": "Search field"})),
            &data
        ));
        assert!(!selector_matches(
            &selector(json!({"value": "hell"})),
            &data
        ));
        assert!(selector_matches(
            &selector(json!({"value": "hello", "name_contains": "SEARCH"})),
            &data
        ));
    }

    #[test]
    fn hit_test_geometry_is_half_open_on_the_far_edge() {
        let rect = Rect {
            x: 100,
            y: 50,
            width: 800,
            height: 600,
        };
        assert!(rect_contains(&rect, 100.0, 50.0));
        assert!(rect_contains(&rect, 899.0, 649.0));
        assert!(!rect_contains(&rect, 900.0, 650.0));
        assert!(!rect_contains(&rect, 99.0, 50.0));
    }

    #[test]
    fn the_deepest_and_then_smallest_element_wins_a_hit_test() {
        let window = {
            let mut data = element(Some("Main Window"), None, Role::Window);
            data.bounds = Some(Rect {
                x: 0,
                y: 0,
                width: 1000,
                height: 800,
            });
            data
        };
        let button = {
            let mut data = element(Some("Submit"), None, Role::Button);
            data.bounds = Some(Rect {
                x: 10,
                y: 10,
                width: 100,
                height: 40,
            });
            data
        };
        let mut best = None;
        consider(&window, 1, 20.0, 20.0, &mut best);
        consider(&button, 2, 20.0, 20.0, &mut best);
        assert_eq!(
            best.expect("a candidate").data.name.as_deref(),
            Some("Submit")
        );
    }

    #[test]
    fn invisible_elements_never_win_a_hit_test() {
        let mut hidden = element(Some("Hidden"), None, Role::Button);
        hidden.states.visible = false;
        hidden.bounds = Some(Rect {
            x: 0,
            y: 0,
            width: 10,
            height: 10,
        });
        let mut best = None;
        consider(&hidden, 3, 5.0, 5.0, &mut best);
        assert!(best.is_none());
    }

    #[test]
    fn the_node_budget_reports_truncation() {
        let mut budget = Budget::new(2);
        assert!(budget.take());
        assert!(budget.take());
        assert!(!budget.take());
        assert!(budget.truncated);
    }

    #[test]
    fn deadlines_expire_and_fail_with_timeout() {
        let future = Deadline::after(Duration::from_secs(30));
        assert!(!future.expired());
        assert!(future.check("query").is_ok());

        let past = Deadline::after(Duration::ZERO);
        let failure = past.check("query").expect_err("must expire");
        assert_eq!(failure.code, codes::TIMEOUT);
        assert!(failure.retryable);
    }

    #[test]
    fn wait_conditions_read_the_observed_element() {
        let mut data = element(Some("Status"), Some("Loading..."), Role::StaticText);
        data.states.visible = false;
        assert!(wait_condition_met(WaitCondition::Exists, Some(&data), None));
        assert!(wait_condition_met(WaitCondition::NotExists, None, None));
        assert!(wait_condition_met(
            WaitCondition::Enabled,
            Some(&data),
            None
        ));
        assert!(!wait_condition_met(
            WaitCondition::Visible,
            Some(&data),
            None
        ));
        assert!(wait_condition_met(
            WaitCondition::ValueChanged,
            Some(&data),
            Some("Ready")
        ));
        assert!(!wait_condition_met(
            WaitCondition::ValueChanged,
            Some(&data),
            Some("Loading...")
        ));
    }

    #[test]
    fn an_absent_application_satisfies_not_exists() {
        let missing = Failure::new(codes::APP_NOT_FOUND, "gone");
        assert!(awaiting_absence(WaitCondition::NotExists, &missing));
        assert!(!awaiting_absence(WaitCondition::Exists, &missing));
        assert!(!awaiting_absence(
            WaitCondition::NotExists,
            &Failure::new(codes::PERMISSION_DENIED, "denied")
        ));
    }

    #[test]
    fn presentation_matching_ignores_a_changed_value() {
        let mut before = element(Some("Search"), Some("hello"), Role::TextField);
        before.bounds = Some(Rect {
            x: 10,
            y: 20,
            width: 100,
            height: 20,
        });
        let mut after = before.clone();
        after.value = Some("world".into());
        after.handle = 999;
        assert!(same_presentation(&before, &after));

        let mut moved = before.clone();
        moved.bounds = Some(Rect {
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        });
        assert!(!same_presentation(&before, &moved));
    }

    #[test]
    fn action_payloads_are_rendered_as_platform_strings() {
        assert_eq!(action_text(&json!("hello")), "hello");
        assert_eq!(action_text(&json!(75)), "75");
        assert_eq!(action_text(&json!(true)), "true");
    }

    #[test]
    fn tree_lines_name_the_role_and_quote_the_text() {
        let mut data = element(Some("Submit"), Some("now"), Role::Button);
        assert_eq!(describe(&data), "button \"Submit\" = \"now\"");
        data.name = None;
        data.value = None;
        assert_eq!(describe(&data), "button");
    }
}
