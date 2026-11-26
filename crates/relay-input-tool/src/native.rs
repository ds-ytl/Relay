//! In-process runtime that turns one validated action into real input events.
//!
//! [`NativeInputRuntime`] is the production [`InputRuntime`](crate::InputRuntime):
//! it opens a platform backend, reads the current pointer position, plans the
//! human-like event stream, and replays it with real pauses. No helper process,
//! plugin CLI, or shell-out is involved.
//!
//! Two consequences are worth stating up front:
//!
//! * **Everything blocking runs on tokio's blocking pool.** Planning, sleeping,
//!   and enigo itself are synchronous, so the async loop never stalls and a
//!   wedged platform call cannot freeze the server.
//! * **The reply is small on purpose.** A click can be thirty pointer samples,
//!   but the model only receives the landing point, the event count, and the
//!   elapsed time; the samples go to the debug log.
//!
//! The backend is injectable, which is how the runtime is tested end to end —
//! plan, replay, and envelope — without touching a real desktop.

use std::{sync::Arc, time::Instant};

use serde_json::{Value, json};
use tracing::{debug, warn};

use crate::backend::{BackendError, EnigoFactory, InputBackend, InputBackendFactory};
use crate::envelope::{self, codes};
use crate::plan::{self, Child, Event, Plan};
use crate::request::{Action, ActionKind, Point, key_label};

/// The production input runtime.
///
/// Cloneable and shareable: the only state is the backend factory, and each
/// call creates its own connection on its own thread.
#[derive(Clone)]
pub struct NativeInputRuntime {
    factory: Arc<dyn InputBackendFactory>,
}

impl NativeInputRuntime {
    /// Creates a runtime over the real platform backend.
    ///
    /// Constructing the runtime never touches the platform: the connection is
    /// opened per call, so a machine without a desktop still starts the server.
    #[must_use]
    pub fn new() -> Self {
        Self::with_factory(Arc::new(EnigoFactory))
    }

    /// Creates a runtime over an injected backend factory.
    #[must_use]
    pub fn with_factory(factory: Arc<dyn InputBackendFactory>) -> Self {
        Self { factory }
    }

    /// Plans and replays one action on the calling thread.
    ///
    /// Every outcome — including "no desktop" and "permission denied" — is a
    /// JSON envelope; this method does not fail.
    #[must_use]
    pub fn run(&self, action: &Action) -> Value {
        let started = Instant::now();
        let mut backend = match self.factory.open() {
            Ok(backend) => backend,
            Err(error) => return failure(&error),
        };
        let cursor = current_position(backend.as_ref());
        debug!(
            action = action.label(),
            behavior = action.behavior.label(),
            ?cursor,
            "planning human-like input"
        );
        let mut rng = rand::rng();
        let plan = plan::plan(action, cursor, &mut rng);
        if let Err(error) = replay(backend.as_mut(), &plan) {
            return failure(&error);
        }
        envelope::success(data(action, &plan), meta(action, started))
    }
}

impl Default for NativeInputRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::InputRuntime for NativeInputRuntime {
    fn execute(&self, action: Action) -> crate::ToolFuture<'_> {
        Box::pin(async move {
            let runtime = self.clone();
            match tokio::task::spawn_blocking(move || runtime.run(&action)).await {
                Ok(value) => Ok(value),
                Err(join_error) => {
                    warn!(error = %join_error, "input worker failed");
                    Ok(envelope::failure(
                        codes::INTERNAL_ERROR,
                        format!("the input worker stopped unexpectedly: {join_error}"),
                    ))
                }
            }
        })
    }
}

/// Replays the plan with real pauses between events.
fn replay(backend: &mut dyn InputBackend, plan: &Plan) -> Result<(), BackendError> {
    debug!(steps = plan.steps.len(), "replaying input plan");
    for step in &plan.steps {
        if !step.delay.is_zero() {
            std::thread::sleep(step.delay);
        }
        match &step.event {
            // A pause is the delay that already elapsed: nothing to emit.
            Event::Pause => {}
            event => {
                if let Event::Move { x, y } = event {
                    // The design is explicit: intermediate samples stay out of
                    // the model's context and only reach the debug log.
                    debug!(x, y, "pointer sample");
                }
                backend.emit(event)?;
            }
        }
    }
    Ok(())
}

/// Best effort: a platform that cannot report the pointer still accepts input,
/// in which case the first step is a plain jump to the destination.
fn current_position(backend: &dyn InputBackend) -> Option<Point> {
    match backend.position() {
        Ok(point) => Some(point),
        Err(error) => {
            debug!(error = error.message(), "pointer position unavailable");
            None
        }
    }
}

fn point_json(point: Point) -> Value {
    json!({"x": point.x, "y": point.y})
}

fn failure(error: &BackendError) -> Value {
    envelope::failure(error.code(), error.message())
}

/// What the model gets back: where the action landed and what it did, never the
/// intermediate samples.
fn data(action: &Action, plan: &Plan) -> Value {
    #[allow(clippy::cast_possible_truncation)]
    let mut data = json!({
        "action": action.label(),
        "behavior": action.behavior.label(),
        "events": plan.events(),
        "duration_ms": plan.duration().as_millis() as u64,
    });
    if let Some(point) = plan.position {
        data["position"] = point_json(point);
    }
    if let Some(point) = plan.origin {
        data["from"] = point_json(point);
    }
    match &action.kind {
        ActionKind::Click { button, count, .. } => {
            data["button"] = json!(button.label());
            data["clicks"] = json!(count);
        }
        ActionKind::Scroll {
            direction, amount, ..
        } => {
            data["direction"] = json!(direction.label());
            data["amount"] = json!(amount);
        }
        ActionKind::Type { text } => {
            data["chars"] = json!(text.chars().count());
        }
        ActionKind::Key(key) => {
            data["key"] = json!(key_label(*key));
        }
        ActionKind::Hotkey(keys) => {
            data["keys"] = json!(keys.iter().map(|key| key_label(*key)).collect::<Vec<_>>());
        }
        ActionKind::Sequence(_) => {
            data["actions"] = Value::Array(plan.children.iter().map(child_json).collect());
        }
        ActionKind::Move { .. } | ActionKind::Drag { .. } => {}
    }
    data
}

fn child_json(child: &Child) -> Value {
    #[allow(clippy::cast_possible_truncation)]
    let mut value = json!({
        "action": child.action,
        "duration_ms": child.duration.as_millis() as u64,
    });
    if let Some(position) = child.position {
        value["position"] = point_json(position);
    }
    if let Some(chars) = child.chars {
        value["chars"] = json!(chars);
    }
    if let Some(key) = &child.key {
        value["key"] = json!(key);
    }
    value
}

fn meta(action: &Action, started: Instant) -> Value {
    #[allow(clippy::cast_possible_truncation)]
    let elapsed_ms = started.elapsed().as_millis() as u64;
    json!({
        "op": action.label(),
        "behavior": action.behavior.label(),
        "elapsed_ms": elapsed_ms,
    })
}

/// Records events instead of emitting them, for tests.
#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use crate::request::{Behavior, MouseButton, ScrollDirection, Target};

    use super::*;

    /// A backend that appends every event to a shared log.
    struct Recorder {
        log: Arc<Mutex<Vec<Event>>>,
        cursor: Option<Point>,
        refuse: Option<BackendError>,
    }

    impl InputBackend for Recorder {
        fn emit(&mut self, event: &Event) -> Result<(), BackendError> {
            if let Some(error) = &self.refuse {
                return Err(error.clone());
            }
            self.log.lock().unwrap().push(event.clone());
            Ok(())
        }

        fn position(&self) -> Result<Point, BackendError> {
            self.cursor
                .ok_or_else(|| BackendError::Unavailable("no pointer".into()))
        }
    }

    /// A factory that hands out recorders, and can fail to open or to emit.
    struct RecorderFactory {
        log: Arc<Mutex<Vec<Event>>>,
        cursor: Option<Point>,
        refuse: Option<BackendError>,
        open_error: Option<BackendError>,
    }

    impl RecorderFactory {
        fn new(cursor: Option<Point>) -> Self {
            Self {
                log: Arc::new(Mutex::new(Vec::new())),
                cursor,
                refuse: None,
                open_error: None,
            }
        }

        /// A factory that cannot open a backend at all.
        fn failing(error: BackendError) -> Self {
            Self {
                open_error: Some(error),
                ..Self::new(None)
            }
        }

        /// A factory whose backend refuses every event.
        fn refusing(error: BackendError) -> Self {
            Self {
                refuse: Some(error),
                ..Self::new(Some(Point { x: 0, y: 0 }))
            }
        }

        fn events(&self) -> Vec<Event> {
            self.log.lock().unwrap().clone()
        }
    }

    impl InputBackendFactory for RecorderFactory {
        fn open(&self) -> Result<Box<dyn InputBackend>, BackendError> {
            if let Some(error) = &self.open_error {
                return Err(error.clone());
            }
            Ok(Box::new(Recorder {
                log: Arc::clone(&self.log),
                cursor: self.cursor,
                refuse: self.refuse.clone(),
            }))
        }
    }

    fn action(kind: ActionKind) -> Action {
        Action {
            behavior: Behavior::Fast,
            kind,
        }
    }

    fn fast_rect() -> Target {
        Target {
            x: 100,
            y: 100,
            width: Some(40),
            height: Some(20),
        }
    }

    fn point(x: i32, y: i32) -> Point {
        Point { x, y }
    }

    fn rect(x: i32, y: i32, width: i32, height: i32) -> Target {
        Target {
            x,
            y,
            width: Some(width),
            height: Some(height),
        }
    }

    fn parse(name: &str) -> enigo::Key {
        crate::request::parse_key(name).expect("test key must parse")
    }

    fn moves(events: &[Event]) -> Vec<(i32, i32)> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Move { x, y } => Some((*x, *y)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_click_reports_the_landing_point_and_hides_the_samples() {
        let factory = Arc::new(RecorderFactory::new(Some(point(50, 50))));
        let runtime = NativeInputRuntime::with_factory(factory.clone());
        let reply = runtime.run(&action(ActionKind::Click {
            target: fast_rect(),
            button: MouseButton::Left,
            count: 1,
        }));

        assert_eq!(reply["ok"], json!(true));
        assert_eq!(reply["data"]["action"], json!("click"));
        assert_eq!(reply["data"]["button"], json!("left"));
        assert_eq!(reply["data"]["clicks"], json!(1));
        assert_eq!(reply["meta"]["op"], json!("click"));
        assert_eq!(reply["meta"]["behavior"], json!("fast"));
        let position = &reply["data"]["position"];
        let x = position["x"].as_i64().expect("x");
        let y = position["y"].as_i64().expect("y");
        assert!((102..=138).contains(&x), "{position}");
        assert!((102..=118).contains(&y), "{position}");

        let events = factory.events();
        // The approach starts next to the cursor and ends on the landing point.
        let (start_x, start_y) = moves(&events)[0];
        assert!(
            (0..100).contains(&start_x) && (0..100).contains(&start_y),
            "the approach did not start at the cursor"
        );
        assert_eq!(
            moves(&events).last().copied(),
            Some((i32::try_from(x).unwrap(), i32::try_from(y).unwrap()))
        );
        assert_eq!(
            events.last().cloned(),
            Some(Event::Button {
                button: MouseButton::Left,
                press: false
            })
        );
        // The reply never carries the intermediate samples.
        assert!(reply["data"].get("samples").is_none());
        assert!(reply["data"].get("steps").is_none());
    }

    #[test]
    fn a_missing_desktop_is_reported_instead_of_panicking() {
        let factory = Arc::new(RecorderFactory::failing(BackendError::Unavailable(
            "no display server".into(),
        )));
        let runtime = NativeInputRuntime::with_factory(factory.clone());
        let reply = runtime.run(&action(ActionKind::Move {
            target: fast_rect(),
        }));

        assert_eq!(reply["ok"], json!(false));
        assert_eq!(reply["error"]["code"], json!(codes::BACKEND_UNAVAILABLE));
        assert_eq!(reply["error"]["retryable"], json!(true));
        assert!(
            reply["error"]["message"]
                .as_str()
                .expect("message")
                .contains("display")
        );
        assert!(factory.events().is_empty());
    }

    #[test]
    fn a_denied_permission_is_reported_as_not_retryable() {
        let factory = Arc::new(RecorderFactory::failing(BackendError::PermissionDenied(
            "grant Accessibility to the server process".into(),
        )));
        let runtime = NativeInputRuntime::with_factory(factory);
        let reply = runtime.run(&action(ActionKind::Move {
            target: fast_rect(),
        }));

        assert_eq!(reply["error"]["code"], json!(codes::PERMISSION_DENIED));
        assert_eq!(reply["error"]["retryable"], json!(false));
    }

    #[test]
    fn an_event_refused_midway_is_reported_and_stops_the_replay() {
        let factory = Arc::new(RecorderFactory::refusing(BackendError::Rejected(
            "the platform refused the event".into(),
        )));
        let runtime = NativeInputRuntime::with_factory(factory.clone());
        let reply = runtime.run(&action(ActionKind::Click {
            target: fast_rect(),
            button: MouseButton::Left,
            count: 1,
        }));

        assert_eq!(reply["ok"], json!(false));
        assert_eq!(reply["error"]["code"], json!(codes::INPUT_REJECTED));
        assert_eq!(reply["error"]["retryable"], json!(true));
        assert!(factory.events().is_empty());
    }

    #[test]
    fn a_scroll_reports_the_wheel_notches_that_were_emitted() {
        let factory = Arc::new(RecorderFactory::new(Some(point(0, 0))));
        let runtime = NativeInputRuntime::with_factory(factory.clone());
        let reply = runtime.run(&action(ActionKind::Scroll {
            target: None,
            direction: ScrollDirection::Down,
            amount: 4,
        }));

        assert_eq!(reply["data"]["direction"], json!("down"));
        assert_eq!(reply["data"]["amount"], json!(4));
        let notches: i32 = factory
            .events()
            .iter()
            .filter_map(|event| match event {
                Event::Scroll { steps } => Some(*steps),
                _ => None,
            })
            .sum();
        assert_eq!(notches, 4);
    }

    #[test]
    fn a_sequence_reports_each_action_without_echoing_the_text() {
        let factory = Arc::new(RecorderFactory::new(Some(point(10, 10))));
        let runtime = NativeInputRuntime::with_factory(factory.clone());
        let reply = runtime.run(&Action {
            behavior: Behavior::Fast,
            kind: ActionKind::Sequence(vec![
                action(ActionKind::Click {
                    target: rect(100, 100, 40, 20),
                    button: MouseButton::Left,
                    count: 1,
                }),
                action(ActionKind::Type {
                    text: "hello@example.com".to_owned(),
                }),
                action(ActionKind::Key(parse("Enter"))),
            ]),
        });

        assert_eq!(reply["data"]["action"], json!("sequence"));
        let actions = reply["data"]["actions"].as_array().expect("actions");
        assert_eq!(actions.len(), 3);
        assert_eq!(actions[0]["action"], json!("click"));
        assert!(actions[0]["position"]["x"].is_i64());
        assert_eq!(actions[1]["action"], json!("type"));
        assert_eq!(actions[1]["chars"], json!(17));
        assert_eq!(actions[2]["action"], json!("key"));
        assert_eq!(actions[2]["key"], json!("enter"));
        // The typed text is reported by length, never echoed back.
        assert!(!reply["data"].to_string().contains("hello@example.com"));
    }

    #[tokio::test]
    async fn the_async_entry_point_returns_the_same_envelope() {
        use crate::InputRuntime;

        let factory = Arc::new(RecorderFactory::new(Some(point(5, 5))));
        let runtime = NativeInputRuntime::with_factory(factory.clone());
        let reply = runtime
            .execute(action(ActionKind::Move {
                target: fast_rect(),
            }))
            .await
            .expect("the runtime answers with an envelope");

        assert_eq!(reply["ok"], json!(true));
        assert_eq!(reply["data"]["action"], json!("move"));
        assert!(reply["meta"]["elapsed_ms"].as_u64().expect("elapsed") < 5_000);
        assert!(!factory.events().is_empty());
    }

    #[test]
    fn pauses_do_not_reach_the_backend() {
        let factory = Arc::new(RecorderFactory::new(Some(point(0, 0))));
        let runtime = NativeInputRuntime::with_factory(factory.clone());
        let _ = runtime.run(&action(ActionKind::Click {
            target: fast_rect(),
            button: MouseButton::Left,
            count: 2,
        }));
        assert!(
            !factory.events().contains(&Event::Pause),
            "a pause is a delay, not an event"
        );
    }

    #[test]
    fn a_drag_reports_both_ends() {
        let factory = Arc::new(RecorderFactory::new(Some(point(0, 0))));
        let runtime = NativeInputRuntime::with_factory(factory);
        let reply = runtime.run(&action(ActionKind::Drag {
            from: point_target(20, 20),
            to: point_target(60, 40),
            button: MouseButton::Left,
        }));
        assert_eq!(reply["data"]["action"], json!("drag"));
        assert!(reply["data"]["from"]["x"].is_i64());
        assert!(reply["data"]["position"]["y"].is_i64());
    }

    fn point_target(x: i32, y: i32) -> Target {
        Target {
            x,
            y,
            width: None,
            height: None,
        }
    }

    #[test]
    fn durations_are_milliseconds_and_never_negative() {
        let factory = Arc::new(RecorderFactory::new(Some(point(0, 0))));
        let runtime = NativeInputRuntime::with_factory(factory);
        let reply = runtime.run(&action(ActionKind::Move {
            target: fast_rect(),
        }));
        assert!(reply["data"]["duration_ms"].is_u64());
        assert!(reply["meta"]["elapsed_ms"].is_u64());
    }
}
