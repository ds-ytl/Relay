//! Human-like input planner: one intent in, timed low-level events out.
//!
//! This module is pure computation — no platform call, no sleeping, no global
//! state — so the "human" part of `computer_input` is unit-testable on a
//! headless machine. The runtime plans with a real random source and a real
//! cursor position; a test plans with a seeded generator and an arbitrary
//! cursor, then asserts on the resulting events.
//!
//! Four models cover the whole surface:
//!
//! * **Pointer motion** — an approach is a bowed path sampled with a
//!   smoothstep ease (fast start, faster middle, slow arrival) plus a sub-pixel
//!   tremor, while a drag uses a slower, straighter, steadier path. A move
//!   never teleports: the only instant jump is a move of less than a pixel.
//! * **Target selection** — a rectangle target is first shrunk into a safe zone
//!   (never the frame of a control), then a pixel is drawn inside it, mildly
//!   biased towards the centre. `precise` collapses to the exact centre.
//! * **Typing** — one gap per character, jittered around 150 WPM, longer for
//!   shifted characters, with word and sentence pauses. Uppercase and shifted
//!   punctuation are typed as a real Shift press plus a base key.
//! * **Action timing** — a pre-click settle, a real press duration, a double
//!   click inside the OS threshold at the same pixel, and a gap between the
//!   actions of a sequence so the UI can react.
//!
//! The plan deliberately reports only what the model needs: the landing point
//! and the duration. The intermediate samples go to the debug log.

use std::time::Duration;

use enigo::Key;
use rand::Rng;

use crate::request::{Action, ActionKind, Behavior, MouseButton, Point, Target, is_modifier};

/// Pointer speed of a human hand, in pixels per second.
const MOVE_SPEED_RANGE: (f64, f64) = (700.0, 1500.0);

/// Dragging is slower and steadier than free motion.
const DRAG_SPEED_RANGE: (f64, f64) = (320.0, 620.0);

/// Typing baseline: 150 WPM, i.e. 80 ms per character before jitter.
const BASE_CHAR_SECONDS: f64 = 0.08;

/// Upper bound for any single planned pause, in seconds.
const MAX_PAUSE_SECONDS: f64 = 600.0;

/// Even a cross-screen move must not occupy a call for longer than this.
const MAX_MOVE_SECONDS: f64 = 5.0;

/// Characters injected per text event when the keyboard cannot express them.
const TEXT_CHUNK_CHARS: usize = 24;

/// One low-level event with the pause that precedes it.
#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    /// Pause observed before the event is emitted.
    pub delay: Duration,
    /// What to emit.
    pub event: Event,
}

/// One synthetic input event, in the platform-independent vocabulary.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// Move the pointer to an absolute position.
    Move {
        /// Horizontal position.
        x: i32,
        /// Vertical position.
        y: i32,
    },
    /// Press or release a mouse button.
    Button {
        /// Which button.
        button: MouseButton,
        /// `true` presses, `false` releases.
        press: bool,
    },
    /// Press or release a key.
    Key {
        /// Which key.
        key: Key,
        /// `true` presses, `false` releases.
        press: bool,
    },
    /// Inject text as one platform text event, bypassing the key layout.
    Text {
        /// The characters to enter.
        text: String,
    },
    /// Turn the wheel: positive scrolls down, negative up.
    Scroll {
        /// Signed number of wheel notches.
        steps: i32,
    },
    /// Nothing to emit; the pause alone is the point.
    Pause,
}

/// What one action inside a sequence did, for the result envelope.
#[derive(Clone, Debug, PartialEq)]
pub struct Child {
    /// Operation label, e.g. `click`.
    pub action: &'static str,
    /// Planned duration of the action.
    pub duration: Duration,
    /// Where the action left the pointer, when it moved it.
    pub position: Option<Point>,
    /// Characters typed, for `type`.
    pub chars: Option<usize>,
    /// Key tapped, for `key`.
    pub key: Option<String>,
}

/// One planned action: its timed steps plus the facts worth reporting.
#[derive(Clone, Debug, Default)]
pub struct Plan {
    /// The steps, in emission order.
    pub steps: Vec<Step>,
    /// Where the pointer ended up, when the action moved it.
    pub position: Option<Point>,
    /// Where a drag started, when the action dragged.
    pub origin: Option<Point>,
    /// Per-action summaries, populated by `op=sequence`.
    pub children: Vec<Child>,
}

impl Plan {
    /// Total planned duration: the sum of every step's pause.
    #[must_use]
    pub fn duration(&self) -> Duration {
        self.steps.iter().map(|step| step.delay).sum()
    }

    /// Number of events a backend will actually receive (pauses excluded).
    #[must_use]
    pub fn events(&self) -> usize {
        self.steps
            .iter()
            .filter(|step| step.event != Event::Pause)
            .count()
    }
}

/// Humanization profile: the numbers behind `fast`, `natural`, `precise`.
#[derive(Clone, Copy, Debug)]
struct Profile {
    /// Multiplies pointer, typing, and pause speed; below one is slower.
    speed: f64,
    /// Scales hand tremor and timing jitter.
    jitter: f64,
    /// `0` aims at the exact centre, `1` spreads across the whole safe zone.
    spread: f64,
    /// Spacing of sampled pointer positions, in pixels.
    step_px: f64,
}

impl Profile {
    const fn of(behavior: Behavior) -> Self {
        match behavior {
            Behavior::Fast => Self {
                speed: 2.2,
                jitter: 0.5,
                spread: 1.0,
                step_px: 14.0,
            },
            Behavior::Natural => Self {
                speed: 1.0,
                jitter: 1.0,
                spread: 0.75,
                step_px: 9.0,
            },
            Behavior::Precise => Self {
                speed: 0.55,
                jitter: 0.2,
                spread: 0.0,
                step_px: 6.0,
            },
        }
    }

    fn speed_factor(self) -> f64 {
        if self.speed > 0.0 { self.speed } else { 1.0 }
    }

    /// One jittered pause from a range, scaled by the profile.
    fn pause<R: Rng>(self, rng: &mut R, low: f64, high: f64) -> Duration {
        seconds(rng.random_range(low..high) / self.speed_factor())
    }
}

/// Shape of one pointer path.
#[derive(Clone, Copy, Debug)]
struct TrackStyle {
    /// Speed range in pixels per second.
    speed: (f64, f64),
    /// Lateral bow, as a fraction of the distance.
    bend: (f64, f64),
    /// Hard cap on the bow, in pixels.
    bend_cap: f64,
    /// Scales the per-step tremor.
    jitter: f64,
}

/// A hand approaching a target: a visible arc with a settling tremor.
const APPROACH: TrackStyle = TrackStyle {
    speed: MOVE_SPEED_RANGE,
    bend: (0.05, 0.18),
    bend_cap: 160.0,
    jitter: 1.0,
};

/// A hand holding a button: slow, straight, and steady.
const GLIDE: TrackStyle = TrackStyle {
    speed: DRAG_SPEED_RANGE,
    bend: (0.01, 0.05),
    bend_cap: 40.0,
    jitter: 0.4,
};

/// Plans one validated action, starting from the current pointer position.
///
/// `cursor` is the pointer position the runtime read before planning; when the
/// platform cannot report it, the plan simply starts at the destination and the
/// first step becomes a plain jump.
#[must_use]
pub fn plan<R: Rng>(action: &Action, cursor: Option<Point>, rng: &mut R) -> Plan {
    let mut plan = Plan::default();
    let profile = Profile::of(action.behavior);
    let outcome = plan_into(
        action,
        cursor,
        profile,
        rng,
        &mut plan.steps,
        &mut plan.children,
    );
    plan.position = outcome.position;
    plan.origin = outcome.origin;
    plan
}

/// Where an action left the pointer.
#[derive(Clone, Copy, Debug, Default)]
struct Outcome {
    position: Option<Point>,
    origin: Option<Point>,
}

#[allow(clippy::too_many_lines)]
fn plan_into<R: Rng>(
    action: &Action,
    cursor: Option<Point>,
    profile: Profile,
    rng: &mut R,
    steps: &mut Vec<Step>,
    children: &mut Vec<Child>,
) -> Outcome {
    match &action.kind {
        ActionKind::Move { target } => {
            let point = resolve(*target, profile, rng);
            let from = cursor.unwrap_or(point);
            track(from, point, APPROACH, profile.step_px, profile, rng, steps);
            Outcome {
                position: Some(point),
                origin: None,
            }
        }
        ActionKind::Click {
            target,
            button,
            count,
        } => {
            let point = resolve(*target, profile, rng);
            let from = cursor.unwrap_or(point);
            track(from, point, APPROACH, profile.step_px, profile, rng, steps);
            // A hand settles before it presses, and presses for a real moment.
            for index in 0..*count {
                let delay = if index == 0 {
                    profile.pause(rng, 0.06, 0.18)
                } else {
                    // Still inside the OS double-click threshold, on the same
                    // pixel: some toolkits reject a burst that drifts.
                    profile.pause(rng, 0.09, 0.17)
                };
                steps.push(Step {
                    delay,
                    event: Event::Button {
                        button: *button,
                        press: true,
                    },
                });
                steps.push(Step {
                    delay: profile.pause(rng, 0.045, 0.095),
                    event: Event::Button {
                        button: *button,
                        press: false,
                    },
                });
            }
            Outcome {
                position: Some(point),
                origin: None,
            }
        }
        ActionKind::Drag { from, to, button } => {
            let origin = resolve(*from, profile, rng);
            let destination = resolve(*to, profile, rng);
            track(
                cursor.unwrap_or(origin),
                origin,
                APPROACH,
                profile.step_px,
                profile,
                rng,
                steps,
            );
            steps.push(Step {
                delay: profile.pause(rng, 0.08, 0.22),
                event: Event::Button {
                    button: *button,
                    press: true,
                },
            });
            track(
                origin,
                destination,
                GLIDE,
                profile.step_px / 1.5,
                profile,
                rng,
                steps,
            );
            steps.push(Step {
                delay: profile.pause(rng, 0.06, 0.16),
                event: Event::Button {
                    button: *button,
                    press: false,
                },
            });
            Outcome {
                position: Some(destination),
                origin: Some(origin),
            }
        }
        ActionKind::Scroll {
            target,
            direction,
            amount,
        } => {
            let mut position = None;
            if let Some(target) = target {
                let point = resolve(*target, profile, rng);
                track(
                    cursor.unwrap_or(point),
                    point,
                    APPROACH,
                    profile.step_px,
                    profile,
                    rng,
                    steps,
                );
                position = Some(point);
            }
            let sign = direction.sign();
            let mut remaining = *amount;
            let mut first = true;
            while remaining > 0 {
                // A wheel never delivers one huge jump: it ticks.
                let chunk = remaining.min(rng.random_range(1..=3));
                remaining -= chunk;
                let delay = if first {
                    first = false;
                    profile.pause(rng, 0.03, 0.12)
                } else {
                    profile.pause(rng, 0.05, 0.20)
                };
                steps.push(Step {
                    delay,
                    event: Event::Scroll {
                        steps: sign * chunk,
                    },
                });
            }
            Outcome {
                position,
                origin: None,
            }
        }
        ActionKind::Type { text } => {
            plan_text(text, profile, rng, steps);
            Outcome::default()
        }
        ActionKind::Key(key) => {
            tap(*key, profile.pause(rng, 0.02, 0.06), profile, rng, steps);
            Outcome::default()
        }
        ActionKind::Hotkey(keys) => {
            plan_hotkey(keys, profile, rng, steps);
            Outcome::default()
        }
        ActionKind::Sequence(actions) => {
            let mut current = cursor;
            let mut first = true;
            for action in actions {
                if !first {
                    // Let the UI react before the next action: a real user
                    // waits for the focus ring, the menu, or the cursor.
                    steps.push(Step {
                        delay: profile.pause(rng, 0.25, 0.60),
                        event: Event::Pause,
                    });
                }
                first = false;
                let start = steps.len();
                let outcome = plan_into(action, current, profile, rng, steps, children);
                if let Some(point) = outcome.position {
                    current = Some(point);
                }
                let duration = steps[start..].iter().map(|step| step.delay).sum();
                children.push(Child {
                    action: action.label(),
                    duration,
                    position: outcome.position,
                    chars: chars_of(action),
                    key: key_of(action),
                });
            }
            Outcome {
                position: current,
                origin: None,
            }
        }
    }
}

/// How many characters an action types, when it types.
fn chars_of(action: &Action) -> Option<usize> {
    match &action.kind {
        ActionKind::Type { text } => Some(text.chars().count()),
        _ => None,
    }
}

/// Which key an action presses, when it names one.
fn key_of(action: &Action) -> Option<String> {
    match &action.kind {
        ActionKind::Key(key) => Some(crate::request::key_label(*key)),
        ActionKind::Hotkey(keys) => Some(
            keys.iter()
                .map(|key| crate::request::key_label(*key))
                .collect::<Vec<_>>()
                .join("+"),
        ),
        _ => None,
    }
}

/// Samples one pointer path from `from` to `to`.
///
/// The path bows sideways by a small random amount, is sampled at a fixed
/// pixel spacing, and is eased with a smoothstep so the hand accelerates out of
/// the start and decelerates into the landing point. The final sample is exact:
/// a target is never missed by a tremor.
#[allow(clippy::cast_precision_loss)]
fn track<R: Rng>(
    from: Point,
    to: Point,
    style: TrackStyle,
    step_px: f64,
    profile: Profile,
    rng: &mut R,
    steps: &mut Vec<Step>,
) {
    let dx = f64::from(to.x - from.x);
    let dy = f64::from(to.y - from.y);
    let distance = dx.hypot(dy);

    if distance < 0.5 {
        // Already there: a hand settles, it does not travel.
        steps.push(Step {
            delay: profile.pause(rng, 0.03, 0.09),
            event: Event::Move { x: to.x, y: to.y },
        });
        return;
    }

    let speed = rng.random_range(style.speed.0..style.speed.1) * profile.speed_factor();
    let total = (distance / speed).clamp(0.05, MAX_MOVE_SECONDS);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let samples = ((distance / step_px).clamp(4.0, 160.0) as usize).max(2);

    // Perpendicular bow: the path is a quadratic Bézier whose control point
    // sits off the straight line by a small, capped, random amount.
    let (normal_x, normal_y) = (-dy / distance, dx / distance);
    let sign = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
    let bow = (rng.random_range(style.bend.0..style.bend.1) * distance * sign)
        .clamp(-style.bend_cap, style.bend_cap);
    let control = (
        f64::from(from.x) + dx * 0.5 + normal_x * bow,
        f64::from(from.y) + dy * 0.5 + normal_y * bow,
    );
    let (start, end) = (
        (f64::from(from.x), f64::from(from.y)),
        (f64::from(to.x), f64::from(to.y)),
    );

    let tremor = style.jitter * profile.jitter;
    let mut previous = 0.0;
    for sample in 1..=samples {
        let progress = ease(sample as f64 / samples as f64);
        let (mut x, mut y) = quadratic(start, control, end, progress);
        if sample == samples {
            // The last sample is the target itself.
            (x, y) = end;
        } else if tremor > 0.0 {
            x += gaussian(rng) * 0.5 * tremor;
            y += gaussian(rng) * 0.5 * tremor;
        }
        steps.push(Step {
            delay: seconds((total * (progress - previous)).max(0.003)),
            event: Event::Move {
                x: round(x),
                y: round(y),
            },
        });
        previous = progress;
    }
}

/// Picks the pixel an action lands on.
fn resolve<R: Rng>(target: Target, profile: Profile, rng: &mut R) -> Point {
    let Some((x, y, width, height)) = target.area() else {
        // An exact point is an exact point: the model already computed the
        // pixel, so the runtime does not second-guess it. The human variation
        // belongs to rectangle targets, where the caller said "this control".
        return Point {
            x: target.x,
            y: target.y,
        };
    };

    // Never aim at the frame: shrink the rectangle into a safe zone.
    let margin = (f64::from(width.min(height)) * 0.18).clamp(1.0, 12.0);
    let (left, top) = (f64::from(x) + margin, f64::from(y) + margin);
    let (right, bottom) = (
        f64::from(x + width) - margin,
        f64::from(y + height) - margin,
    );
    if right <= left || bottom <= top {
        // Too small for a safe zone: the centre is the only sane choice.
        return Point {
            x: x + width / 2,
            y: y + height / 2,
        };
    }

    // Average of two draws: uniform sampling would drift towards the edges,
    // while this leans mildly towards the middle of the control. `spread = 0`
    // (precise) collapses to the exact centre.
    let fx = (1.0 - profile.spread) * 0.5 + profile.spread * mean_of_two(rng);
    let fy = (1.0 - profile.spread) * 0.5 + profile.spread * mean_of_two(rng);
    Point {
        x: round(left + fx * (right - left)),
        y: round(top + fy * (bottom - top)),
    }
}

/// Presses and releases one key, wrapping shifted characters in a real Shift.
fn tap<R: Rng>(key: Key, delay: Duration, profile: Profile, rng: &mut R, steps: &mut Vec<Step>) {
    let hold = profile.pause(rng, 0.025, 0.06);
    if let Key::Unicode(character) = key {
        let (base, shift) = keystrokes(character);
        if shift {
            steps.push(Step {
                delay,
                event: Event::Key {
                    key: Key::Shift,
                    press: true,
                },
            });
            steps.push(Step {
                delay: profile.pause(rng, 0.012, 0.030),
                event: Event::Key {
                    key: Key::Unicode(base),
                    press: true,
                },
            });
            steps.push(Step {
                delay: hold,
                event: Event::Key {
                    key: Key::Unicode(base),
                    press: false,
                },
            });
            steps.push(Step {
                delay: profile.pause(rng, 0.010, 0.022),
                event: Event::Key {
                    key: Key::Shift,
                    press: false,
                },
            });
            return;
        }
        steps.push(Step {
            delay,
            event: Event::Key {
                key: Key::Unicode(base),
                press: true,
            },
        });
        steps.push(Step {
            delay: hold,
            event: Event::Key {
                key: Key::Unicode(base),
                press: false,
            },
        });
        return;
    }
    steps.push(Step {
        delay,
        event: Event::Key { key, press: true },
    });
    steps.push(Step {
        delay: hold,
        event: Event::Key { key, press: false },
    });
}

/// Types one text payload, run by run.
///
/// Characters the keyboard can express are typed as real key strokes; a run the
/// layout cannot express (CJK, emoji, accented letters on a US layout) is
/// injected as short text events instead. Mixing both keeps mixed ASCII / CJK text close
/// to what a human at a bilingual keyboard would do.
fn plan_text<R: Rng>(text: &str, profile: Profile, rng: &mut R, steps: &mut Vec<Step>) {
    let mut run = String::new();
    let mut run_is_typeable = false;
    for character in text.chars() {
        let character_is_typeable = typeable(character);
        if !run.is_empty() && character_is_typeable != run_is_typeable {
            flush_run(&run, run_is_typeable, profile, rng, steps);
            run.clear();
        }
        if run.is_empty() {
            run_is_typeable = character_is_typeable;
        }
        run.push(character);
    }
    if !run.is_empty() {
        flush_run(&run, run_is_typeable, profile, rng, steps);
    }
}

fn flush_run<R: Rng>(
    run: &str,
    run_is_typeable: bool,
    profile: Profile,
    rng: &mut R,
    steps: &mut Vec<Step>,
) {
    if run_is_typeable {
        for character in run.chars() {
            plan_character(character, profile, rng, steps);
        }
    } else {
        plan_text_chunks(run, profile, rng, steps);
    }
}

/// Types one character with the pause a human would leave after it.
fn plan_character<R: Rng>(character: char, profile: Profile, rng: &mut R, steps: &mut Vec<Step>) {
    if let Some(key) = control_key(character) {
        tap(key, typing_gap(profile, rng), profile, rng, steps);
        steps.push(Step {
            delay: profile.pause(rng, 0.10, 0.30),
            event: Event::Pause,
        });
        return;
    }
    tap(
        Key::Unicode(character),
        typing_gap(profile, rng),
        profile,
        rng,
        steps,
    );
    if let Some((low, high)) = word_pause(character) {
        steps.push(Step {
            delay: profile.pause(rng, low, high),
            event: Event::Pause,
        });
    }
}

/// Injects text the keyboard layout cannot express as short text events.
fn plan_text_chunks<R: Rng>(text: &str, profile: Profile, rng: &mut R, steps: &mut Vec<Step>) {
    let mut chunk = String::new();
    let mut delay = profile.pause(rng, 0.05, 0.15);
    for character in text.chars() {
        chunk.push(character);
        if !character.is_whitespace() && chunk.chars().count() < TEXT_CHUNK_CHARS {
            continue;
        }
        steps.push(Step {
            delay,
            event: Event::Text {
                text: std::mem::take(&mut chunk),
            },
        });
        delay = profile.pause(rng, 0.04, 0.14);
        if matches!(character, '.' | '!' | '?' | '\n') {
            steps.push(Step {
                delay: profile.pause(rng, 0.20, 0.50),
                event: Event::Pause,
            });
        }
    }
    if !chunk.is_empty() {
        steps.push(Step {
            delay,
            event: Event::Text { text: chunk },
        });
    }
}

/// Presses the modifiers, taps the key, releases the modifiers.
#[allow(clippy::cast_precision_loss)]
fn plan_hotkey<R: Rng>(keys: &[Key], profile: Profile, rng: &mut R, steps: &mut Vec<Step>) {
    let (modifiers, taps): (Vec<Key>, Vec<Key>) = keys.iter().partition(|key| is_modifier(**key));
    for (index, modifier) in modifiers.iter().enumerate() {
        let delay = if index == 0 {
            profile.pause(rng, 0.01, 0.04)
        } else {
            profile.pause(rng, 0.03, 0.08)
        };
        steps.push(Step {
            delay,
            event: Event::Key {
                key: *modifier,
                press: true,
            },
        });
    }
    for tap in &taps {
        steps.push(Step {
            delay: profile.pause(rng, 0.03, 0.08),
            event: Event::Key {
                key: *tap,
                press: true,
            },
        });
        // Every extra modifier keeps the combination held a little longer.
        let base = profile.pause(rng, 0.045, 0.095);
        let extra = seconds(0.012 * modifiers.len() as f64 / profile.speed_factor());
        steps.push(Step {
            delay: base + extra,
            event: Event::Key {
                key: *tap,
                press: false,
            },
        });
    }
    for modifier in modifiers.iter().rev() {
        steps.push(Step {
            delay: profile.pause(rng, 0.02, 0.06),
            event: Event::Key {
                key: *modifier,
                press: false,
            },
        });
    }
}

/// Upper bound of what an action costs, used to refuse calls that would block
/// the tool loop for too long. Deliberately generous: it only has to be right
/// about the order of magnitude.
#[must_use]
pub fn nominal_duration(action: &Action) -> Duration {
    #[allow(clippy::cast_precision_loss)]
    fn seconds_of(action: &Action) -> f64 {
        match &action.kind {
            ActionKind::Move { .. } | ActionKind::Click { .. } => 1.2,
            ActionKind::Drag { .. } => 2.4,
            ActionKind::Scroll { target, amount, .. } => {
                f64::from(*amount) * 0.12 + if target.is_some() { 1.2 } else { 0.0 }
            }
            ActionKind::Type { text } => text.chars().count() as f64 * 0.090,
            ActionKind::Key(_) => 0.25,
            ActionKind::Hotkey(keys) => 0.45 + keys.len() as f64 * 0.04,
            ActionKind::Sequence(actions) => {
                actions.iter().map(|action| seconds_of(action) + 0.4).sum()
            }
        }
    }
    seconds(seconds_of(action) / Profile::of(action.behavior).speed_factor())
}

/// Smoothstep ease: quick out of the start, quickest in the middle, slow into
/// the target. Values outside `[0, 1]` are clamped.
fn ease(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn quadratic(start: (f64, f64), control: (f64, f64), end: (f64, f64), t: f64) -> (f64, f64) {
    let inverse = 1.0 - t;
    (
        inverse * inverse * start.0 + 2.0 * inverse * t * control.0 + t * t * end.0,
        inverse * inverse * start.1 + 2.0 * inverse * t * control.1 + t * t * end.1,
    )
}

/// Roughly standard-normal noise without pulling in `rand_distr`: the sum of
/// four uniforms is bell-shaped enough for a hand tremor.
fn gaussian<R: Rng>(rng: &mut R) -> f64 {
    let sum: f64 = (0..4).map(|_| rng.random_range(0.0..1.0)).sum();
    (sum - 2.0) * 1.732_050_8
}

fn mean_of_two<R: Rng>(rng: &mut R) -> f64 {
    let first: f64 = rng.random_range(0.0..1.0);
    let second: f64 = rng.random_range(0.0..1.0);
    first.midpoint(second)
}

/// Builds a duration, ignoring nonsense values instead of panicking.
fn seconds(value: f64) -> Duration {
    if value.is_finite() && value > 0.0 {
        Duration::from_secs_f64(value.min(MAX_PAUSE_SECONDS))
    } else {
        Duration::ZERO
    }
}

#[allow(clippy::cast_possible_truncation)]
fn round(value: f64) -> i32 {
    value
        .round()
        .clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32
}

/// Whether the keyboard can express this character as a key stroke.
fn typeable(character: char) -> bool {
    character.is_ascii_graphic() || matches!(character, ' ' | '\n' | '\r' | '\t')
}

/// Characters with a dedicated key, mapped to a name the planner can tap.
fn control_key(character: char) -> Option<Key> {
    match character {
        '\n' | '\r' => Some(Key::Return),
        '\t' => Some(Key::Tab),
        _ => None,
    }
}

/// Shifted characters and the base key they sit on, on a US layout.
const SHIFTED: [(char, char); 21] = [
    ('!', '1'),
    ('@', '2'),
    ('#', '3'),
    ('$', '4'),
    ('%', '5'),
    ('^', '6'),
    ('&', '7'),
    ('*', '8'),
    ('(', '9'),
    (')', '0'),
    ('_', '-'),
    ('+', '='),
    ('{', '['),
    ('}', ']'),
    ('|', '\\'),
    (':', ';'),
    ('"', '\''),
    ('<', ','),
    ('>', '.'),
    ('?', '/'),
    ('~', '`'),
];

/// The physical stroke behind one printable character on a US layout.
fn keystrokes(character: char) -> (char, bool) {
    if character.is_ascii_uppercase() {
        return (character.to_ascii_lowercase(), true);
    }
    for (shifted, base) in SHIFTED {
        if character == shifted {
            return (base, true);
        }
    }
    (character, false)
}

/// The pause a human leaves after this character, if any.
fn word_pause(character: char) -> Option<(f64, f64)> {
    match character {
        ' ' => Some((0.05, 0.16)),
        ',' | ';' | ':' => Some((0.12, 0.34)),
        '.' | '!' | '?' => Some((0.22, 0.60)),
        _ => None,
    }
}

/// One typing gap: 150 WPM ± 35 %, scaled by the profile.
fn typing_gap<R: Rng>(profile: Profile, rng: &mut R) -> Duration {
    seconds(BASE_CHAR_SECONDS * rng.random_range(0.62..1.55) / profile.speed_factor())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::{ActionKind, Behavior, MouseButton, ScrollDirection, Target, parse_key};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    fn point(x: i32, y: i32) -> Target {
        Target {
            x,
            y,
            width: None,
            height: None,
        }
    }

    fn rect(x: i32, y: i32, width: i32, height: i32) -> Target {
        Target {
            x,
            y,
            width: Some(width),
            height: Some(height),
        }
    }

    fn action(kind: ActionKind) -> Action {
        Action {
            behavior: Behavior::Natural,
            kind,
        }
    }

    fn moves(plan: &Plan) -> Vec<(i32, i32)> {
        plan.steps
            .iter()
            .filter_map(|step| match step.event {
                Event::Move { x, y } => Some((x, y)),
                _ => None,
            })
            .collect()
    }

    fn keys(plan: &Plan) -> Vec<(Key, bool)> {
        plan.steps
            .iter()
            .filter_map(|step| match step.event {
                Event::Key { key, press } => Some((key, press)),
                _ => None,
            })
            .collect()
    }

    fn buttons(plan: &Plan) -> Vec<(MouseButton, bool)> {
        plan.steps
            .iter()
            .filter_map(|step| match step.event {
                Event::Button { button, press } => Some((button, press)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_move_is_an_arc_that_ends_exactly_on_target() {
        let mut rng = rng(7);
        let plan = plan(
            &action(ActionKind::Move {
                target: point(1240, 680),
            }),
            Some(Point { x: 20, y: 40 }),
            &mut rng,
        );
        let samples = moves(&plan);
        assert!(
            samples.len() >= 8,
            "expected a sampled path, got {samples:?}"
        );
        assert_eq!(*samples.last().unwrap(), (1240, 680));
        // The path leaves the cursor immediately, but only a little.
        let (first_x, first_y) = samples[0];
        assert!(
            (20..120).contains(&first_x),
            "jumped at the start: {first_x}"
        );
        assert!(
            (40..140).contains(&first_y),
            "jumped at the start: {first_y}"
        );
        // The path bows away from the straight line, without a wild detour.
        let deviation = samples
            .iter()
            .map(|(x, y)| {
                // Distance from the line from (20,40) to (1240,680).
                (f64::from(1220 * (40 - y) - 640 * (20 - x)) / 1377.0).abs()
            })
            .fold(0.0, f64::max);
        assert!(deviation > 0.5, "the path is straight: {deviation}");
        assert!(deviation < 160.0, "the path detours too far: {deviation}");
        assert!(plan.duration() > Duration::ZERO);
    }

    #[test]
    fn a_subpixel_move_just_settles() {
        let mut rng = rng(3);
        let plan = plan(
            &action(ActionKind::Move {
                target: point(100, 100),
            }),
            Some(Point { x: 100, y: 100 }),
            &mut rng,
        );
        assert_eq!(moves(&plan), vec![(100, 100)]);
    }

    #[test]
    fn pointer_plans_are_deterministic_for_one_seed_and_varied_across_seeds() {
        let request = action(ActionKind::Move {
            target: point(900, 500),
        });
        let first = plan(&request, Some(Point { x: 0, y: 0 }), &mut rng(11));
        let again = plan(&request, Some(Point { x: 0, y: 0 }), &mut rng(11));
        assert_eq!(first.steps, again.steps);
        let other = plan(&request, Some(Point { x: 0, y: 0 }), &mut rng(12));
        assert_ne!(first.steps, other.steps);
    }

    #[test]
    fn behaviour_trades_speed_for_steadiness() {
        let request = |behavior| Action {
            behavior,
            kind: ActionKind::Move {
                target: point(1400, 900),
            },
        };
        let cursor = Some(Point { x: 0, y: 0 });
        let fast = plan(&request(Behavior::Fast), cursor, &mut rng(5));
        let natural = plan(&request(Behavior::Natural), cursor, &mut rng(5));
        let precise = plan(&request(Behavior::Precise), cursor, &mut rng(5));
        assert!(fast.duration() < natural.duration());
        assert!(natural.duration() < precise.duration());
    }

    #[test]
    fn a_rectangle_target_lands_inside_a_safe_zone() {
        let request = |behavior| Action {
            behavior,
            kind: ActionKind::Click {
                target: rect(820, 640, 120, 36),
                button: MouseButton::Left,
                count: 1,
            },
        };
        for seed in 0..40 {
            let plan = plan(&request(Behavior::Natural), None, &mut rng(seed));
            let (x, y) = *moves(&plan).last().unwrap();
            // 120x36 shrunk by an 6.5 px margin: x in [827, 933], y in [647, 669].
            assert!((827..=933).contains(&x), "x={x} escaped the safe zone");
            assert!((647..=669).contains(&y), "y={y} escaped the safe zone");
        }
        // Precise aims at the exact centre and never drifts.
        let plan = plan(&request(Behavior::Precise), None, &mut rng(9));
        assert_eq!(*moves(&plan).last().unwrap(), (880, 658));
    }

    #[test]
    fn a_click_settles_presses_holds_and_releases() {
        let mut rng = rng(21);
        let plan = plan(
            &action(ActionKind::Click {
                target: point(600, 400),
                button: MouseButton::Right,
                count: 1,
            }),
            Some(Point { x: 100, y: 100 }),
            &mut rng,
        );
        assert_eq!(
            buttons(&plan),
            vec![(MouseButton::Right, true), (MouseButton::Right, false)]
        );
        let press = plan
            .steps
            .iter()
            .position(|step| {
                step.event
                    == Event::Button {
                        button: MouseButton::Right,
                        press: true,
                    }
            })
            .unwrap();
        // Everything after the settle is a press, not a jump: the path is there.
        assert!(press > 0);
        let hold = plan.steps[press + 1].delay;
        assert!(
            hold >= Duration::from_millis(30) && hold <= Duration::from_millis(120),
            "{hold:?}"
        );
    }

    #[test]
    fn a_double_click_is_two_bursts_on_the_same_pixel() {
        let mut rng = rng(4);
        let plan = plan(
            &action(ActionKind::Click {
                target: point(600, 400),
                button: MouseButton::Left,
                count: 2,
            }),
            None,
            &mut rng,
        );
        assert_eq!(
            buttons(&plan),
            vec![
                (MouseButton::Left, true),
                (MouseButton::Left, false),
                (MouseButton::Left, true),
                (MouseButton::Left, false),
            ]
        );
        let gap = plan.steps[3].delay;
        assert!(
            gap >= Duration::from_millis(40),
            "the burst is too fast: {gap:?}"
        );
        assert!(
            gap <= Duration::from_millis(200),
            "outside the double-click threshold: {gap:?}"
        );
    }

    #[test]
    fn a_drag_presses_before_gliding_and_releases_after() {
        let mut rng = rng(6);
        let plan = plan(
            &action(ActionKind::Drag {
                from: point(400, 300),
                to: point(900, 500),
                button: MouseButton::Left,
            }),
            Some(Point { x: 10, y: 10 }),
            &mut rng,
        );
        let events = &plan.steps;
        let press = events
            .iter()
            .position(|step| {
                step.event
                    == Event::Button {
                        button: MouseButton::Left,
                        press: true,
                    }
            })
            .unwrap();
        let release = events
            .iter()
            .position(|step| {
                step.event
                    == Event::Button {
                        button: MouseButton::Left,
                        press: false,
                    }
            })
            .unwrap();
        let hold_moves: Vec<(i32, i32)> = events[press..release]
            .iter()
            .filter_map(|step| match step.event {
                Event::Move { x, y } => Some((x, y)),
                _ => None,
            })
            .collect();
        assert!(
            !hold_moves.is_empty(),
            "nothing moved while the button was down"
        );
        assert_eq!(*hold_moves.last().unwrap(), (900, 500));
        assert!(
            hold_moves.first().unwrap().0 >= 380,
            "the glide started elsewhere"
        );
        assert_eq!(plan.origin, Some(Point { x: 400, y: 300 }));
        assert_eq!(plan.position, Some(Point { x: 900, y: 500 }));
    }

    #[test]
    fn a_drag_glide_is_slower_than_a_free_move_of_the_same_distance() {
        let distance = |plan: &Plan| plan.duration();
        let mut first = rng(8);
        let drag = action(ActionKind::Drag {
            from: point(400, 300),
            to: point(900, 300),
            button: MouseButton::Left,
        });
        let move_action = action(ActionKind::Move {
            target: point(900, 300),
        });
        let drag_plan = plan(&drag, Some(Point { x: 400, y: 300 }), &mut first);
        let mut second = rng(8);
        let move_plan = plan(&move_action, Some(Point { x: 400, y: 300 }), &mut second);
        assert!(
            distance(&drag_plan) > distance(&move_plan),
            "a drag must be steadier than a free move"
        );
    }

    #[test]
    fn a_scroll_is_split_into_ticks_that_add_up() {
        let mut rng = rng(13);
        let down = plan(
            &action(ActionKind::Scroll {
                target: None,
                direction: ScrollDirection::Down,
                amount: 5,
            }),
            None,
            &mut rng,
        );
        let ticks: Vec<i32> = down
            .steps
            .iter()
            .filter_map(|step| match step.event {
                Event::Scroll { steps } => Some(steps),
                _ => None,
            })
            .collect();
        assert!(ticks.len() >= 2, "one huge jump is not a wheel: {ticks:?}");
        assert_eq!(ticks.iter().sum::<i32>(), 5);
        assert!(ticks.iter().all(|tick| (1..=3).contains(tick)));

        let up = plan(
            &action(ActionKind::Scroll {
                target: None,
                direction: ScrollDirection::Up,
                amount: 4,
            }),
            None,
            &mut rng,
        );
        assert_eq!(
            up.steps
                .iter()
                .filter_map(|step| match step.event {
                    Event::Scroll { steps } => Some(steps),
                    _ => None,
                })
                .sum::<i32>(),
            -4
        );
    }

    #[test]
    fn typing_presses_shift_for_uppercase_and_punctuation() {
        let mut rng = rng(15);
        let plan = plan(
            &action(ActionKind::Type {
                text: "Hi!".to_owned(),
            }),
            None,
            &mut rng,
        );
        assert_eq!(
            keys(&plan),
            vec![
                // H is Shift + h.
                (Key::Shift, true),
                (Key::Unicode('h'), true),
                (Key::Unicode('h'), false),
                (Key::Shift, false),
                (Key::Unicode('i'), true),
                (Key::Unicode('i'), false),
                // ! is Shift + 1.
                (Key::Shift, true),
                (Key::Unicode('1'), true),
                (Key::Unicode('1'), false),
                (Key::Shift, false),
            ]
        );
        // The Shift key is never left down.
        assert!(plan.events() == keys(&plan).len());
    }

    #[test]
    fn typing_uses_control_keys_and_leaves_pauses_after_words() {
        let mut rng = rng(16);
        let plan = plan(
            &action(ActionKind::Type {
                text: "a b.\n".to_owned(),
            }),
            None,
            &mut rng,
        );
        let strokes = keys(&plan);
        assert!(strokes.contains(&(Key::Return, true)));
        assert!(strokes.contains(&(Key::Return, false)));
        assert!(plan.steps.iter().any(|step| step.event == Event::Pause));
        assert_eq!(plan.events(), strokes.len());
        assert!(plan.duration() > Duration::ZERO);
    }

    #[test]
    fn non_ascii_text_goes_through_text_events_while_ascii_stays_on_the_keyboard() {
        let mut rng = rng(17);
        let plan = plan(
            &action(ActionKind::Type {
                text: "\u{4f60}\u{597d} world".to_owned(),
            }),
            None,
            &mut rng,
        );
        let texts: Vec<String> = plan
            .steps
            .iter()
            .filter_map(|step| match &step.event {
                Event::Text { text } => Some(text.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(texts.concat(), "\u{4f60}\u{597d}");
        // The space and "world" are typed as ordinary keys.
        let strokes = keys(&plan);
        let typed: String = strokes
            .iter()
            .filter(|(_, press)| *press)
            .filter_map(|(key, _)| match key {
                Key::Unicode(character) => Some(*character),
                _ => None,
            })
            .collect();
        assert_eq!(typed, " world");
    }

    #[test]
    fn one_key_is_a_real_press_with_a_hold() {
        let mut rng = rng(18);
        let plan = plan(
            &action(ActionKind::Key(parse_key("Enter").unwrap())),
            None,
            &mut rng,
        );
        assert_eq!(keys(&plan), vec![(Key::Return, true), (Key::Return, false)]);
    }

    #[test]
    fn a_hotkey_presses_modifiers_first_and_releases_them_last() {
        let mut rng = rng(19);
        let keys = vec![
            parse_key("CTRL").unwrap(),
            parse_key("SHIFT").unwrap(),
            parse_key("L").unwrap(),
        ];
        let plan = plan(&action(ActionKind::Hotkey(keys)), None, &mut rng);
        assert_eq!(
            keys_of(&plan),
            vec![
                (Key::Control, true),
                (Key::Shift, true),
                (Key::Unicode('L'), true),
                (Key::Unicode('L'), false),
                (Key::Shift, false),
                (Key::Control, false),
            ]
        );
    }

    fn keys_of(plan: &Plan) -> Vec<(Key, bool)> {
        keys(plan)
    }

    #[test]
    fn a_sequence_keeps_the_pointer_state_and_waits_between_actions() {
        let mut rng = rng(20);
        let sequence = action(ActionKind::Sequence(vec![
            action(ActionKind::Click {
                target: rect(800, 600, 200, 40),
                button: MouseButton::Left,
                count: 1,
            }),
            action(ActionKind::Type {
                text: "hello@example.com".to_owned(),
            }),
            action(ActionKind::Key(parse_key("Enter").unwrap())),
        ]));
        let plan = plan(&sequence, Some(Point { x: 50, y: 50 }), &mut rng);
        assert_eq!(plan.children.len(), 3);
        assert_eq!(plan.children[0].action, "click");
        assert_eq!(plan.children[1].chars, Some(17));
        assert_eq!(plan.children[2].key.as_deref(), Some("enter"));
        // The typed text never moves the pointer.
        let landing = plan.children[0].position.unwrap();
        assert_eq!(plan.position, Some(landing));
        assert!(plan.children[1].position.is_none());
        // The first action approaches from the real cursor.
        let (start_x, start_y) = moves(&plan)[0];
        assert!(
            start_x < 200 && start_y < 200,
            "the approach did not start at the cursor"
        );
        // Gaps exist between the three actions.
        let pauses = plan
            .steps
            .iter()
            .filter(|step| step.event == Event::Pause)
            .count();
        assert!(pauses >= 2, "expected settle gaps, got {pauses}");
    }

    #[test]
    fn nominal_durations_scale_with_the_profile_and_the_payload() {
        let typed = |behavior, count: usize| Action {
            behavior,
            kind: ActionKind::Type {
                text: "x".repeat(count),
            },
        };
        assert!(
            nominal_duration(&typed(Behavior::Fast, 100))
                < nominal_duration(&typed(Behavior::Natural, 100))
        );
        assert!(
            nominal_duration(&typed(Behavior::Natural, 100))
                < nominal_duration(&typed(Behavior::Natural, 200))
        );
        assert_eq!(
            nominal_duration(&typed(Behavior::Natural, 100)).as_secs(),
            9
        );
        assert!(
            nominal_duration(&action(ActionKind::Move {
                target: point(0, 0)
            })) > Duration::ZERO
        );
    }

    #[test]
    fn clicks_and_keys_are_counted_but_pauses_are_not_events() {
        let mut rng = rng(22);
        let plan = plan(
            &action(ActionKind::Click {
                target: point(10, 10),
                button: MouseButton::Left,
                count: 2,
            }),
            Some(Point { x: 10, y: 10 }),
            &mut rng,
        );
        assert!(plan.events() >= buttons(&plan).len());
        assert_eq!(buttons(&plan).len(), 4);
    }
}
