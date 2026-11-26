//! Menu bar / system tray entry and animation driver.
//!
//! The icon itself is drawn frame by frame per status by `relay-platform`; this only does two things:
//! 1. translate runtime signals into [`TrayStatus`] (highest priority wins with several tasks);
//! 2. advance frames on the GPUI main thread as needed (macOS only updates trays there).
//!
//! Understanding happens before the task exists and the runtime emits no task events then,
//! so understanding events are subscribed separately to keep the tray from sitting in Ready.

use std::collections::BTreeMap;
use std::time::Duration;

use gpui_kit::*;
use relay_core::{RuntimeEventKind, TaskId, TaskState};
use relay_platform::{Tray, TrayStatus};
use relay_runtime::UnderstandStage;
use tokio::sync::broadcast::error::RecvError;
use tokio::sync::mpsc::UnboundedSender;

pub use relay_platform::TrayAction;

/// Animation heartbeat; frames advance by each status own duration, this is only the poll interval.
const TICK: Duration = Duration::from_millis(30);

/// The tray object is not Send; it lives in a main-thread Entity.
struct TraySlot {
    tray: Tray,
}

impl TraySlot {
    /// Apply one queued update on the main thread.
    fn apply(&mut self, update: Update) {
        match update {
            Update::Status(status) => {
                // A new task takes over the menu bar: drop the previous task's conclusion.
                if !matches!(status, TrayStatus::Idle | TrayStatus::Success) {
                    self.tray.set_headline(None);
                }
                self.tray.set_status(status);
            }
            Update::Headline(reply) => self.tray.set_headline(Some(reply)),
        }
    }
}

/// Work queued from the runtime signals to the main thread.
enum Update {
    Status(TrayStatus),
    /// Final reply of a finished task.
    Headline(String),
}

/// Install the tray and start the animation driver; on failure the caller opens the panel instead.
pub fn install(
    cx: &mut App,
    runtime: &relay_runtime::Runtime,
    sender: UnboundedSender<TrayAction>,
) -> Result<(), String> {
    let tray = relay_platform::install_tray(sender).map_err(|err| err.to_string())?;
    let slot = cx.new(|_| TraySlot { tray });

    // Runtime signals (background thread) -> update channel -> main thread icon and title refresh.
    let (update_tx, mut update_rx) = tokio::sync::mpsc::unbounded_channel::<Update>();
    let mut tasks = runtime.subscribe();
    let mut understanding = runtime.subscribe_understanding();
    cx.background_executor()
        .spawn(async move {
            let mut board = StatusBoard::default();
            loop {
                let signal = tokio::select! {
                    event = tasks.recv() => match event {
                        Ok(event) => match event.kind {
                            RuntimeEventKind::StateChanged { to, .. } => {
                                Signal::Task { id: event.task_id, state: to }
                            }
                            RuntimeEventKind::Completed { reply, .. } => Signal::Headline(reply),
                            _ => continue,
                        },
                        Err(RecvError::Lagged(_)) => continue,
                        Err(RecvError::Closed) => break,
                    },
                    event = understanding.recv() => match event {
                        Ok(event) => Signal::Understanding(event.stage),
                        Err(RecvError::Lagged(_)) => continue,
                        Err(RecvError::Closed) => break,
                    },
                };
                let update = match signal {
                    Signal::Headline(reply) => Update::Headline(reply),
                    task => Update::Status(board.apply(task)),
                };
                if update_tx.send(update).is_err() {
                    break;
                }
            }
        })
        .detach();

    cx.spawn(async move |cx| {
        loop {
            // Several updates may queue up; apply them in order.
            while let Ok(update) = update_rx.try_recv() {
                slot.update(cx, |slot, _| slot.apply(update));
            }
            let needs_tick = slot.update(cx, |slot, _| {
                slot.tray.tick();
                slot.tray.needs_tick()
            });
            if needs_tick {
                cx.background_executor().timer(TICK).await;
            } else {
                // Static state: do not spin, wait for the next signal.
                match update_rx.recv().await {
                    Some(update) => slot.update(cx, |slot, _| slot.apply(update)),
                    None => return,
                }
            }
        }
    })
    .detach();

    Ok(())
}

/// Where tray status comes from.
enum Signal {
    /// Understanding progressed (the task does not exist yet).
    Understanding(UnderstandStage),
    /// Task state changed.
    Task { id: TaskId, state: TaskState },
    /// Task finished with its final reply.
    Headline(String),
}

/// Status aggregation across parallel tasks: confirmation > paused > working > thinking.
///
/// When one task ends while others still run, the tray must not jump to Completed,
/// otherwise users think everything is finished.
#[derive(Default)]
struct StatusBoard {
    tasks: BTreeMap<TaskId, TaskState>,
    /// Understanding phase (especially the part before the task exists).
    understanding: bool,
}

impl StatusBoard {
    /// Record one signal and return the animation the tray should show now.
    fn apply(&mut self, signal: Signal) -> TrayStatus {
        match signal {
            // Headlines bypass the board and never change the icon.
            Signal::Headline(_) => self.aggregate(),
            Signal::Understanding(stage) => {
                if stage == UnderstandStage::Failed {
                    self.understanding = false;
                    if self.tasks.is_empty() {
                        return TrayStatus::Error;
                    }
                } else if self.tasks.is_empty() {
                    // Once the task exists, late understanding events must not pull the animation back to Thinking.
                    self.understanding = true;
                }
                self.aggregate()
            }
            Signal::Task { id, state } => {
                self.understanding = false;
                if state.is_terminal() {
                    self.tasks.remove(&id);
                    if self.tasks.is_empty() {
                        // One-shot animation; the tray returns to Ready on its own when it ends.
                        return status_for(state);
                    }
                    return self.aggregate();
                }
                self.tasks.insert(id, state);
                self.aggregate()
            }
        }
    }

    fn aggregate(&self) -> TrayStatus {
        let mut best = if self.understanding {
            TrayStatus::Thinking
        } else {
            TrayStatus::Idle
        };
        for state in self.tasks.values() {
            let status = status_for(*state);
            if rank(status) > rank(best) {
                best = status;
            }
        }
        best
    }
}

fn rank(status: TrayStatus) -> u8 {
    match status {
        TrayStatus::Attention => 4,
        TrayStatus::Paused => 3,
        TrayStatus::Working => 2,
        TrayStatus::Thinking => 1,
        _ => 0,
    }
}

/// Task state -> tray animation.
fn status_for(state: TaskState) -> TrayStatus {
    match state {
        TaskState::Pending | TaskState::Understanding => TrayStatus::Thinking,
        TaskState::AwaitingConfirmation => TrayStatus::Attention,
        TaskState::Executing | TaskState::Verifying | TaskState::Delivering => TrayStatus::Working,
        TaskState::Paused => TrayStatus::Paused,
        TaskState::Succeeded => TrayStatus::Success,
        TaskState::Failed => TrayStatus::Error,
        TaskState::Cancelled => TrayStatus::Idle,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // gpui_kit re-exports its own `test` macro via glob; use the built-in attribute explicitly.
    #[::core::prelude::v1::test]
    fn task_states_map_to_distinct_animations() {
        assert_eq!(status_for(TaskState::Pending), TrayStatus::Thinking);
        assert_eq!(status_for(TaskState::Understanding), TrayStatus::Thinking);
        assert_eq!(
            status_for(TaskState::AwaitingConfirmation),
            TrayStatus::Attention
        );
        assert_eq!(status_for(TaskState::Executing), TrayStatus::Working);
        assert_eq!(status_for(TaskState::Verifying), TrayStatus::Working);
        assert_eq!(status_for(TaskState::Delivering), TrayStatus::Working);
        assert_eq!(status_for(TaskState::Paused), TrayStatus::Paused);
        assert_eq!(status_for(TaskState::Succeeded), TrayStatus::Success);
        assert_eq!(status_for(TaskState::Failed), TrayStatus::Error);
        assert_eq!(status_for(TaskState::Cancelled), TrayStatus::Idle);
    }

    #[::core::prelude::v1::test]
    fn understanding_shows_thinking_before_task_exists() {
        let mut board = StatusBoard::default();
        assert_eq!(board.aggregate(), TrayStatus::Idle);
        assert_eq!(
            board.apply(Signal::Understanding(UnderstandStage::Started)),
            TrayStatus::Thinking
        );
        assert_eq!(
            board.apply(Signal::Understanding(UnderstandStage::ToolCall)),
            TrayStatus::Thinking
        );

        // After the task exists, late understanding events must not pull the animation back to Thinking.
        let task = TaskId::new();
        assert_eq!(
            board.apply(Signal::Task {
                id: task,
                state: TaskState::AwaitingConfirmation,
            }),
            TrayStatus::Attention
        );
        assert_eq!(
            board.apply(Signal::Understanding(UnderstandStage::Decided)),
            TrayStatus::Attention
        );

        // Understanding failed (no task) shows a one-shot error.
        let mut board = StatusBoard::default();
        board.apply(Signal::Understanding(UnderstandStage::Started));
        assert_eq!(
            board.apply(Signal::Understanding(UnderstandStage::Failed)),
            TrayStatus::Error
        );
    }

    #[::core::prelude::v1::test]
    fn parallel_tasks_keep_the_busiest_animation() {
        let mut board = StatusBoard::default();
        let (a, b) = (TaskId::new(), TaskId::new());
        assert_eq!(
            board.apply(Signal::Task {
                id: a,
                state: TaskState::Understanding,
            }),
            TrayStatus::Thinking
        );
        assert_eq!(
            board.apply(Signal::Task {
                id: b,
                state: TaskState::Executing,
            }),
            TrayStatus::Working
        );
        // One task done while another runs: must not jump to Completed.
        assert_eq!(
            board.apply(Signal::Task {
                id: a,
                state: TaskState::Succeeded,
            }),
            TrayStatus::Working
        );
        assert_eq!(
            board.apply(Signal::Task {
                id: b,
                state: TaskState::Paused,
            }),
            TrayStatus::Paused
        );
        // Play the one-shot hint only when everything finished.
        assert_eq!(
            board.apply(Signal::Task {
                id: b,
                state: TaskState::Failed,
            }),
            TrayStatus::Error
        );
    }
}
