use std::fmt;

use serde::{Deserialize, Serialize};

use crate::error::{CoreError, Result};

/// Task state machine. Transitions are only allowed as shown below;
/// the runtime is the only writer.
///
/// ```text
/// Pending -> Understanding -> AwaitingConfirmation -> Executing -> Verifying -> Delivering -> Succeeded
///                    ^                   |                |  ^                      |
///                    |                   |                |  |                      v
///                    +---(edit)-- redo ---+                |  +--(approve)---+          Failed
///                                                          v              |
///                                                        Paused ------- Executing
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskState {
    Pending,
    Understanding,
    AwaitingConfirmation,
    Executing,
    Paused,
    Verifying,
    Delivering,
    Succeeded,
    Failed,
    Cancelled,
}

impl TaskState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Understanding => "understanding",
            Self::AwaitingConfirmation => "awaiting_confirmation",
            Self::Executing => "executing",
            Self::Paused => "paused",
            Self::Verifying => "verifying",
            Self::Delivering => "delivering",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed | Self::Cancelled)
    }

    pub fn is_active(self) -> bool {
        !self.is_terminal() && self != Self::AwaitingConfirmation && self != Self::Paused
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Pending => "Queued",
            Self::Understanding => "Understanding",
            Self::AwaitingConfirmation => "Awaiting confirmation",
            Self::Executing => "Running",
            Self::Paused => "Paused",
            Self::Verifying => "Verifying",
            Self::Delivering => "Delivering",
            Self::Succeeded => "Completed",
            Self::Failed => "Failed",
            Self::Cancelled => "Cancelled",
        }
    }

    fn allowed_from(self) -> &'static [TaskState] {
        use TaskState::*;
        match self {
            Pending => &[Understanding, Cancelled, Failed],
            Understanding => &[AwaitingConfirmation, Executing, Failed, Cancelled],
            AwaitingConfirmation => &[Executing, Understanding, Cancelled, Failed],
            Executing => &[AwaitingConfirmation, Paused, Verifying, Failed, Cancelled],
            Paused => &[Executing, Cancelled, Failed],
            Verifying => &[Delivering, Executing, Failed, Cancelled],
            Delivering => &[Succeeded, Failed, Cancelled],
            Succeeded | Failed | Cancelled => &[],
        }
    }

    pub fn can_transition_to(self, next: TaskState) -> bool {
        self.allowed_from().contains(&next)
    }

    pub fn transition_to(self, next: TaskState) -> Result<TaskState> {
        if self.can_transition_to(next) {
            Ok(next)
        } else {
            Err(CoreError::IllegalTransition {
                from: self.as_str(),
                to: next.as_str(),
            })
        }
    }
}

impl fmt::Display for TaskState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for TaskState {
    type Err = CoreError;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "pending" => Ok(Self::Pending),
            "understanding" => Ok(Self::Understanding),
            "awaiting_confirmation" => Ok(Self::AwaitingConfirmation),
            "executing" => Ok(Self::Executing),
            "paused" => Ok(Self::Paused),
            "verifying" => Ok(Self::Verifying),
            "delivering" => Ok(Self::Delivering),
            "succeeded" => Ok(Self::Succeeded),
            "failed" => Ok(Self::Failed),
            "cancelled" => Ok(Self::Cancelled),
            other => Err(CoreError::Invalid(format!("unknown task state: {other}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TaskState::*;
    use super::*;

    #[test]
    fn happy_path_is_reachable() {
        let mut state = Pending;
        for next in [
            Understanding,
            AwaitingConfirmation,
            Executing,
            Verifying,
            Delivering,
            Succeeded,
        ] {
            state = state
                .transition_to(next)
                .expect("transition should be allowed");
        }
        assert!(state.is_terminal());
    }

    #[test]
    fn terminal_states_are_frozen() {
        assert!(Succeeded.transition_to(Executing).is_err());
        assert!(Failed.transition_to(Understanding).is_err());
        assert!(Cancelled.transition_to(Cancelled).is_err());
    }

    #[test]
    fn execution_can_pause_and_request_approval() {
        assert!(Executing.can_transition_to(Paused));
        assert!(Executing.can_transition_to(AwaitingConfirmation));
        assert!(AwaitingConfirmation.can_transition_to(Understanding));
        assert!(!Verifying.can_transition_to(Succeeded));
    }

    #[test]
    fn roundtrips_through_string() {
        for state in [
            Pending,
            Understanding,
            AwaitingConfirmation,
            Executing,
            Paused,
            Verifying,
            Delivering,
            Succeeded,
            Failed,
            Cancelled,
        ] {
            assert_eq!(state.as_str().parse::<TaskState>().unwrap(), state);
        }
    }
}
