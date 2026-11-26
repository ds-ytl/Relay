//! Agent Runtime: state machine, task scheduling, context, approval, execution and delivery.
//!
//! Router / Understand / Executor / Verifier share one Runtime and differ only in System Prompt,
//! context and permissions; no extra multi-agent framework is introduced.

pub mod agents;
pub mod budget;
pub(crate) mod computer;
pub mod config;
pub mod error;
mod model_history;
pub mod runtime;
pub(crate) mod tools;
pub mod understanding;
pub mod vision;

pub use config::{resolve_within, slugify, RuntimeConfig};
pub use error::{Result, RuntimeError};
pub use runtime::{Confirmation, Runtime, TaskRequest, TaskView};
pub use understanding::{
    is_blank_instruction, normalize_instruction, PrefetchedScene, UnderstandEvent, UnderstandStage,
    Understanding, UnderstandingAgent, UnderstandingInput, BLANK_INSTRUCTION,
};
