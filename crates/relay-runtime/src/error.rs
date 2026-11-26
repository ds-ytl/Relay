use relay_core::{CoreError, TaskId};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("task not found: {0}")]
    TaskNotFound(TaskId),
    #[error("storage error: {0}")]
    Storage(#[from] relay_storage::StorageError),
    #[error("model error: {0}")]
    Model(#[from] relay_openai::ModelError),
    #[error("execution error: {0}")]
    Executor(#[from] relay_executor::ExecutorError),
    #[error("context error: {0}")]
    Context(#[from] relay_context::ContextError),
    #[error("domain error: {0}")]
    Core(#[from] CoreError),
    #[error("no pending confirmation: {0}")]
    NoPendingConfirmation(TaskId),
    #[error("invalid request: {0}")]
    Invalid(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, RuntimeError>;
