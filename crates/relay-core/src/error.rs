use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("illegal task state transition: {from} -> {to}")]
    IllegalTransition {
        from: &'static str,
        to: &'static str,
    },
    #[error("path is outside the authorized scope: {0}")]
    UnauthorizedPath(String),
    #[error("invalid argument: {0}")]
    Invalid(String),
}

pub type Result<T, E = CoreError> = std::result::Result<T, E>;
