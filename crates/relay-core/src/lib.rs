//! Relay core domain: workspaces, sessions, tasks, the task state machine,
//! environment context and the execution policy shared by every agent.

pub mod context;
pub mod error;
pub mod event;
pub mod ids;
pub mod message;
pub mod model;
pub mod plan;
pub mod policy;
pub mod state;

pub use context::{ContextSource, EnvironmentSnapshot};
pub use error::{CoreError, Result};
pub use event::{RuntimeEvent, RuntimeEventKind};
pub use ids::{SessionId, TaskId, WorkspaceId};
pub use message::{plan_text, user_text, ChatMessageRecord, MessageRole, ToolMeta};
pub use model::{
    Artifact, ArtifactKind, ContextUsage, ControlPhase, ControlSignal, DeliveryMode, Session, Task,
    TaskReport, TaskStep, Verdict, Workspace,
};
pub use plan::Plan;
pub use policy::ExecutionPolicy;
pub use state::TaskState;
