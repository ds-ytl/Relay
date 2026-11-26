use chrono::Utc;
use relay_core::{Task, TaskId, TaskState, WorkspaceId};
use rusqlite::{params, OptionalExtension, Row};

use crate::{fmt_time, parse_opt_time, parse_time, Result, Store};

const SELECT: &str =
    "SELECT id, workspace_id, session_id, title, instruction, supplement, context, plan, \
    state, report, delivery, error, created_at, updated_at, started_at, finished_at, usage FROM tasks";

type RawTask = (
    String,
    String,
    String,
    String,
    String,
    Option<String>,
    String,
    Option<String>,
    String,
    Option<String>,
    String,
    Option<String>,
    String,
    String,
    Option<String>,
    Option<String>,
    Option<String>,
);

fn to_raw(row: &Row<'_>) -> rusqlite::Result<RawTask> {
    Ok((
        row.get(0)?,
        row.get(1)?,
        row.get(2)?,
        row.get(3)?,
        row.get(4)?,
        row.get(5)?,
        row.get(6)?,
        row.get(7)?,
        row.get(8)?,
        row.get(9)?,
        row.get(10)?,
        row.get(11)?,
        row.get(12)?,
        row.get(13)?,
        row.get(14)?,
        row.get(15)?,
        row.get(16)?,
    ))
}

fn build(raw: RawTask) -> Result<Task> {
    let invalid = |e: uuid::Error| crate::StorageError::InvalidId(e.to_string());
    Ok(Task {
        id: raw.0.parse().map_err(invalid)?,
        workspace_id: raw.1.parse().map_err(invalid)?,
        session_id: raw.2.parse().map_err(invalid)?,
        title: raw.3,
        instruction: raw.4,
        supplement: raw.5,
        context: serde_json::from_str(&raw.6)?,
        usage: match raw.16 {
            Some(text) if !text.trim().is_empty() => serde_json::from_str(&text)?,
            _ => relay_core::ContextUsage::default(),
        },
        plan: raw.7.map(|v| serde_json::from_str(&v)).transpose()?,
        state: raw
            .8
            .parse()
            .map_err(|_| crate::StorageError::Invalid(format!("task state: {}", raw.8)))?,
        report: raw.9.map(|v| serde_json::from_str(&v)).transpose()?,
        delivery: raw
            .10
            .parse()
            .map_err(|_| crate::StorageError::Invalid(format!("delivery mode: {}", raw.10)))?,
        error: raw.11,
        created_at: parse_time(raw.12)?,
        updated_at: parse_time(raw.13)?,
        started_at: parse_opt_time(raw.14)?,
        finished_at: parse_opt_time(raw.15)?,
    })
}

impl Store {
    pub fn save_task(&self, task: &Task) -> Result<()> {
        self.conn().execute(
            "INSERT INTO tasks (id, workspace_id, session_id, title, instruction, supplement, context,
                plan, state, report, delivery, error, created_at, updated_at, started_at, finished_at, usage)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)
             ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                instruction = excluded.instruction,
                supplement = excluded.supplement,
                context = excluded.context,
                plan = excluded.plan,
                state = excluded.state,
                report = excluded.report,
                delivery = excluded.delivery,
                usage = excluded.usage,
                error = excluded.error,
                updated_at = excluded.updated_at,
                started_at = excluded.started_at,
                finished_at = excluded.finished_at",
            params![
                task.id.to_string(),
                task.workspace_id.to_string(),
                task.session_id.to_string(),
                task.title,
                task.instruction,
                task.supplement,
                serde_json::to_string(&task.context)?,
                task.plan.as_ref().map(serde_json::to_string).transpose()?,
                task.state.as_str(),
                task.report.as_ref().map(serde_json::to_string).transpose()?,
                task.delivery.as_str(),
                task.error,
                fmt_time(task.created_at),
                fmt_time(task.updated_at),
                task.started_at.map(fmt_time),
                task.finished_at.map(fmt_time),
                serde_json::to_string(&task.usage)?,
            ],
        )?;
        Ok(())
    }

    pub fn task(&self, id: TaskId) -> Result<Option<Task>> {
        let raw = self
            .conn()
            .query_row(
                &format!("{SELECT} WHERE id = ?1"),
                params![id.to_string()],
                to_raw,
            )
            .optional()?;
        raw.map(build).transpose()
    }

    pub fn recent_tasks(&self, limit: u32) -> Result<Vec<Task>> {
        self.query_tasks(
            &format!("{SELECT} ORDER BY created_at DESC LIMIT ?1"),
            params![limit],
        )
    }

    pub fn tasks_by_session(
        &self,
        session_id: relay_core::SessionId,
        limit: u32,
    ) -> Result<Vec<Task>> {
        self.query_tasks(
            &format!("{SELECT} WHERE session_id = ?1 ORDER BY created_at DESC LIMIT ?2"),
            params![session_id.to_string(), limit],
        )
    }

    pub fn tasks_by_workspace(&self, workspace_id: WorkspaceId, limit: u32) -> Result<Vec<Task>> {
        self.query_tasks(
            &format!("{SELECT} WHERE workspace_id = ?1 ORDER BY created_at DESC LIMIT ?2"),
            params![workspace_id.to_string(), limit],
        )
    }

    /// Recover tasks: tasks still executing after a process restart must be taken over or marked
    /// as failed.
    pub fn unfinished_tasks(&self) -> Result<Vec<Task>> {
        self.query_tasks(
            &format!(
                "{SELECT} WHERE state NOT IN ('succeeded', 'failed', 'cancelled') ORDER BY created_at DESC"
            ),
            [],
        )
    }

    fn query_tasks(&self, sql: &str, params: impl rusqlite::Params) -> Result<Vec<Task>> {
        let conn = self.conn();
        let mut stmt = conn.prepare(sql)?;
        let rows = stmt
            .query_map(params, to_raw)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        drop(stmt);
        drop(conn);
        rows.into_iter().map(build).collect()
    }

    pub fn set_task_state(&self, id: TaskId, state: TaskState) -> Result<()> {
        let changed = self.conn().execute(
            "UPDATE tasks SET state = ?2, updated_at = ?3 WHERE id = ?1",
            params![id.to_string(), state.as_str(), fmt_time(Utc::now())],
        )?;
        if changed == 0 {
            return Err(crate::StorageError::NotFound(id.to_string()));
        }
        Ok(())
    }
}
