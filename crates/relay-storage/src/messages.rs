use chrono::Utc;
use relay_core::{
    plan_text, user_text, ChatMessageRecord, MessageRole, SessionId, Task, TaskId, TaskState,
    ToolMeta,
};
use rusqlite::{params, OptionalExtension};

use crate::{fmt_time, parse_time, Result, Store};

const SELECT: &str =
    "SELECT id, session_id, task_id, turn, role, content, meta, created_at FROM messages";

fn build(row: &rusqlite::Row<'_>) -> Result<ChatMessageRecord> {
    let session_id: String = row.get(1)?;
    let task_id: String = row.get(2)?;
    let role: String = row.get(4)?;
    let meta: Option<String> = row.get(6)?;
    let created_at: String = row.get(7)?;
    let invalid = |e: uuid::Error| crate::StorageError::InvalidId(e.to_string());
    Ok(ChatMessageRecord {
        id: row.get(0)?,
        session_id: session_id.parse().map_err(invalid)?,
        task_id: task_id.parse().map_err(invalid)?,
        turn: row.get(3)?,
        role: MessageRole::parse(&role)
            .ok_or_else(|| crate::StorageError::Invalid(format!("message role: {role}")))?,
        content: row.get(5)?,
        meta: meta.map(|raw| serde_json::from_str(&raw)).transpose()?,
        created_at: parse_time(created_at)?,
    })
}

impl Store {
    /// Create the message row first, then append content: while streaming, every block has a stable
    /// id, so the UI can grow it in place.
    pub fn open_message(
        &self,
        session_id: SessionId,
        task_id: TaskId,
        turn: u32,
        role: MessageRole,
        content: &str,
        meta: Option<&ToolMeta>,
    ) -> Result<i64> {
        self.open_message_at(session_id, task_id, turn, role, content, meta, Utc::now())
    }

    /// Persist with an explicit time: backfilling history keeps the time the task and steps happened.
    pub fn open_message_at(
        &self,
        session_id: SessionId,
        task_id: TaskId,
        turn: u32,
        role: MessageRole,
        content: &str,
        meta: Option<&ToolMeta>,
        created_at: chrono::DateTime<Utc>,
    ) -> Result<i64> {
        let meta = meta.map(serde_json::to_string).transpose()?;
        let conn = self.conn();
        conn.execute(
            "INSERT INTO messages (session_id, task_id, turn, role, content, meta, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                session_id.to_string(),
                task_id.to_string(),
                turn,
                role.as_str(),
                content,
                meta,
                fmt_time(created_at)
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// Streamed deltas are appended straight to the end of the column, so the caller only keeps a
    /// small buffer of what is not yet persisted.
    pub fn append_message(&self, id: i64, delta: &str) -> Result<()> {
        if delta.is_empty() {
            return Ok(());
        }
        self.conn().execute(
            "UPDATE messages SET content = content || ?2 WHERE id = ?1",
            params![id, delta],
        )?;
        Ok(())
    }

    pub fn update_message_meta(&self, id: i64, meta: &ToolMeta) -> Result<()> {
        self.conn().execute(
            "UPDATE messages SET meta = ?2 WHERE id = ?1",
            params![id, serde_json::to_string(meta)?],
        )?;
        Ok(())
    }

    pub fn message(&self, id: i64) -> Result<Option<ChatMessageRecord>> {
        let raw = self
            .conn()
            .query_row(&format!("{SELECT} WHERE id = ?1"), params![id], |row| {
                Ok(build(row))
            })
            .optional()?;
        raw.transpose()
    }

    /// Whether the task already has visible reply text (used to complement turns that only submit a
    /// conclusion).
    pub fn has_assistant_text(&self, task_id: TaskId) -> Result<bool> {
        let found: Option<i64> = self
            .conn()
            .query_row(
                "SELECT 1 FROM messages
                 WHERE task_id = ?1 AND role = 'assistant' AND trim(content) <> ''
                 LIMIT 1",
                params![task_id.to_string()],
                |row| row.get(0),
            )
            .optional()?;
        Ok(found.is_some())
    }

    /// The task's final reply: the last assistant message with visible text.
    pub fn last_assistant_text(&self, task_id: TaskId) -> Result<Option<String>> {
        let text: Option<String> = self
            .conn()
            .query_row(
                "SELECT content FROM messages
                 WHERE task_id = ?1 AND role = 'assistant' AND trim(content) <> ''
                 ORDER BY id DESC LIMIT 1",
                params![task_id.to_string()],
                |row| row.get(0),
            )
            .optional()?;
        Ok(text)
    }

    /// Session replay: returned in persistence order, i.e. the real chronology of user messages /
    /// thinking / tools / replies.
    pub fn messages_for_session(&self, session_id: SessionId) -> Result<Vec<ChatMessageRecord>> {
        let conn = self.conn();
        let mut stmt = conn.prepare(&format!("{SELECT} WHERE session_id = ?1 ORDER BY id ASC"))?;
        let rows = stmt
            .query_map(params![session_id.to_string()], |row| Ok(build(row)))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        drop(stmt);
        drop(conn);
        rows.into_iter().collect()
    }

    /// Read only tasks in the same workspace submitted before the current user message. A queued
    /// predecessor may produce its final reply after the current message, so the boundary applies
    /// to user messages.
    pub fn prior_dialogue_messages(
        &self,
        session: SessionId,
        current: TaskId,
        turns: usize,
    ) -> Result<Vec<ChatMessageRecord>> {
        let conn = self.conn();
        let sql = format!(
            "{SELECT}
            WHERE session_id = ?1 AND role IN ('user', 'assistant') AND task_id IN (
                SELECT m.task_id FROM messages m JOIN tasks t ON t.id = m.task_id
                WHERE m.session_id = ?1 AND m.role = 'user'
                AND m.id < (SELECT MIN(id) FROM messages WHERE task_id = ?2 AND role = 'user')
                AND t.workspace_id = (SELECT workspace_id FROM tasks WHERE id = ?2)
                ORDER BY m.id DESC LIMIT ?3
            ) ORDER BY id"
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(
            params![session.to_string(), current.to_string(), turns],
            |row| Ok(build(row)),
        )?;
        rows.collect::<rusqlite::Result<Vec<_>>>()?
            .into_iter()
            .collect()
    }

    /// Legacy tasks have no message records: backfill them from the task, plan and steps, so old
    /// sessions replay in the new UI too.
    pub fn backfill_messages(&self) -> Result<()> {
        let task_ids: Vec<String> = {
            let conn = self.conn();
            let mut stmt = conn.prepare(
                "SELECT tasks.id FROM tasks
                 WHERE NOT EXISTS (SELECT 1 FROM messages WHERE messages.task_id = tasks.id)",
            )?;
            let collected = stmt
                .query_map([], |row| row.get::<_, String>(0))?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            drop(stmt);
            drop(conn);
            collected
        };
        for raw in task_ids {
            let Ok(task_id) = raw.parse::<TaskId>() else {
                continue;
            };
            let Ok(Some(task)) = self.task(task_id) else {
                continue;
            };
            self.backfill_task(&task)?;
        }
        Ok(())
    }

    fn backfill_task(&self, task: &Task) -> Result<()> {
        self.open_message_at(
            task.session_id,
            task.id,
            0,
            MessageRole::User,
            &user_text(&task.instruction, task.supplement.as_deref()),
            None,
            task.created_at,
        )?;
        if let Some(plan) = &task.plan {
            self.open_message_at(
                task.session_id,
                task.id,
                0,
                MessageRole::Plan,
                &plan_text(plan),
                None,
                task.created_at,
            )?;
        }
        for step in self.steps(task.id)? {
            let mut output = step.stdout.trim_end().to_string();
            let stderr = step.stderr.trim_end();
            if !stderr.is_empty() {
                if !output.is_empty() {
                    output.push('\n');
                }
                output.push_str(stderr);
            }
            let meta = ToolMeta {
                done: true,
                exit_code: step.exit_code,
                timed_out: step.timed_out,
                duration_ms: step.duration_ms,
                output,
            };
            self.open_message_at(
                task.session_id,
                task.id,
                step.turn,
                MessageRole::Tool,
                &step.script,
                Some(&meta),
                step.created_at,
            )?;
        }
        if let Some(error) = &task.error {
            self.open_message_at(
                task.session_id,
                task.id,
                0,
                MessageRole::Error,
                error,
                None,
                task.updated_at,
            )?;
        } else if task.state == TaskState::Cancelled {
            self.open_message_at(
                task.session_id,
                task.id,
                0,
                MessageRole::Notice,
                "Task cancelled",
                None,
                task.updated_at,
            )?;
        }
        Ok(())
    }
}
