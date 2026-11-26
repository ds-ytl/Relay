use chrono::{DateTime, Utc};
use relay_core::{Session, SessionId, WorkspaceId};
use rusqlite::{params, OptionalExtension, Row};

use crate::{fmt_time, parse_time, Result, Store};

/// Lightweight view for session lists (avoids parsing the whole task table).
#[derive(Debug, Clone)]
pub struct SessionRow {
    pub id: SessionId,
    pub workspace_id: WorkspaceId,
    pub title: String,
    pub summary: String,
    pub last_context: Option<String>,
    pub last_active_at: DateTime<Utc>,
}

const SELECT: &str = "SELECT id, workspace_id, title, summary, last_context, created_at, last_active_at FROM sessions";

fn to_raw(
    row: &Row<'_>,
) -> rusqlite::Result<(
    String,
    String,
    String,
    String,
    Option<String>,
    String,
    String,
)> {
    Ok((
        row.get(0)?,
        row.get(1)?,
        row.get(2)?,
        row.get(3)?,
        row.get(4)?,
        row.get(5)?,
        row.get(6)?,
    ))
}

fn build(
    raw: (
        String,
        String,
        String,
        String,
        Option<String>,
        String,
        String,
    ),
) -> Result<Session> {
    Ok(Session {
        id: raw
            .0
            .parse()
            .map_err(|e: uuid::Error| crate::StorageError::InvalidId(e.to_string()))?,
        workspace_id: raw
            .1
            .parse()
            .map_err(|e: uuid::Error| crate::StorageError::InvalidId(e.to_string()))?,
        title: raw.2,
        summary: raw.3,
        last_context: raw.4,
        created_at: parse_time(raw.5)?,
        last_active_at: parse_time(raw.6)?,
    })
}

impl Store {
    pub fn save_session(&self, session: &Session) -> Result<()> {
        self.conn().execute(
            "INSERT INTO sessions (id, workspace_id, title, summary, last_context, created_at, last_active_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                summary = excluded.summary,
                last_context = excluded.last_context,
                last_active_at = excluded.last_active_at",
            params![
                session.id.to_string(),
                session.workspace_id.to_string(),
                session.title,
                session.summary,
                session.last_context,
                fmt_time(session.created_at),
                fmt_time(session.last_active_at),
            ],
        )?;
        Ok(())
    }

    pub fn session(&self, id: SessionId) -> Result<Option<Session>> {
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

    pub fn sessions(&self, workspace_id: WorkspaceId, limit: u32) -> Result<Vec<Session>> {
        let conn = self.conn();
        let mut stmt = conn.prepare(&format!(
            "{SELECT} WHERE workspace_id = ?1 ORDER BY last_active_at DESC LIMIT ?2"
        ))?;
        let rows = stmt
            .query_map(params![workspace_id.to_string(), limit], to_raw)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        drop(stmt);
        drop(conn);
        rows.into_iter().map(build).collect()
    }

    /// Most recently active session in a workspace, used for "continue last".
    pub fn latest_session(&self, workspace_id: WorkspaceId) -> Result<Option<Session>> {
        let raw = self
            .conn()
            .query_row(
                &format!("{SELECT} WHERE workspace_id = ?1 ORDER BY last_active_at DESC LIMIT 1"),
                params![workspace_id.to_string()],
                to_raw,
            )
            .optional()?;
        raw.map(build).transpose()
    }

    pub fn touch_session(&self, id: SessionId, summary: &str, context: Option<&str>) -> Result<()> {
        self.conn().execute(
            "UPDATE sessions SET last_active_at = ?2, summary = ?3, last_context = COALESCE(?4, last_context) WHERE id = ?1",
            params![id.to_string(), fmt_time(Utc::now()), summary, context],
        )?;
        Ok(())
    }

    pub fn sessions_view(&self, workspace_id: WorkspaceId, limit: u32) -> Result<Vec<SessionRow>> {
        Ok(self
            .sessions(workspace_id, limit)?
            .into_iter()
            .map(|s| SessionRow {
                id: s.id,
                workspace_id: s.workspace_id,
                title: s.title,
                summary: s.summary,
                last_context: s.last_context,
                last_active_at: s.last_active_at,
            })
            .collect())
    }
}

impl Store {
    /// Recent sessions across workspaces, needed for routing.
    pub fn all_sessions(&self, limit: u32) -> Result<Vec<Session>> {
        let conn = self.conn();
        let mut stmt = conn.prepare(&format!("{SELECT} ORDER BY last_active_at DESC LIMIT ?1"))?;
        let rows = stmt
            .query_map(params![limit], to_raw)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        drop(stmt);
        drop(conn);
        rows.into_iter().map(build).collect()
    }
}
