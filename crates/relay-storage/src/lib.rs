//! SQLite persistence layer. The runtime is the only writer; the UI is read-only.

mod artifacts;
mod model_requests;
pub use model_requests::ExecutorCheckpoint;
mod events;
mod messages;
mod sessions;
mod steps;
mod tasks;
mod workspaces;

use std::path::Path;
use std::sync::{Mutex, MutexGuard};

use chrono::{DateTime, Utc};
use rusqlite::Connection;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum StorageError {
    #[error("database error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("data serialization error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("time format error: {0}")]
    Time(String),
    #[error("invalid ID: {0}")]
    InvalidId(String),
    #[error("invalid data: {0}")]
    Invalid(String),
    #[error("record not found: {0}")]
    NotFound(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, StorageError>;

const SCHEMA: &str = include_str!("schema.sql");

/// Single-process SQLite store: on desktop only the runtime writes to the database, so one lock
/// keeps the implementation simple.
pub struct Store {
    conn: Mutex<Connection>,
}

impl Store {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;
        Self::setup(conn)
    }

    pub fn open_in_memory() -> Result<Self> {
        Self::setup(Connection::open_in_memory()?)
    }

    fn setup(conn: Connection) -> Result<Self> {
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        conn.execute_batch(SCHEMA)?;
        ensure_columns(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    pub(crate) fn conn(&self) -> MutexGuard<'_, Connection> {
        self.conn
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// Add columns to legacy databases: `CREATE TABLE IF NOT EXISTS` only applies to new databases,
/// existing tables need an explicit ALTER.
fn ensure_columns(conn: &Connection) -> Result<()> {
    let mut stmt = conn.prepare("PRAGMA table_info(tasks)")?;
    let columns: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    drop(stmt);
    if !columns.iter().any(|name| name == "usage") {
        conn.execute(
            "ALTER TABLE tasks ADD COLUMN usage TEXT NOT NULL DEFAULT '{}'",
            [],
        )?;
    }
    Ok(())
}

pub(crate) fn fmt_time(value: DateTime<Utc>) -> String {
    value.to_rfc3339()
}

pub(crate) fn parse_time(value: String) -> Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(&value)
        .map(|t| t.with_timezone(&Utc))
        .map_err(|err| StorageError::Time(err.to_string()))
}

pub(crate) fn parse_opt_time(value: Option<String>) -> Result<Option<DateTime<Utc>>> {
    value.map(parse_time).transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Legacy databases have no usage column: it is added on open, and existing rows read back as
    /// the default value.
    #[test]
    fn adds_usage_column_to_legacy_database() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE tasks (
                id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL, session_id TEXT NOT NULL,
                title TEXT NOT NULL, instruction TEXT NOT NULL, supplement TEXT,
                context TEXT NOT NULL, plan TEXT, state TEXT NOT NULL, report TEXT,
                delivery TEXT NOT NULL, error TEXT, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, started_at TEXT, finished_at TEXT
             );
             INSERT INTO tasks VALUES ('t1','w1','s1','Title','Instruction',NULL,'{}',NULL,'pending',
                NULL,'notify_only',NULL,'2026-01-01T00:00:00+00:00','2026-01-01T00:00:00+00:00',NULL,NULL);",
        )
        .unwrap();

        ensure_columns(&conn).unwrap();
        let usage: String = conn
            .query_row("SELECT usage FROM tasks WHERE id = 't1'", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(usage, "{}");

        // Re-running once the column exists must not fail.
        ensure_columns(&conn).unwrap();
    }
}
