use relay_core::{RuntimeEvent, TaskId};
use rusqlite::params;

use crate::{fmt_time, Result, Store};

impl Store {
    /// Events drive UI pushes and also serve as an audit trail; kind is its own column for filtering.
    pub fn log_event(&self, event: &RuntimeEvent) -> Result<()> {
        let payload = serde_json::to_string(event)?;
        let kind = event.kind.tag().to_string();
        self.conn().execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![event.task_id.to_string(), kind, payload, fmt_time(event.at)],
        )?;
        Ok(())
    }

    pub fn events_for(&self, task_id: TaskId, limit: u32) -> Result<Vec<RuntimeEvent>> {
        let conn = self.conn();
        let mut stmt = conn.prepare(
            "SELECT payload FROM task_events WHERE task_id = ?1 ORDER BY id DESC LIMIT ?2",
        )?;
        let rows = stmt
            .query_map(params![task_id.to_string(), limit], |row| {
                row.get::<_, String>(0)
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        drop(stmt);
        drop(conn);
        rows.into_iter()
            .map(|payload| serde_json::from_str(&payload).map_err(Into::into))
            .collect()
    }
}
