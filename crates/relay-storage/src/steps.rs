use relay_core::{ControlPhase, TaskId, TaskStep};
use rusqlite::params;

use crate::{fmt_time, parse_time, Result, Store};

impl Store {
    pub fn add_step(&self, step: &TaskStep) -> Result<()> {
        self.conn().execute(
            "INSERT INTO task_steps (id, task_id, turn, script, stdout, stderr, exit_code, timed_out,
                duration_ms, phase, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                step.id,
                step.task_id.to_string(),
                step.turn,
                step.script,
                step.stdout,
                step.stderr,
                step.exit_code,
                step.timed_out as i64,
                step.duration_ms as i64,
                step.phase.map(|p| p.as_str()),
                fmt_time(step.created_at),
            ],
        )?;
        Ok(())
    }

    pub fn steps(&self, task_id: TaskId) -> Result<Vec<TaskStep>> {
        let conn = self.conn();
        let mut stmt = conn.prepare(
            "SELECT id, task_id, turn, script, stdout, stderr, exit_code, timed_out, duration_ms, phase, created_at
             FROM task_steps WHERE task_id = ?1 ORDER BY turn ASC, created_at ASC",
        )?;
        let rows = stmt
            .query_map(params![task_id.to_string()], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, u32>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, Option<i32>>(6)?,
                    row.get::<_, i64>(7)?,
                    row.get::<_, i64>(8)?,
                    row.get::<_, Option<String>>(9)?,
                    row.get::<_, String>(10)?,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        drop(stmt);
        drop(conn);

        rows.into_iter()
            .map(|raw| {
                Ok(TaskStep {
                    id: raw.0,
                    task_id: raw
                        .1
                        .parse()
                        .map_err(|e: uuid::Error| crate::StorageError::InvalidId(e.to_string()))?,
                    turn: raw.2,
                    script: raw.3,
                    stdout: raw.4,
                    stderr: raw.5,
                    exit_code: raw.6,
                    timed_out: raw.7 != 0,
                    duration_ms: raw.8.max(0) as u64,
                    phase: raw
                        .9
                        .map(|p| p.parse::<ControlPhase>())
                        .transpose()
                        .map_err(|e| crate::StorageError::Invalid(e.to_string()))?,
                    created_at: parse_time(raw.10)?,
                })
            })
            .collect()
    }
}
