use relay_core::{Artifact, ArtifactKind, TaskId};
use rusqlite::params;

use crate::{fmt_time, parse_time, Result, Store};

fn kind_str(kind: &ArtifactKind) -> &'static str {
    match kind {
        ArtifactKind::File => "file",
        ArtifactKind::Folder => "folder",
        ArtifactKind::Log => "log",
        ArtifactKind::Other => "other",
    }
}

fn kind_from(value: &str) -> ArtifactKind {
    match value {
        "folder" => ArtifactKind::Folder,
        "log" => ArtifactKind::Log,
        "other" => ArtifactKind::Other,
        _ => ArtifactKind::File,
    }
}

impl Store {
    pub fn add_artifact(&self, artifact: &Artifact) -> Result<()> {
        self.conn().execute(
            "INSERT INTO artifacts (id, task_id, path, kind, size, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(id) DO NOTHING",
            params![
                artifact.id,
                artifact.task_id.to_string(),
                artifact.path.to_string_lossy(),
                kind_str(&artifact.kind),
                artifact.size as i64,
                fmt_time(artifact.created_at),
            ],
        )?;
        Ok(())
    }

    pub fn artifacts(&self, task_id: TaskId) -> Result<Vec<Artifact>> {
        let conn = self.conn();
        let mut stmt = conn.prepare(
            "SELECT id, task_id, path, kind, size, created_at FROM artifacts WHERE task_id = ?1 ORDER BY created_at ASC",
        )?;
        let rows = stmt
            .query_map(params![task_id.to_string()], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, String>(5)?,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        drop(stmt);
        drop(conn);

        rows.into_iter()
            .map(|raw| {
                Ok(Artifact {
                    id: raw.0,
                    task_id: raw
                        .1
                        .parse()
                        .map_err(|e: uuid::Error| crate::StorageError::InvalidId(e.to_string()))?,
                    path: raw.2.into(),
                    kind: kind_from(&raw.3),
                    size: raw.4.max(0) as u64,
                    created_at: parse_time(raw.5)?,
                })
            })
            .collect()
    }
}
