use relay_core::{Workspace, WorkspaceId};
use rusqlite::{params, OptionalExtension, Row};

use crate::{fmt_time, parse_time, Result, Store};

const SELECT: &str =
    "SELECT id, name, root_dir, rules, keywords, created_at, updated_at FROM workspaces";

fn to_workspace(
    row: &Row<'_>,
) -> rusqlite::Result<(String, String, String, String, String, String, String)> {
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

fn build(row: (String, String, String, String, String, String, String)) -> Result<Workspace> {
    Ok(Workspace {
        id: row
            .0
            .parse()
            .map_err(|e: uuid::Error| crate::StorageError::InvalidId(e.to_string()))?,
        name: row.1,
        root_dir: row.2.into(),
        rules: row.3,
        keywords: serde_json::from_str(&row.4)?,
        created_at: parse_time(row.5)?,
        updated_at: parse_time(row.6)?,
    })
}

impl Store {
    pub fn save_workspace(&self, workspace: &Workspace) -> Result<()> {
        self.conn().execute(
            "INSERT INTO workspaces (id, name, root_dir, rules, keywords, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                root_dir = excluded.root_dir,
                rules = excluded.rules,
                keywords = excluded.keywords,
                updated_at = excluded.updated_at",
            params![
                workspace.id.to_string(),
                workspace.name,
                workspace.root_dir.to_string_lossy(),
                workspace.rules,
                serde_json::to_string(&workspace.keywords)?,
                fmt_time(workspace.created_at),
                fmt_time(workspace.updated_at),
            ],
        )?;
        Ok(())
    }

    pub fn workspace(&self, id: WorkspaceId) -> Result<Option<Workspace>> {
        let row = self
            .conn()
            .query_row(
                &format!("{SELECT} WHERE id = ?1"),
                params![id.to_string()],
                to_workspace,
            )
            .optional()?;
        row.map(build).transpose()
    }

    pub fn workspaces(&self) -> Result<Vec<Workspace>> {
        let conn = self.conn();
        let mut stmt = conn.prepare(&format!("{SELECT} ORDER BY updated_at DESC"))?;
        let rows = stmt
            .query_map([], to_workspace)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        drop(stmt);
        drop(conn);
        rows.into_iter().map(build).collect()
    }

    pub fn workspace_by_name(&self, name: &str) -> Result<Option<Workspace>> {
        let row = self
            .conn()
            .query_row(
                &format!("{SELECT} WHERE name = ?1"),
                params![name],
                to_workspace,
            )
            .optional()?;
        row.map(build).transpose()
    }
}
