use crate::{Result, StorageError, Store};
use relay_core::TaskId;
use rusqlite::{params, OptionalExtension};
use serde_json::Value;
use sha2::{Digest, Sha256};

pub struct ExecutorCheckpoint {
    pub turn: u32,
    pub payload: Value,
    pub suffix: Vec<Value>,
}

impl Store {
    /// Commit every input and its ordered manifest before dispatch. A crash never leaves
    /// a manifest pointing at a partially stored request.
    pub fn record_model_request(
        &self,
        task: TaskId,
        provider: &str,
        mut body: Value,
    ) -> Result<i64> {
        let inputs = body
            .get_mut("input")
            .and_then(Value::as_array_mut)
            .ok_or_else(|| StorageError::Invalid("expected a Responses input array".into()))?;
        let mut conn = self.conn();
        let tx = conn.transaction()?;
        let mut keys = Vec::with_capacity(inputs.len());
        for input in inputs.drain(..) {
            let bytes = serde_json::to_vec(&input)?;
            let key = format!("{:x}", Sha256::digest(&bytes));
            tx.execute(
                "INSERT OR IGNORE INTO model_inputs(digest,value) VALUES(?1,?2)",
                params![key, bytes],
            )?;
            keys.push(key);
        }
        tx.execute(
            "INSERT INTO model_requests(task_id,provider,header,input_keys) VALUES(?1,?2,?3,?4)",
            params![
                task.to_string(),
                provider,
                serde_json::to_vec(&body)?,
                serde_json::to_string(&keys)?
            ],
        )?;
        let id = tx.last_insert_rowid();
        tx.commit()?;
        Ok(id)
    }

    pub fn model_request(&self, id: i64) -> Result<Value> {
        let conn = self.conn();
        let (header, keys): (Vec<u8>, String) = conn.query_row(
            "SELECT header,input_keys FROM model_requests WHERE id=?1",
            [id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        let mut payload: Value = serde_json::from_slice(&header)?;
        let mut query = conn.prepare("SELECT value FROM model_inputs WHERE digest=?1")?;
        let mut items = Vec::new();
        for key in serde_json::from_str::<Vec<String>>(&keys)? {
            let bytes: Vec<u8> = query.query_row([key], |r| r.get(0))?;
            items.push(serde_json::from_slice(&bytes)?);
        }
        payload["input"] = Value::Array(items);
        Ok(payload)
    }

    /// Publish only a completed exchange, after its tool outcomes are known.
    pub fn checkpoint_executor(
        &self,
        task: TaskId,
        request: i64,
        turn: u32,
        suffix: &[Value],
    ) -> Result<()> {
        self.conn().execute("INSERT INTO executor_checkpoints(task_id,request_id,turn,suffix) VALUES(?1,?2,?3,?4)
            ON CONFLICT(task_id) DO UPDATE SET request_id=excluded.request_id,turn=excluded.turn,suffix=excluded.suffix
            WHERE excluded.request_id >= executor_checkpoints.request_id",
            params![task.to_string(),request,turn,serde_json::to_vec(suffix)?])?;
        Ok(())
    }

    pub fn executor_checkpoint(
        &self,
        task: TaskId,
        provider: &str,
    ) -> Result<Option<ExecutorCheckpoint>> {
        let record: Option<(i64,u32,Vec<u8>)> = self.conn().query_row(
            "SELECT c.request_id,c.turn,c.suffix FROM executor_checkpoints c JOIN model_requests r ON r.id=c.request_id
             WHERE c.task_id=?1 AND r.provider=?2", params![task.to_string(),provider],
            |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional()?;
        record
            .map(|(id, turn, bytes)| {
                Ok(ExecutorCheckpoint {
                    turn,
                    payload: self.model_request(id)?,
                    suffix: serde_json::from_slice(&bytes)?,
                })
            })
            .transpose()
    }

    /// Headers of the most recent requests of a task, newest first.
    ///
    /// A crash can land between `record_model_request` and `checkpoint_executor`. The request is
    /// then the only copy of what the model really saw, so a reopened task can replay it. Headers
    /// are stored without their input array, which keeps this scan cheap; the caller picks the
    /// request it can replay and loads it with `model_request`.
    pub fn recent_model_requests(
        &self,
        task: TaskId,
        provider: &str,
        limit: u32,
    ) -> Result<Vec<(i64, Value)>> {
        let conn = self.conn();
        let mut query = conn.prepare(
            "SELECT id,header FROM model_requests WHERE task_id=?1 AND provider=?2 ORDER BY id DESC LIMIT ?3",
        )?;
        let rows = query.query_map(params![task.to_string(), provider, limit], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, Vec<u8>>(1)?))
        })?;
        let mut headers = Vec::new();
        for row in rows {
            let (id, bytes) = row?;
            headers.push((id, serde_json::from_slice(&bytes)?));
        }
        Ok(headers)
    }
}
