PRAGMA user_version = 1;

CREATE TABLE IF NOT EXISTS workspaces (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    root_dir    TEXT NOT NULL,
    rules       TEXT NOT NULL DEFAULT '',
    keywords    TEXT NOT NULL DEFAULT '[]',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS workspaces_name ON workspaces(name);

CREATE TABLE IF NOT EXISTS sessions (
    id             TEXT PRIMARY KEY,
    workspace_id   TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    title          TEXT NOT NULL,
    summary        TEXT NOT NULL DEFAULT '',
    last_context   TEXT,
    created_at     TEXT NOT NULL,
    last_active_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS sessions_workspace ON sessions(workspace_id, last_active_at DESC);

CREATE TABLE IF NOT EXISTS tasks (
    id           TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    session_id   TEXT NOT NULL,
    title        TEXT NOT NULL,
    instruction  TEXT NOT NULL,
    supplement   TEXT,
    context      TEXT NOT NULL,
    usage        TEXT NOT NULL DEFAULT '{}',
    plan         TEXT,
    state        TEXT NOT NULL,
    report       TEXT,
    delivery     TEXT NOT NULL,
    error        TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    started_at   TEXT,
    finished_at  TEXT
);
CREATE INDEX IF NOT EXISTS tasks_session ON tasks(session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS tasks_state ON tasks(state, created_at DESC);

CREATE TABLE IF NOT EXISTS task_steps (
    id          TEXT PRIMARY KEY,
    task_id     TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    turn        INTEGER NOT NULL,
    script      TEXT NOT NULL,
    stdout      TEXT NOT NULL DEFAULT '',
    stderr      TEXT NOT NULL DEFAULT '',
    exit_code   INTEGER,
    timed_out   INTEGER NOT NULL DEFAULT 0,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    phase       TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS task_steps_task ON task_steps(task_id, turn);

CREATE TABLE IF NOT EXISTS artifacts (
    id         TEXT PRIMARY KEY,
    task_id    TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    path       TEXT NOT NULL,
    kind       TEXT NOT NULL,
    size       INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS artifacts_task ON artifacts(task_id);

CREATE TABLE IF NOT EXISTS task_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    TEXT NOT NULL,
    kind       TEXT NOT NULL,
    payload    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS task_events_task ON task_events(task_id, id);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Session message stream: one record is one bubble / card in the UI; ascending id is the real order.
-- Thinking and reply are streamed blocks: create the row first, then keep appending content, so a
-- crashed process loses no generated part.
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    task_id    TEXT NOT NULL,
    turn       INTEGER NOT NULL DEFAULT 0,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL DEFAULT '',
    meta       TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_session ON messages(session_id, id);
CREATE INDEX IF NOT EXISTS messages_task ON messages(task_id, id);

-- Requests are immutable. Shared input items (including inline images) are stored once.
CREATE TABLE IF NOT EXISTS model_inputs (
    digest TEXT PRIMARY KEY,
    value BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS model_requests (
    id INTEGER PRIMARY KEY,
    task_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    header BLOB NOT NULL,
    input_keys TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS model_requests_task ON model_requests(task_id, id);
CREATE TABLE IF NOT EXISTS executor_checkpoints (
    task_id TEXT PRIMARY KEY,
    request_id INTEGER NOT NULL REFERENCES model_requests(id),
    turn INTEGER NOT NULL,
    suffix BLOB NOT NULL
);
