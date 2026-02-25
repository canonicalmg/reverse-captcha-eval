import json
import sqlite3
from datetime import datetime, timezone
from uuid import uuid4


def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS models (
            model_id TEXT PRIMARY KEY,
            name     TEXT NOT NULL,
            provider TEXT NOT NULL,
            version  TEXT
        );

        CREATE TABLE IF NOT EXISTS runs (
            run_id      TEXT PRIMARY KEY,
            created_at  TEXT NOT NULL,
            git_sha     TEXT,
            pack_id     TEXT NOT NULL,
            model_id    TEXT NOT NULL,
            params_json TEXT,
            FOREIGN KEY (model_id) REFERENCES models(model_id)
        );

        CREATE TABLE IF NOT EXISTS cases (
            case_id       TEXT PRIMARY KEY,
            pack_id       TEXT NOT NULL,
            scheme        TEXT,
            metadata_json TEXT,
            expected      TEXT
        );

        CREATE TABLE IF NOT EXISTS outputs (
            output_id  TEXT PRIMARY KEY,
            run_id     TEXT NOT NULL,
            case_id    TEXT NOT NULL,
            raw_text   TEXT,
            latency_ms REAL,
            tokens_in  INTEGER,
            tokens_out INTEGER,
            FOREIGN KEY (run_id)  REFERENCES runs(run_id),
            FOREIGN KEY (case_id) REFERENCES cases(case_id)
        );

        CREATE TABLE IF NOT EXISTS scores (
            output_id    TEXT PRIMARY KEY,
            score        REAL,
            label        TEXT,
            reason       TEXT,
            details_json TEXT,
            FOREIGN KEY (output_id) REFERENCES outputs(output_id)
        );

        CREATE INDEX IF NOT EXISTS idx_outputs_run_case
            ON outputs(run_id, case_id);

        CREATE INDEX IF NOT EXISTS idx_runs_model
            ON runs(model_id);
    """)

    # Migration: add tool_meta_json column if missing
    cols = {row[1] for row in conn.execute("PRAGMA table_info(outputs)").fetchall()}
    if "tool_meta_json" not in cols:
        conn.execute("ALTER TABLE outputs ADD COLUMN tool_meta_json TEXT")

    conn.commit()
    return conn


def insert_model(
    conn: sqlite3.Connection,
    *,
    model_id: str,
    name: str,
    provider: str,
    version: str | None = None,
) -> str:
    conn.execute(
        "INSERT OR IGNORE INTO models (model_id, name, provider, version) VALUES (?, ?, ?, ?)",
        (model_id, name, provider, version),
    )
    conn.commit()
    return model_id


def insert_run(
    conn: sqlite3.Connection,
    *,
    pack_id: str,
    model_id: str,
    git_sha: str | None = None,
    params: dict | None = None,
) -> str:
    run_id = uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO runs (run_id, created_at, git_sha, pack_id, model_id, params_json) VALUES (?, ?, ?, ?, ?, ?)",
        (run_id, created_at, git_sha, pack_id, model_id, json.dumps(params) if params else None),
    )
    conn.commit()
    return run_id


def insert_case(
    conn: sqlite3.Connection,
    *,
    pack_id: str,
    scheme: str | None = None,
    metadata: dict | None = None,
    expected: str | None = None,
    case_id: str | None = None,
) -> str:
    case_id = case_id or uuid4().hex
    conn.execute(
        "INSERT OR IGNORE INTO cases (case_id, pack_id, scheme, metadata_json, expected) VALUES (?, ?, ?, ?, ?)",
        (case_id, pack_id, scheme, json.dumps(metadata) if metadata else None, expected),
    )
    conn.commit()
    return case_id


def insert_output(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    case_id: str,
    raw_text: str,
    latency_ms: float,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    tool_meta: dict | None = None,
) -> str:
    output_id = uuid4().hex
    conn.execute(
        "INSERT INTO outputs (output_id, run_id, case_id, raw_text, latency_ms, tokens_in, tokens_out, tool_meta_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (output_id, run_id, case_id, raw_text, latency_ms, tokens_in, tokens_out,
         json.dumps(tool_meta) if tool_meta else None),
    )
    conn.commit()
    return output_id


def insert_score(
    conn: sqlite3.Connection,
    *,
    output_id: str,
    score: float,
    label: str | None = None,
    reason: str | None = None,
    details: dict | None = None,
) -> None:
    conn.execute(
        "INSERT INTO scores (output_id, score, label, reason, details_json) VALUES (?, ?, ?, ?, ?)",
        (output_id, score, label, reason, json.dumps(details) if details else None),
    )
    conn.commit()


def get_run_results(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT o.output_id, o.case_id, o.raw_text, o.latency_ms,
               o.tokens_in, o.tokens_out,
               s.score, s.label, s.reason, s.details_json,
               c.expected, c.scheme, c.metadata_json
        FROM outputs o
        LEFT JOIN scores s  ON s.output_id = o.output_id
        LEFT JOIN cases c   ON c.case_id   = o.case_id
        WHERE o.run_id = ?
        ORDER BY o.case_id
        """,
        (run_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_scores_by_run(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT s.output_id, s.score, s.label, s.reason, s.details_json,
               o.case_id, o.latency_ms, o.tokens_in, o.tokens_out
        FROM scores s
        JOIN outputs o ON o.output_id = s.output_id
        WHERE o.run_id = ?
        ORDER BY o.case_id
        """,
        (run_id,),
    ).fetchall()
    return [dict(r) for r in rows]
