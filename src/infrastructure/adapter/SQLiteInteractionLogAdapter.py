import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

from core.models import InteractionLogEntry


class SQLiteInteractionLogAdapter:
    """SQLite-backed audit log for all LLM interactions."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _managed_connection(self):
        conn = self._connection()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._managed_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS interaction_logs (
                    interaction_id TEXT PRIMARY KEY,
                    interaction_run_id TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    source TEXT NOT NULL,
                    model TEXT NOT NULL,
                    messages_json TEXT NOT NULL,
                    tools_json TEXT,
                    image_path TEXT,
                    response_text TEXT,
                    reasoning_text TEXT,
                    tool_calls_json TEXT,
                    error_text TEXT,
                    duration_ms INTEGER,
                    metadata_json TEXT,
                    report_json TEXT
                )
                """
            )
            self._ensure_columns(conn)
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_interaction_logs_created_at
                ON interaction_logs (created_at DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_interaction_logs_source
                ON interaction_logs (source, created_at DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_interaction_logs_run_id
                ON interaction_logs (interaction_run_id, created_at DESC)
                """
            )

    def _ensure_columns(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute("PRAGMA table_info(interaction_logs)").fetchall()
        existing = {row["name"] for row in rows}
        if "interaction_run_id" not in existing:
            conn.execute("ALTER TABLE interaction_logs ADD COLUMN interaction_run_id TEXT")
        if "report_json" not in existing:
            conn.execute("ALTER TABLE interaction_logs ADD COLUMN report_json TEXT")

    def insert(self, entry: InteractionLogEntry) -> None:
        with self._managed_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO interaction_logs (
                    interaction_id, interaction_run_id, created_at, completed_at, source, model, messages_json,
                    tools_json, image_path, response_text, reasoning_text, tool_calls_json,
                    error_text, duration_ms, metadata_json, report_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.interaction_id,
                    entry.interaction_run_id,
                    entry.created_at,
                    entry.completed_at,
                    entry.source,
                    entry.model,
                    entry.messages_json,
                    entry.tools_json,
                    entry.image_path,
                    entry.response_text,
                    entry.reasoning_text,
                    entry.tool_calls_json,
                    entry.error_text,
                    entry.duration_ms,
                    entry.metadata_json,
                    entry.report_json,
                ),
            )

    def attach_report(self, interaction_run_id: str, report_json: str) -> None:
        with self._managed_connection() as conn:
            conn.execute(
                """
                UPDATE interaction_logs
                SET report_json = ?
                WHERE interaction_id = (
                    SELECT interaction_id
                    FROM interaction_logs
                    WHERE interaction_run_id = ?
                    ORDER BY created_at DESC, rowid DESC
                    LIMIT 1
                )
                """,
                (report_json, interaction_run_id),
            )

    def list_recent(self, limit: int = 50, source: Optional[str] = None) -> List[InteractionLogEntry]:
        query = "SELECT * FROM interaction_logs"
        params: List[object] = []
        if source:
            query += " WHERE source = ?"
            params.append(source)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._managed_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._from_row(row) for row in rows]

    def list_entries(
        self,
        *,
        limit: int = 500,
        offset: int = 0,
        source: Optional[str] = None,
    ) -> List[InteractionLogEntry]:
        query = "SELECT * FROM interaction_logs"
        params: List[object] = []
        if source:
            query += " WHERE source = ?"
            params.append(source)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with self._managed_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._from_row(row) for row in rows]

    def get_by_interaction_id(self, interaction_id: str) -> Optional[InteractionLogEntry]:
        with self._managed_connection() as conn:
            row = conn.execute(
                "SELECT * FROM interaction_logs WHERE interaction_id = ?",
                (interaction_id,),
            ).fetchone()
        return self._from_row(row) if row is not None else None

    def list_recent_reports(self, limit: int = 50) -> List[InteractionLogEntry]:
        with self._managed_connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM interaction_logs
                WHERE report_json IS NOT NULL AND TRIM(report_json) <> ''
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def _from_row(self, row: sqlite3.Row) -> InteractionLogEntry:
        return InteractionLogEntry(
            interaction_id=row["interaction_id"],
            interaction_run_id=row["interaction_run_id"],
            created_at=row["created_at"],
            completed_at=row["completed_at"],
            source=row["source"],
            model=row["model"],
            messages_json=row["messages_json"],
            tools_json=row["tools_json"],
            image_path=row["image_path"],
            response_text=row["response_text"],
            reasoning_text=row["reasoning_text"],
            tool_calls_json=row["tool_calls_json"],
            error_text=row["error_text"],
            duration_ms=row["duration_ms"],
            metadata_json=row["metadata_json"],
            report_json=row["report_json"],
        )
