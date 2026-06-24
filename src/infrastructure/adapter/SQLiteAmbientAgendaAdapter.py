import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from application.ports.ambient_agenda_port import AmbientAgendaPort
from core.models import AmbientAgendaItem


class SQLiteAmbientAgendaAdapter(AmbientAgendaPort):
    """SQLite-backed store for bounded ambient agenda items."""

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
                CREATE TABLE IF NOT EXISTS ambient_agenda (
                    agenda_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_ref TEXT NOT NULL,
                    priority_score REAL NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    due_at TEXT,
                    last_considered_at TEXT,
                    backing_topic_id TEXT,
                    backing_memory_ids TEXT NOT NULL,
                    surface_message TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ambient_agenda_status_priority
                ON ambient_agenda (status, priority_score DESC, updated_at DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ambient_agenda_source
                ON ambient_agenda (source_type, source_ref, kind)
                """
            )

    def _from_row(self, row: sqlite3.Row) -> AmbientAgendaItem:
        return AmbientAgendaItem(
            agenda_id=row["agenda_id"],
            title=row["title"],
            kind=row["kind"],
            source_type=row["source_type"],
            source_ref=row["source_ref"],
            priority_score=float(row["priority_score"]),
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            due_at=row["due_at"],
            last_considered_at=row["last_considered_at"],
            backing_topic_id=row["backing_topic_id"],
            backing_memory_ids=json.loads(row["backing_memory_ids"]),
            surface_message=row["surface_message"],
        )

    def create_item(self, item: AmbientAgendaItem) -> AmbientAgendaItem:
        with self._managed_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO ambient_agenda (
                    agenda_id, title, kind, source_type, source_ref, priority_score,
                    status, created_at, updated_at, due_at, last_considered_at,
                    backing_topic_id, backing_memory_ids, surface_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.agenda_id,
                    item.title,
                    item.kind,
                    item.source_type,
                    item.source_ref,
                    item.priority_score,
                    item.status,
                    item.created_at,
                    item.updated_at,
                    item.due_at,
                    item.last_considered_at,
                    item.backing_topic_id,
                    json.dumps(item.backing_memory_ids),
                    item.surface_message,
                ),
            )
        return item

    def update_item(self, item: AmbientAgendaItem) -> AmbientAgendaItem:
        return self.create_item(item)

    def get_item(self, agenda_id: str) -> Optional[AmbientAgendaItem]:
        with self._managed_connection() as conn:
            row = conn.execute(
                "SELECT * FROM ambient_agenda WHERE agenda_id = ?",
                (agenda_id,),
            ).fetchone()
        return self._from_row(row) if row else None

    def list_items(
        self,
        statuses: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[AmbientAgendaItem]:
        query = "SELECT * FROM ambient_agenda"
        params: List[object] = []
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            query += f" WHERE status IN ({placeholders})"
            params.extend(statuses)
        query += " ORDER BY priority_score DESC, updated_at DESC LIMIT ?"
        params.append(limit)
        with self._managed_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._from_row(row) for row in rows]

    def find_by_source(
        self,
        source_type: str,
        source_ref: str,
        kind: Optional[str] = None,
        statuses: Optional[List[str]] = None,
    ) -> Optional[AmbientAgendaItem]:
        query = (
            "SELECT * FROM ambient_agenda WHERE source_type = ? AND source_ref = ?"
        )
        params: List[object] = [source_type, source_ref]
        if kind is not None:
            query += " AND kind = ?"
            params.append(kind)
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            query += f" AND status IN ({placeholders})"
            params.extend(statuses)
        query += " ORDER BY updated_at DESC LIMIT 1"
        with self._managed_connection() as conn:
            row = conn.execute(query, params).fetchone()
        return self._from_row(row) if row else None
