import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from application.ports.proactive_topic_queue_port import ProactiveTopicQueuePort
from core.models import ProactiveTopicCandidate


class SQLiteProactiveTopicQueueAdapter(ProactiveTopicQueuePort):
    """SQLite-backed queue for proactive research topics."""

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
                CREATE TABLE IF NOT EXISTS proactive_topics (
                    topic_id TEXT PRIMARY KEY,
                    normalized_topic TEXT NOT NULL UNIQUE,
                    display_title TEXT NOT NULL,
                    source_ref TEXT NOT NULL,
                    speaker_label TEXT NOT NULL,
                    summary_hint TEXT NOT NULL,
                    salience_score REAL NOT NULL,
                    status TEXT NOT NULL,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    artifact_path TEXT,
                    last_researched_at TEXT
                )
                """
            )

    def _from_row(self, row: sqlite3.Row) -> ProactiveTopicCandidate:
        return ProactiveTopicCandidate(
            topic_id=row["topic_id"],
            normalized_topic=row["normalized_topic"],
            display_title=row["display_title"],
            source_ref=row["source_ref"],
            speaker_label=row["speaker_label"],
            summary_hint=row["summary_hint"],
            salience_score=row["salience_score"],
            status=row["status"],
            first_seen_at=row["first_seen_at"],
            last_seen_at=row["last_seen_at"],
            artifact_path=row["artifact_path"],
            last_researched_at=row["last_researched_at"],
        )

    def upsert_topic(self, topic: ProactiveTopicCandidate) -> ProactiveTopicCandidate:
        existing = self.find_by_normalized_topic(topic.normalized_topic)
        now = datetime.now().isoformat()
        if existing is not None:
            merged = ProactiveTopicCandidate(
                topic_id=existing.topic_id,
                normalized_topic=existing.normalized_topic,
                display_title=topic.display_title or existing.display_title,
                source_ref=topic.source_ref,
                speaker_label=topic.speaker_label,
                summary_hint=topic.summary_hint,
                salience_score=max(existing.salience_score, topic.salience_score),
                status="pending",
                first_seen_at=existing.first_seen_at,
                last_seen_at=now,
                artifact_path=existing.artifact_path,
                last_researched_at=existing.last_researched_at,
            )
        else:
            merged = ProactiveTopicCandidate(
                topic_id=topic.topic_id,
                normalized_topic=topic.normalized_topic,
                display_title=topic.display_title,
                source_ref=topic.source_ref,
                speaker_label=topic.speaker_label,
                summary_hint=topic.summary_hint,
                salience_score=topic.salience_score,
                status=topic.status,
                first_seen_at=topic.first_seen_at or now,
                last_seen_at=topic.last_seen_at or now,
                artifact_path=topic.artifact_path,
                last_researched_at=topic.last_researched_at,
            )

        with self._managed_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO proactive_topics (
                    topic_id, normalized_topic, display_title, source_ref, speaker_label,
                    summary_hint, salience_score, status, first_seen_at, last_seen_at,
                    artifact_path, last_researched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    merged.topic_id,
                    merged.normalized_topic,
                    merged.display_title,
                    merged.source_ref,
                    merged.speaker_label,
                    merged.summary_hint,
                    merged.salience_score,
                    merged.status,
                    merged.first_seen_at,
                    merged.last_seen_at,
                    merged.artifact_path,
                    merged.last_researched_at,
                ),
            )
        return merged

    def get_topic(self, topic_id: str) -> Optional[ProactiveTopicCandidate]:
        with self._managed_connection() as conn:
            row = conn.execute(
                "SELECT * FROM proactive_topics WHERE topic_id = ?",
                (topic_id,),
            ).fetchone()
        return self._from_row(row) if row else None

    def find_by_normalized_topic(self, normalized_topic: str) -> Optional[ProactiveTopicCandidate]:
        with self._managed_connection() as conn:
            row = conn.execute(
                "SELECT * FROM proactive_topics WHERE normalized_topic = ?",
                (normalized_topic,),
            ).fetchone()
        return self._from_row(row) if row else None

    def get_pending_topics(self, limit: int = 20) -> List[ProactiveTopicCandidate]:
        return self.list_topics(statuses=["pending"], limit=limit)

    def list_topics(
        self,
        statuses: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[ProactiveTopicCandidate]:
        query = "SELECT * FROM proactive_topics"
        params: List[object] = []
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            query += f" WHERE status IN ({placeholders})"
            params.extend(statuses)
        query += " ORDER BY salience_score DESC, last_seen_at DESC LIMIT ?"
        params.append(limit)
        with self._managed_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._from_row(row) for row in rows]

    def mark_topic_status(
        self,
        topic_id: str,
        status: str,
        artifact_path: Optional[str] = None,
        last_researched_at: Optional[str] = None,
    ) -> None:
        with self._managed_connection() as conn:
            conn.execute(
                """
                UPDATE proactive_topics
                SET status = ?, artifact_path = COALESCE(?, artifact_path), last_researched_at = COALESCE(?, last_researched_at)
                WHERE topic_id = ?
                """,
                (status, artifact_path, last_researched_at, topic_id),
            )
