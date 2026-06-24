import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from application.ports.memory_port import MemoryPort
from core.models import MemoryEvent, MemoryFact, MemoryReflection, SpeakerRecord


class SQLiteMemoryAdapter(MemoryPort):
    """SQLite-backed memory store plus prompt-facing markdown memory files."""

    def __init__(self, db_path: str, memory_root: str):
        self.db_path = Path(db_path)
        self.memory_root = Path(memory_root)
        self.speakers_dir = self.memory_root / "speakers"
        self.recent_context_path = self.memory_root / "recent_context.md"
        self.index_path = self.memory_root / "index.json"

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.speakers_dir.mkdir(parents=True, exist_ok=True)
        self.memory_root.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._write_index()

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
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS speakers (
                    speaker_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    source_label TEXT NOT NULL,
                    voice_embedding_uid INTEGER,
                    is_user INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_events (
                    event_id TEXT PRIMARY KEY,
                    speaker_id TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_ref TEXT NOT NULL,
                    event_kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    consolidated_at TEXT,
                    FOREIGN KEY (speaker_id) REFERENCES speakers (speaker_id)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_facts (
                    fact_id TEXT PRIMARY KEY,
                    speaker_id TEXT NOT NULL,
                    fact_text TEXT NOT NULL,
                    topic TEXT,
                    valid_from TEXT NOT NULL,
                    valid_to TEXT,
                    superseded_by TEXT,
                    source_event_ids TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (speaker_id) REFERENCES speakers (speaker_id)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_reflections (
                    reflection_id TEXT PRIMARY KEY,
                    speaker_id TEXT,
                    summary TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    source_event_ids TEXT NOT NULL,
                    FOREIGN KEY (speaker_id) REFERENCES speakers (speaker_id)
                )
                """
            )
    def _now(self) -> str:
        return datetime.now().isoformat()

    def _speaker_profile_path(self, speaker_id: str) -> Path:
        return self.speakers_dir / f"{speaker_id}.md"

    def _speaker_from_row(self, row: sqlite3.Row) -> SpeakerRecord:
        return SpeakerRecord(
            speaker_id=row["speaker_id"],
            display_name=row["display_name"],
            source_label=row["source_label"],
            voice_embedding_uid=row["voice_embedding_uid"],
            is_user=bool(row["is_user"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _event_from_row(self, row: sqlite3.Row) -> MemoryEvent:
        return MemoryEvent(
            event_id=row["event_id"],
            speaker_id=row["speaker_id"],
            source_type=row["source_type"],
            source_ref=row["source_ref"],
            event_kind=row["event_kind"],
            content=row["content"],
            confidence=row["confidence"],
            status=row["status"],
            created_at=row["created_at"],
            consolidated_at=row["consolidated_at"],
        )

    def _fact_from_row(self, row: sqlite3.Row) -> MemoryFact:
        return MemoryFact(
            fact_id=row["fact_id"],
            speaker_id=row["speaker_id"],
            fact_text=row["fact_text"],
            topic=row["topic"],
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
            superseded_by=row["superseded_by"],
            source_event_ids=json.loads(row["source_event_ids"]),
            updated_at=row["updated_at"],
        )

    def _write_index(self) -> None:
        speakers = [
            {
                "speaker_id": speaker.speaker_id,
                "display_name": speaker.display_name,
                "source_label": speaker.source_label,
                "is_user": speaker.is_user,
            }
            for speaker in self.list_speakers()
        ]
        self.index_path.write_text(json.dumps(speakers, indent=2), encoding="utf-8")

    def upsert_speaker(
        self,
        display_name: str,
        source_label: str,
        voice_embedding_uid: Optional[int] = None,
        is_user: bool = False,
        speaker_id: Optional[str] = None,
    ) -> SpeakerRecord:
        now = self._now()
        if speaker_id is None:
            existing = self.find_speaker_by_display_name(display_name)
            if existing:
                speaker_id = existing.speaker_id
            else:
                speaker_id = uuid.uuid4().hex

        with self._managed_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO speakers (
                    speaker_id, display_name, source_label, voice_embedding_uid, is_user, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(speaker_id) DO UPDATE SET
                    display_name=excluded.display_name,
                    source_label=excluded.source_label,
                    voice_embedding_uid=COALESCE(excluded.voice_embedding_uid, speakers.voice_embedding_uid),
                    is_user=excluded.is_user,
                    updated_at=excluded.updated_at
                """,
                (
                    speaker_id,
                    display_name,
                    source_label,
                    voice_embedding_uid,
                    int(is_user),
                    now,
                    now,
                ),
            )
        profile_path = self._speaker_profile_path(speaker_id)
        if not profile_path.exists():
            profile_path.write_text(
                f"# {display_name}\n\n- Source label: {source_label}\n",
                encoding="utf-8",
            )
        self._write_index()
        speaker = self.get_speaker(speaker_id)
        if speaker is None:
            raise RuntimeError(f"Failed to upsert speaker {display_name}")
        return speaker

    def get_speaker(self, speaker_id: str) -> Optional[SpeakerRecord]:
        with self._managed_connection() as conn:
            row = conn.execute(
                "SELECT * FROM speakers WHERE speaker_id = ?",
                (speaker_id,),
            ).fetchone()
        return self._speaker_from_row(row) if row else None

    def find_speaker_by_display_name(self, display_name: str) -> Optional[SpeakerRecord]:
        with self._managed_connection() as conn:
            row = conn.execute(
                "SELECT * FROM speakers WHERE lower(display_name) = lower(?) ORDER BY updated_at DESC LIMIT 1",
                (display_name,),
            ).fetchone()
        return self._speaker_from_row(row) if row else None

    def list_speakers(self) -> List[SpeakerRecord]:
        with self._managed_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM speakers ORDER BY updated_at DESC"
            ).fetchall()
        return [self._speaker_from_row(row) for row in rows]

    def append_event(self, event: MemoryEvent) -> None:
        with self._managed_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_events (
                    event_id, speaker_id, source_type, source_ref, event_kind, content,
                    confidence, status, created_at, consolidated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.speaker_id,
                    event.source_type,
                    event.source_ref,
                    event.event_kind,
                    event.content,
                    event.confidence,
                    event.status,
                    event.created_at,
                    event.consolidated_at,
                ),
            )
    def get_recent_events(
        self,
        speaker_ids: Optional[List[str]] = None,
        limit: int = 10,
        status: Optional[str] = None,
    ) -> List[MemoryEvent]:
        query = "SELECT * FROM memory_events"
        conditions: List[str] = []
        params: List[object] = []

        if speaker_ids:
            placeholders = ", ".join("?" for _ in speaker_ids)
            conditions.append(f"speaker_id IN ({placeholders})")
            params.extend(speaker_ids)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._managed_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._event_from_row(row) for row in rows]

    def get_pending_consolidation(self, limit: int = 100) -> List[MemoryEvent]:
        return self.get_recent_events(limit=limit, status="candidate")

    def upsert_fact(self, fact: MemoryFact) -> MemoryFact:
        with self._managed_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_facts (
                    fact_id, speaker_id, fact_text, topic, valid_from, valid_to,
                    superseded_by, source_event_ids, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fact.fact_id,
                    fact.speaker_id,
                    fact.fact_text,
                    fact.topic,
                    fact.valid_from,
                    fact.valid_to,
                    fact.superseded_by,
                    json.dumps(fact.source_event_ids),
                    fact.updated_at,
                ),
            )
        return fact

    def get_facts(self, speaker_id: str) -> List[MemoryFact]:
        with self._managed_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM memory_facts
                WHERE speaker_id = ?
                ORDER BY updated_at DESC
                """,
                (speaker_id,),
            ).fetchall()
        return [self._fact_from_row(row) for row in rows]

    def mark_events_consolidated(self, event_ids: List[str], consolidated_at: str) -> None:
        if not event_ids:
            return
        placeholders = ", ".join("?" for _ in event_ids)
        with self._managed_connection() as conn:
            conn.execute(
                f"""
                UPDATE memory_events
                SET status = 'consolidated', consolidated_at = ?
                WHERE event_id IN ({placeholders})
                """,
                [consolidated_at, *event_ids],
            )
    def add_reflection(self, reflection: MemoryReflection) -> None:
        with self._managed_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_reflections (
                    reflection_id, speaker_id, summary, created_at, source_event_ids
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    reflection.reflection_id,
                    reflection.speaker_id,
                    reflection.summary,
                    reflection.created_at,
                    json.dumps(reflection.source_event_ids),
                ),
            )
    def get_speaker_profile(self, speaker_id: str) -> str:
        path = self._speaker_profile_path(speaker_id)
        if not path.exists():
            speaker = self.get_speaker(speaker_id)
            if speaker is None:
                return ""
            path.write_text(
                f"# {speaker.display_name}\n\n_No curated memory yet._\n",
                encoding="utf-8",
            )
        return path.read_text(encoding="utf-8")

    def save_speaker_profile(self, speaker_id: str, content: str) -> None:
        self._speaker_profile_path(speaker_id).write_text(content, encoding="utf-8")

    def get_recent_context(self) -> str:
        if not self.recent_context_path.exists():
            return ""
        return self.recent_context_path.read_text(encoding="utf-8")

    def save_recent_context(self, content: str) -> None:
        self.recent_context_path.write_text(content, encoding="utf-8")
