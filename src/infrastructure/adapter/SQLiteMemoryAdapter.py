import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from application.ports.memory_port import MemoryPort
from core.models import (
    ConversationSession,
    MemoryEvent,
    MemoryFact,
    MemoryReflection,
    OpenLoop,
    SpeakerRecord,
    TranscriptEvidence,
    UserProfileFacet,
    VisualObservation,
    VisualUserFact,
    VisualSession,
)


class SQLiteMemoryAdapter(MemoryPort):
    """SQLite-backed memory store plus prompt-facing markdown memory files."""

    def __init__(self, db_path: str, memory_root: str):
        self.db_path = Path(db_path)
        self.memory_root = Path(memory_root)
        self.speakers_dir = self.memory_root / "speakers"
        self.recent_context_path = self.memory_root / "recent_context.md"
        self.session_digest_path = self.memory_root / "session_digest.md"
        self.open_loop_digest_path = self.memory_root / "open_loops.md"
        self.visual_digest_path = self.memory_root / "visual_context.md"
        self.user_info_path = self.memory_root.parent / "USER_INFO.md"
        self.index_path = self.memory_root / "index.json"

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.speakers_dir.mkdir(parents=True, exist_ok=True)
        self.memory_root.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._ensure_visual_observation_columns()
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
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS transcript_evidence (
                    evidence_id TEXT PRIMARY KEY,
                    source_ref TEXT NOT NULL,
                    speaker_id TEXT NOT NULL,
                    speaker_label TEXT NOT NULL,
                    session_id TEXT,
                    signal_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    normalized_entities TEXT NOT NULL,
                    time_hints TEXT NOT NULL,
                    action_hints TEXT NOT NULL,
                    trust_score REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (speaker_id) REFERENCES speakers (speaker_id)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_sessions (
                    session_id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    ended_at TEXT NOT NULL,
                    participant_ids TEXT NOT NULL,
                    status TEXT NOT NULL,
                    topic_summary TEXT NOT NULL,
                    entity_summary TEXT NOT NULL,
                    recent_turn_summary TEXT NOT NULL,
                    last_activity_at TEXT NOT NULL,
                    continuation_score REAL NOT NULL,
                    derived_loop_ids TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS open_loops (
                    loop_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    loop_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    owner_speaker_id TEXT NOT NULL,
                    source_session_id TEXT NOT NULL,
                    supporting_event_ids TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    urgency REAL NOT NULL,
                    due_hint TEXT,
                    next_action_hint TEXT,
                    last_updated_at TEXT NOT NULL,
                    resolution_summary TEXT,
                    FOREIGN KEY (owner_speaker_id) REFERENCES speakers (speaker_id)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profile_facets (
                    facet_id TEXT PRIMARY KEY,
                    speaker_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    strength INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    source_event_ids TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (speaker_id) REFERENCES speakers (speaker_id)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS visual_observations (
                    observation_id TEXT PRIMARY KEY,
                    screenshot_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    observation_type TEXT NOT NULL,
                    app_name TEXT,
                    window_title TEXT,
                    page_hint TEXT,
                    summary TEXT NOT NULL,
                    inferred_user_activity TEXT NOT NULL,
                    previous_activity_status TEXT NOT NULL DEFAULT 'unclear',
                    salient_entities TEXT NOT NULL,
                    completed_items TEXT NOT NULL DEFAULT '[]',
                    open_loops TEXT NOT NULL,
                    possible_next_task TEXT,
                    suggested_research_topics TEXT NOT NULL,
                    user_fact_hypotheses TEXT NOT NULL DEFAULT '[]',
                    confidence REAL NOT NULL,
                    session_id TEXT,
                    raw_payload_json TEXT
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS visual_sessions (
                    session_id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    ended_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    activity_summary TEXT NOT NULL,
                    app_name TEXT,
                    window_title TEXT,
                    page_hint TEXT,
                    last_activity_at TEXT NOT NULL,
                    continuation_score REAL NOT NULL,
                    observation_ids TEXT NOT NULL,
                    related_loop_ids TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS visual_user_facts (
                    fact_id TEXT PRIMARY KEY,
                    fact_key TEXT NOT NULL UNIQUE,
                    category TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    status TEXT NOT NULL,
                    score REAL NOT NULL,
                    observation_count INTEGER NOT NULL,
                    session_count INTEGER NOT NULL,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    source_observation_ids TEXT NOT NULL,
                    source_session_ids TEXT NOT NULL
                )
                """
            )

    def _ensure_visual_observation_columns(self) -> None:
        required_columns = {
            "previous_activity_status": "TEXT NOT NULL DEFAULT 'unclear'",
            "completed_items": "TEXT NOT NULL DEFAULT '[]'",
            "possible_next_task": "TEXT",
            "user_fact_hypotheses": "TEXT NOT NULL DEFAULT '[]'",
        }
        with self._managed_connection() as conn:
            existing = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(visual_observations)").fetchall()
            }
            for column_name, column_def in required_columns.items():
                if column_name in existing:
                    continue
                conn.execute(
                    f"ALTER TABLE visual_observations ADD COLUMN {column_name} {column_def}"
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

    def _evidence_from_row(self, row: sqlite3.Row) -> TranscriptEvidence:
        return TranscriptEvidence(
            evidence_id=row["evidence_id"],
            source_ref=row["source_ref"],
            speaker_id=row["speaker_id"],
            speaker_label=row["speaker_label"],
            session_id=row["session_id"],
            signal_type=row["signal_type"],
            content=row["content"],
            normalized_entities=json.loads(row["normalized_entities"]),
            time_hints=json.loads(row["time_hints"]),
            action_hints=json.loads(row["action_hints"]),
            trust_score=float(row["trust_score"]),
            created_at=row["created_at"],
        )

    def _session_from_row(self, row: sqlite3.Row) -> ConversationSession:
        return ConversationSession(
            session_id=row["session_id"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            participant_ids=json.loads(row["participant_ids"]),
            status=row["status"],
            topic_summary=row["topic_summary"],
            entity_summary=row["entity_summary"],
            recent_turn_summary=row["recent_turn_summary"],
            last_activity_at=row["last_activity_at"],
            continuation_score=float(row["continuation_score"]),
            derived_loop_ids=json.loads(row["derived_loop_ids"]),
        )

    def _loop_from_row(self, row: sqlite3.Row) -> OpenLoop:
        return OpenLoop(
            loop_id=row["loop_id"],
            title=row["title"],
            loop_type=row["loop_type"],
            status=row["status"],
            owner_speaker_id=row["owner_speaker_id"],
            source_session_id=row["source_session_id"],
            supporting_event_ids=json.loads(row["supporting_event_ids"]),
            confidence=float(row["confidence"]),
            urgency=float(row["urgency"]),
            due_hint=row["due_hint"],
            next_action_hint=row["next_action_hint"],
            last_updated_at=row["last_updated_at"],
            resolution_summary=row["resolution_summary"],
        )

    def _facet_from_row(self, row: sqlite3.Row) -> UserProfileFacet:
        return UserProfileFacet(
            facet_id=row["facet_id"],
            speaker_id=row["speaker_id"],
            category=row["category"],
            title=row["title"],
            summary=row["summary"],
            confidence=float(row["confidence"]),
            strength=int(row["strength"]),
            status=row["status"],
            source_event_ids=json.loads(row["source_event_ids"]),
            updated_at=row["updated_at"],
        )

    def _visual_observation_from_row(self, row: sqlite3.Row) -> VisualObservation:
        return VisualObservation(
            observation_id=row["observation_id"],
            screenshot_path=row["screenshot_path"],
            created_at=row["created_at"],
            observation_type=row["observation_type"],
            app_name=row["app_name"],
            window_title=row["window_title"],
            page_hint=row["page_hint"],
            summary=row["summary"],
            inferred_user_activity=row["inferred_user_activity"],
            previous_activity_status=row["previous_activity_status"],
            salient_entities=json.loads(row["salient_entities"]),
            completed_items=json.loads(row["completed_items"]),
            open_loops=json.loads(row["open_loops"]),
            possible_next_task=row["possible_next_task"],
            suggested_research_topics=json.loads(row["suggested_research_topics"]),
            user_fact_hypotheses=json.loads(row["user_fact_hypotheses"]),
            confidence=float(row["confidence"]),
            session_id=row["session_id"],
            raw_payload_json=row["raw_payload_json"],
        )

    def _visual_user_fact_from_row(self, row: sqlite3.Row) -> VisualUserFact:
        return VisualUserFact(
            fact_id=row["fact_id"],
            fact_key=row["fact_key"],
            category=row["category"],
            title=row["title"],
            summary=row["summary"],
            status=row["status"],
            score=float(row["score"]),
            observation_count=int(row["observation_count"]),
            session_count=int(row["session_count"]),
            first_seen_at=row["first_seen_at"],
            last_seen_at=row["last_seen_at"],
            source_observation_ids=json.loads(row["source_observation_ids"]),
            source_session_ids=json.loads(row["source_session_ids"]),
        )

    def _visual_session_from_row(self, row: sqlite3.Row) -> VisualSession:
        return VisualSession(
            session_id=row["session_id"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            status=row["status"],
            activity_summary=row["activity_summary"],
            app_name=row["app_name"],
            window_title=row["window_title"],
            page_hint=row["page_hint"],
            last_activity_at=row["last_activity_at"],
            continuation_score=float(row["continuation_score"]),
            observation_ids=json.loads(row["observation_ids"]),
            related_loop_ids=json.loads(row["related_loop_ids"]),
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

    def append_evidence(self, evidence: TranscriptEvidence) -> None:
        with self._managed_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO transcript_evidence (
                    evidence_id, source_ref, speaker_id, speaker_label, session_id,
                    signal_type, content, normalized_entities, time_hints, action_hints,
                    trust_score, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evidence.evidence_id,
                    evidence.source_ref,
                    evidence.speaker_id,
                    evidence.speaker_label,
                    evidence.session_id,
                    evidence.signal_type,
                    evidence.content,
                    json.dumps(evidence.normalized_entities),
                    json.dumps(evidence.time_hints),
                    json.dumps(evidence.action_hints),
                    evidence.trust_score,
                    evidence.created_at,
                ),
            )

    def get_recent_evidence(
        self,
        speaker_ids: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[TranscriptEvidence]:
        query = "SELECT * FROM transcript_evidence"
        params: List[object] = []
        if speaker_ids:
            placeholders = ", ".join("?" for _ in speaker_ids)
            query += f" WHERE speaker_id IN ({placeholders})"
            params.extend(speaker_ids)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._managed_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._evidence_from_row(row) for row in rows]

    def upsert_session(self, session: ConversationSession) -> ConversationSession:
        with self._managed_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO conversation_sessions (
                    session_id, started_at, ended_at, participant_ids, status,
                    topic_summary, entity_summary, recent_turn_summary, last_activity_at,
                    continuation_score, derived_loop_ids
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.session_id,
                    session.started_at,
                    session.ended_at,
                    json.dumps(session.participant_ids),
                    session.status,
                    session.topic_summary,
                    session.entity_summary,
                    session.recent_turn_summary,
                    session.last_activity_at,
                    session.continuation_score,
                    json.dumps(session.derived_loop_ids),
                ),
            )
        return session

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        with self._managed_connection() as conn:
            row = conn.execute(
                "SELECT * FROM conversation_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return self._session_from_row(row) if row else None

    def list_sessions(self, statuses: Optional[List[str]] = None, limit: int = 20) -> List[ConversationSession]:
        query = "SELECT * FROM conversation_sessions"
        params: List[object] = []
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            query += f" WHERE status IN ({placeholders})"
            params.extend(statuses)
        query += " ORDER BY last_activity_at DESC LIMIT ?"
        params.append(limit)
        with self._managed_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._session_from_row(row) for row in rows]

    def upsert_open_loop(self, loop: OpenLoop) -> OpenLoop:
        with self._managed_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO open_loops (
                    loop_id, title, loop_type, status, owner_speaker_id, source_session_id,
                    supporting_event_ids, confidence, urgency, due_hint, next_action_hint,
                    last_updated_at, resolution_summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    loop.loop_id,
                    loop.title,
                    loop.loop_type,
                    loop.status,
                    loop.owner_speaker_id,
                    loop.source_session_id,
                    json.dumps(loop.supporting_event_ids),
                    loop.confidence,
                    loop.urgency,
                    loop.due_hint,
                    loop.next_action_hint,
                    loop.last_updated_at,
                    loop.resolution_summary,
                ),
            )
        return loop

    def get_open_loop(self, loop_id: str) -> Optional[OpenLoop]:
        with self._managed_connection() as conn:
            row = conn.execute(
                "SELECT * FROM open_loops WHERE loop_id = ?",
                (loop_id,),
            ).fetchone()
        return self._loop_from_row(row) if row else None

    def list_open_loops(self, statuses: Optional[List[str]] = None, limit: int = 20) -> List[OpenLoop]:
        query = "SELECT * FROM open_loops"
        params: List[object] = []
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            query += f" WHERE status IN ({placeholders})"
            params.extend(statuses)
        query += " ORDER BY last_updated_at DESC, urgency DESC LIMIT ?"
        params.append(limit)
        with self._managed_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._loop_from_row(row) for row in rows]

    def upsert_profile_facet(self, facet: UserProfileFacet) -> UserProfileFacet:
        with self._managed_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO user_profile_facets (
                    facet_id, speaker_id, category, title, summary, confidence,
                    strength, status, source_event_ids, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    facet.facet_id,
                    facet.speaker_id,
                    facet.category,
                    facet.title,
                    facet.summary,
                    facet.confidence,
                    facet.strength,
                    facet.status,
                    json.dumps(facet.source_event_ids),
                    facet.updated_at,
                ),
            )
        return facet

    def get_profile_facets(
        self,
        speaker_id: str,
        categories: Optional[List[str]] = None,
        statuses: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[UserProfileFacet]:
        query = "SELECT * FROM user_profile_facets WHERE speaker_id = ?"
        params: List[object] = [speaker_id]
        if categories:
            placeholders = ", ".join("?" for _ in categories)
            query += f" AND category IN ({placeholders})"
            params.extend(categories)
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            query += f" AND status IN ({placeholders})"
            params.extend(statuses)
        query += " ORDER BY strength DESC, updated_at DESC LIMIT ?"
        params.append(limit)
        with self._managed_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._facet_from_row(row) for row in rows]

    def save_session_digest(self, content: str) -> None:
        self.session_digest_path.write_text(content, encoding="utf-8")

    def get_session_digest(self) -> str:
        if not self.session_digest_path.exists():
            return ""
        return self.session_digest_path.read_text(encoding="utf-8")

    def save_open_loop_digest(self, content: str) -> None:
        self.open_loop_digest_path.write_text(content, encoding="utf-8")

    def get_open_loop_digest(self) -> str:
        if not self.open_loop_digest_path.exists():
            return ""
        return self.open_loop_digest_path.read_text(encoding="utf-8")

    def save_user_info(self, content: str) -> None:
        self.user_info_path.write_text(content, encoding="utf-8")

    def get_user_info(self) -> str:
        if not self.user_info_path.exists():
            return ""
        return self.user_info_path.read_text(encoding="utf-8")

    def append_visual_observation(self, observation: VisualObservation) -> None:
        with self._managed_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO visual_observations (
                    observation_id, screenshot_path, created_at, observation_type,
                    app_name, window_title, page_hint, summary, inferred_user_activity,
                    previous_activity_status, salient_entities, completed_items, open_loops,
                    possible_next_task, suggested_research_topics, user_fact_hypotheses,
                    confidence, session_id, raw_payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    observation.observation_id,
                    observation.screenshot_path,
                    observation.created_at,
                    observation.observation_type,
                    observation.app_name,
                    observation.window_title,
                    observation.page_hint,
                    observation.summary,
                    observation.inferred_user_activity,
                    observation.previous_activity_status,
                    json.dumps(observation.salient_entities),
                    json.dumps(observation.completed_items),
                    json.dumps(observation.open_loops),
                    observation.possible_next_task,
                    json.dumps(observation.suggested_research_topics),
                    json.dumps(observation.user_fact_hypotheses),
                    observation.confidence,
                    observation.session_id,
                    observation.raw_payload_json,
                ),
            )

    def get_recent_visual_observations(self, limit: int = 10) -> List[VisualObservation]:
        with self._managed_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM visual_observations ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._visual_observation_from_row(row) for row in rows]

    def upsert_visual_session(self, session: VisualSession) -> VisualSession:
        with self._managed_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO visual_sessions (
                    session_id, started_at, ended_at, status, activity_summary,
                    app_name, window_title, page_hint, last_activity_at,
                    continuation_score, observation_ids, related_loop_ids
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.session_id,
                    session.started_at,
                    session.ended_at,
                    session.status,
                    session.activity_summary,
                    session.app_name,
                    session.window_title,
                    session.page_hint,
                    session.last_activity_at,
                    session.continuation_score,
                    json.dumps(session.observation_ids),
                    json.dumps(session.related_loop_ids),
                ),
            )
        return session

    def get_visual_session(self, session_id: str) -> Optional[VisualSession]:
        with self._managed_connection() as conn:
            row = conn.execute(
                "SELECT * FROM visual_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return self._visual_session_from_row(row) if row else None

    def list_visual_sessions(self, statuses: Optional[List[str]] = None, limit: int = 20) -> List[VisualSession]:
        query = "SELECT * FROM visual_sessions"
        params: List[object] = []
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            query += f" WHERE status IN ({placeholders})"
            params.extend(statuses)
        query += " ORDER BY last_activity_at DESC LIMIT ?"
        params.append(limit)
        with self._managed_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._visual_session_from_row(row) for row in rows]

    def upsert_visual_user_fact(self, fact: VisualUserFact) -> VisualUserFact:
        with self._managed_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO visual_user_facts (
                    fact_id, fact_key, category, title, summary, status, score,
                    observation_count, session_count, first_seen_at, last_seen_at,
                    source_observation_ids, source_session_ids
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fact.fact_id,
                    fact.fact_key,
                    fact.category,
                    fact.title,
                    fact.summary,
                    fact.status,
                    fact.score,
                    fact.observation_count,
                    fact.session_count,
                    fact.first_seen_at,
                    fact.last_seen_at,
                    json.dumps(fact.source_observation_ids),
                    json.dumps(fact.source_session_ids),
                ),
            )
        return fact

    def get_visual_user_fact(self, fact_key: str) -> Optional[VisualUserFact]:
        with self._managed_connection() as conn:
            row = conn.execute(
                "SELECT * FROM visual_user_facts WHERE fact_key = ?",
                (fact_key,),
            ).fetchone()
        return self._visual_user_fact_from_row(row) if row else None

    def list_visual_user_facts(
        self,
        statuses: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[VisualUserFact]:
        query = "SELECT * FROM visual_user_facts"
        conditions: List[str] = []
        params: List[object] = []
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            conditions.append(f"status IN ({placeholders})")
            params.extend(statuses)
        if categories:
            placeholders = ", ".join("?" for _ in categories)
            conditions.append(f"category IN ({placeholders})")
            params.extend(categories)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY score DESC, last_seen_at DESC LIMIT ?"
        params.append(limit)
        with self._managed_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._visual_user_fact_from_row(row) for row in rows]

    def save_visual_digest(self, content: str) -> None:
        self.visual_digest_path.write_text(content, encoding="utf-8")

    def get_visual_digest(self) -> str:
        if not self.visual_digest_path.exists():
            return ""
        return self.visual_digest_path.read_text(encoding="utf-8")
