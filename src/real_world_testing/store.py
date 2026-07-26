from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


TERMINAL_STATUSES = {"completed", "completed_with_errors", "failed", "cancelled", "interrupted"}


@dataclass(frozen=True)
class TraceEvent:
    event_id: str
    run_id: str
    result_id: str | None
    sequence: int
    created_at: str
    stage: str
    event_type: str
    status: str
    model: str | None
    duration_ms: int | None
    span_id: str | None
    parent_span_id: str | None
    payload: dict[str, Any]


class SQLiteRealWorldTestStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()
        self.recover_interrupted()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS real_world_runs (
                    run_id TEXT PRIMARY KEY, suite_id TEXT, created_at TEXT NOT NULL,
                    started_at TEXT, completed_at TEXT, status TEXT NOT NULL,
                    playback_speed REAL NOT NULL, scenario_ids_json TEXT NOT NULL,
                    model_roles_json TEXT NOT NULL, config_json TEXT, error_text TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS real_world_results (
                    result_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, scenario_id TEXT NOT NULL,
                    title TEXT NOT NULL, modality TEXT NOT NULL, created_at TEXT NOT NULL,
                    completed_at TEXT, status TEXT NOT NULL, rubric_notes TEXT,
                    media_json TEXT, transcript_text TEXT, final_response TEXT,
                    summary_json TEXT, error_text TEXT,
                    FOREIGN KEY(run_id) REFERENCES real_world_runs(run_id)
                );
                CREATE TABLE IF NOT EXISTS real_world_events (
                    event_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, result_id TEXT,
                    sequence INTEGER NOT NULL, created_at TEXT NOT NULL, stage TEXT NOT NULL,
                    event_type TEXT NOT NULL, status TEXT NOT NULL, model TEXT,
                    duration_ms INTEGER, span_id TEXT, parent_span_id TEXT, payload_json TEXT NOT NULL,
                    UNIQUE(run_id, sequence)
                );
                CREATE TABLE IF NOT EXISTS real_world_reviews (
                    review_id TEXT PRIMARY KEY, result_id TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, reviewer TEXT NOT NULL,
                    perception_score INTEGER, decision_score INTEGER, tool_choice_score INTEGER,
                    tool_execution_score INTEGER, final_response_score INTEGER,
                    overall_score INTEGER, notes TEXT
                );
                CREATE TABLE IF NOT EXISTS real_world_media (
                    media_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, kind TEXT NOT NULL,
                    original_name TEXT NOT NULL, path TEXT NOT NULL, size_bytes INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_rw_results_run ON real_world_results(run_id);
                CREATE INDEX IF NOT EXISTS idx_rw_events_run_sequence ON real_world_events(run_id, sequence);
                """
            )

    def recover_interrupted(self) -> None:
        now = datetime.now().isoformat()
        with self._connection() as conn:
            conn.execute(
                "UPDATE real_world_runs SET status='interrupted', completed_at=? WHERE status IN ('queued','running')",
                (now,),
            )
            conn.execute(
                "UPDATE real_world_results SET status='interrupted', completed_at=? WHERE status IN ('queued','running')",
                (now,),
            )

    def create_run(self, *, suite_id: str, scenario_ids: list[str], playback_speed: float,
                   model_roles: dict[str, str], config: dict[str, Any]) -> str:
        run_id = uuid.uuid4().hex
        with self._connection() as conn:
            conn.execute(
                "INSERT INTO real_world_runs VALUES (?, ?, ?, NULL, NULL, 'queued', ?, ?, ?, ?, NULL, 0)",
                (run_id, suite_id, datetime.now().isoformat(), playback_speed,
                 json.dumps(scenario_ids), json.dumps(model_roles), json.dumps(config)),
            )
        return run_id

    def update_run(self, run_id: str, status: str, *, error_text: str | None = None) -> None:
        now = datetime.now().isoformat()
        started = now if status == "running" else None
        completed = now if status in TERMINAL_STATUSES else None
        with self._connection() as conn:
            conn.execute(
                "UPDATE real_world_runs SET status=?, started_at=COALESCE(started_at, ?), "
                "completed_at=COALESCE(?, completed_at), error_text=COALESCE(?, error_text) WHERE run_id=?",
                (status, started, completed, error_text, run_id),
            )

    def create_result(self, run_id: str, scenario: Any) -> str:
        result_id = uuid.uuid4().hex
        media = [asdict(item) for item in scenario.events]
        with self._connection() as conn:
            conn.execute(
                "INSERT INTO real_world_results VALUES (?, ?, ?, ?, ?, ?, NULL, 'queued', ?, ?, NULL, NULL, NULL, NULL)",
                (result_id, run_id, scenario.scenario_id, scenario.title, scenario.modality,
                 datetime.now().isoformat(), scenario.rubric_notes, json.dumps(media)),
            )
        return result_id

    def update_result(self, result_id: str, status: str, *, transcript_text: str | None = None,
                      final_response: str | None = None, summary: dict[str, Any] | None = None,
                      error_text: str | None = None) -> None:
        completed = datetime.now().isoformat() if status in TERMINAL_STATUSES else None
        with self._connection() as conn:
            conn.execute(
                "UPDATE real_world_results SET status=?, completed_at=COALESCE(?, completed_at), "
                "transcript_text=COALESCE(?, transcript_text), final_response=COALESCE(?, final_response), "
                "summary_json=COALESCE(?, summary_json), error_text=COALESCE(?, error_text) WHERE result_id=?",
                (status, completed, transcript_text, final_response,
                 json.dumps(summary) if summary is not None else None, error_text, result_id),
            )

    def append_event(self, *, run_id: str, result_id: str | None, stage: str, event_type: str,
                     status: str = "completed", payload: dict[str, Any] | None = None,
                     model: str | None = None, duration_ms: int | None = None,
                     span_id: str | None = None, parent_span_id: str | None = None) -> TraceEvent:
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence FROM real_world_events WHERE run_id=?",
                (run_id,),
            ).fetchone()
            sequence = int(row["next_sequence"])
            event = TraceEvent(
                event_id=uuid.uuid4().hex, run_id=run_id, result_id=result_id,
                sequence=sequence, created_at=datetime.now().isoformat(), stage=stage,
                event_type=event_type, status=status, model=model, duration_ms=duration_ms,
                span_id=span_id, parent_span_id=parent_span_id, payload=payload or {},
            )
            conn.execute(
                "INSERT INTO real_world_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (event.event_id, event.run_id, event.result_id, event.sequence, event.created_at,
                 event.stage, event.event_type, event.status, event.model, event.duration_ms,
                 event.span_id, event.parent_span_id, json.dumps(event.payload, ensure_ascii=False)),
            )
        return event

    def request_cancel(self, run_id: str) -> bool:
        with self._connection() as conn:
            cursor = conn.execute(
                "UPDATE real_world_runs SET cancel_requested=1 WHERE run_id=? AND status IN ('queued','running')", (run_id,)
            )
        return cursor.rowcount > 0

    def is_cancel_requested(self, run_id: str) -> bool:
        row = self._one("SELECT cancel_requested FROM real_world_runs WHERE run_id=?", (run_id,))
        return bool(row and row["cancel_requested"])

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._all("SELECT * FROM real_world_runs ORDER BY created_at DESC LIMIT ?", (limit,))
        return [self._run_dict(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._one("SELECT * FROM real_world_runs WHERE run_id=?", (run_id,))
        if row is None:
            return None
        item = self._run_dict(row)
        results = self._all("SELECT * FROM real_world_results WHERE run_id=? ORDER BY created_at", (run_id,))
        item["results"] = [self._result_dict(result) for result in results]
        return item

    def get_result(self, result_id: str) -> dict[str, Any] | None:
        row = self._one("SELECT * FROM real_world_results WHERE result_id=?", (result_id,))
        if row is None:
            return None
        item = self._result_dict(row)
        review = self._one("SELECT * FROM real_world_reviews WHERE result_id=?", (result_id,))
        item["review"] = dict(review) if review else None
        return item

    def list_events(self, run_id: str, after_sequence: int = 0, limit: int = 1000) -> list[dict[str, Any]]:
        rows = self._all(
            "SELECT * FROM real_world_events WHERE run_id=? AND sequence>? ORDER BY sequence LIMIT ?",
            (run_id, after_sequence, limit),
        )
        return [{**dict(row), "payload": json.loads(row["payload_json"] or "{}") } for row in rows]

    def upsert_review(self, result_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.get_result(result_id) is None:
            raise KeyError(result_id)
        score_fields = ["perception_score", "decision_score", "tool_choice_score",
                        "tool_execution_score", "final_response_score", "overall_score"]
        values: dict[str, int | None] = {}
        for field in score_fields:
            raw = payload.get(field)
            if raw in (None, ""):
                values[field] = None
            else:
                score = int(raw)
                if not 1 <= score <= 5:
                    raise ValueError(f"{field} must be between 1 and 5")
                values[field] = score
        now = datetime.now().isoformat()
        existing = self._one("SELECT review_id, created_at FROM real_world_reviews WHERE result_id=?", (result_id,))
        review_id = existing["review_id"] if existing else uuid.uuid4().hex
        created_at = existing["created_at"] if existing else now
        with self._connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO real_world_reviews VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (review_id, result_id, created_at, now, str(payload.get("reviewer") or "local-user"),
                 *(values[field] for field in score_fields), str(payload.get("notes") or "").strip() or None),
            )
        return dict(self._one("SELECT * FROM real_world_reviews WHERE result_id=?", (result_id,)))

    def register_media(self, *, kind: str, original_name: str, path: Path) -> dict[str, Any]:
        media_id = uuid.uuid4().hex
        with self._connection() as conn:
            conn.execute("INSERT INTO real_world_media VALUES (?, ?, ?, ?, ?, ?)",
                         (media_id, datetime.now().isoformat(), kind, original_name, str(path), path.stat().st_size))
        return dict(self._one("SELECT * FROM real_world_media WHERE media_id=?", (media_id,)))

    def get_media(self, media_id: str) -> dict[str, Any] | None:
        row = self._one("SELECT * FROM real_world_media WHERE media_id=?", (media_id,))
        return dict(row) if row else None

    def _one(self, query: str, params: tuple[Any, ...]) -> sqlite3.Row | None:
        with self._connection() as conn:
            return conn.execute(query, params).fetchone()

    def _all(self, query: str, params: tuple[Any, ...]) -> list[sqlite3.Row]:
        with self._connection() as conn:
            return conn.execute(query, params).fetchall()

    def _run_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        for key in ("scenario_ids_json", "model_roles_json", "config_json"):
            item[key.removesuffix("_json")] = json.loads(item.pop(key) or "{}")
        item["cancel_requested"] = bool(item["cancel_requested"])
        return item

    def _result_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["media"] = json.loads(item.pop("media_json") or "[]")
        item["summary"] = json.loads(item.pop("summary_json") or "{}")
        return item
