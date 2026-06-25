import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from application.ports.activity_ledger_port import ActivityLedgerPort
from core.models import (
    ActivityArtifact,
    ActivityLink,
    ActivityRun,
    ActivityRunDetail,
    ActivityStep,
    ActivityTraceLink,
)


class SQLiteActivityLedgerAdapter(ActivityLedgerPort):
    """SQLite-backed user-facing ledger for meaningful agent work."""

    def __init__(self, db_path: str, interaction_log_db_path: str | None = None):
        self.db_path = Path(db_path)
        self.interaction_log_db_path = Path(interaction_log_db_path) if interaction_log_db_path else None
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _interaction_connection(self) -> sqlite3.Connection | None:
        if self.interaction_log_db_path is None or not self.interaction_log_db_path.exists():
            return None
        conn = sqlite3.connect(self.interaction_log_db_path)
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
                CREATE TABLE IF NOT EXISTS activity_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT NOT NULL,
                    source_kind TEXT NOT NULL,
                    trigger_kind TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    output_text TEXT NOT NULL,
                    model TEXT NOT NULL,
                    session_id TEXT,
                    parent_run_id TEXT,
                    priority TEXT NOT NULL,
                    error_text TEXT,
                    metadata_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS activity_steps (
                    step_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    step_kind TEXT NOT NULL,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    input_ref TEXT,
                    output_ref TEXT,
                    error_text TEXT,
                    metadata_json TEXT,
                    FOREIGN KEY(run_id) REFERENCES activity_runs(run_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS activity_artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    step_id TEXT,
                    artifact_kind TEXT NOT NULL,
                    title TEXT NOT NULL,
                    path TEXT,
                    mime_type TEXT,
                    text_preview TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES activity_runs(run_id),
                    FOREIGN KEY(step_id) REFERENCES activity_steps(step_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS activity_links (
                    link_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    metadata_json TEXT,
                    FOREIGN KEY(run_id) REFERENCES activity_runs(run_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS activity_tags (
                    tag_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(run_id, tag),
                    FOREIGN KEY(run_id) REFERENCES activity_runs(run_id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_activity_runs_created_at ON activity_runs (created_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_activity_runs_status ON activity_runs (status, created_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_activity_runs_source_kind ON activity_runs (source_kind, created_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_activity_steps_run_id ON activity_steps (run_id, step_index ASC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_activity_artifacts_run_id ON activity_artifacts (run_id, created_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_activity_links_run_id ON activity_links (run_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_activity_tags_tag ON activity_tags (tag, created_at DESC)"
            )

    def create_run(
        self,
        *,
        source_kind: str,
        trigger_kind: str,
        title: str,
        status: str = "queued",
        summary: str = "",
        output_text: str = "",
        model: str = "",
        session_id: str | None = None,
        parent_run_id: str | None = None,
        priority: str = "medium",
        error_text: str | None = None,
        metadata: dict | None = None,
        tags: list[str] | None = None,
    ) -> ActivityRun:
        now = datetime.now().isoformat()
        run = ActivityRun(
            run_id=uuid.uuid4().hex,
            created_at=now,
            completed_at=None,
            status=status,
            source_kind=source_kind,
            trigger_kind=trigger_kind,
            title=title,
            summary=summary,
            output_text=output_text,
            model=model,
            session_id=session_id,
            parent_run_id=parent_run_id,
            priority=priority,
            error_text=error_text,
            metadata_json=self._dump_json(metadata),
        )
        with self._managed_connection() as conn:
            conn.execute(
                """
                INSERT INTO activity_runs (
                    run_id, created_at, completed_at, status, source_kind, trigger_kind,
                    title, summary, output_text, model, session_id, parent_run_id,
                    priority, error_text, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.created_at,
                    run.completed_at,
                    run.status,
                    run.source_kind,
                    run.trigger_kind,
                    run.title,
                    run.summary,
                    run.output_text,
                    run.model,
                    run.session_id,
                    run.parent_run_id,
                    run.priority,
                    run.error_text,
                    run.metadata_json,
                ),
            )
        if tags:
            self.add_tags(run.run_id, tags)
        return run

    def update_run_status(
        self,
        run_id: str,
        *,
        status: str,
        summary: str | None = None,
        output_text: str | None = None,
        completed_at: str | None = None,
        error_text: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        with self._managed_connection() as conn:
            existing = conn.execute(
                "SELECT metadata_json FROM activity_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            merged_metadata = self._merge_metadata(
                self._load_json(existing["metadata_json"]) if existing else None,
                metadata,
            )
            conn.execute(
                """
                UPDATE activity_runs
                SET status = ?,
                    summary = COALESCE(?, summary),
                    output_text = COALESCE(?, output_text),
                    completed_at = COALESCE(?, completed_at),
                    error_text = ?,
                    metadata_json = ?
                WHERE run_id = ?
                """,
                (
                    status,
                    summary,
                    output_text,
                    completed_at,
                    error_text,
                    self._dump_json(merged_metadata),
                    run_id,
                ),
            )

    def append_step(
        self,
        *,
        run_id: str,
        step_kind: str,
        title: str,
        status: str = "running",
        input_ref: str | None = None,
        output_ref: str | None = None,
        error_text: str | None = None,
        metadata: dict | None = None,
    ) -> ActivityStep:
        now = datetime.now().isoformat()
        with self._managed_connection() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(step_index), -1) AS max_step_index FROM activity_steps WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            step = ActivityStep(
                step_id=uuid.uuid4().hex,
                run_id=run_id,
                step_index=int(row["max_step_index"]) + 1,
                step_kind=step_kind,
                title=title,
                status=status,
                started_at=now,
                completed_at=None,
                input_ref=input_ref,
                output_ref=output_ref,
                error_text=error_text,
                metadata_json=self._dump_json(metadata),
            )
            conn.execute(
                """
                INSERT INTO activity_steps (
                    step_id, run_id, step_index, step_kind, title, status,
                    started_at, completed_at, input_ref, output_ref, error_text, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    step.step_id,
                    step.run_id,
                    step.step_index,
                    step.step_kind,
                    step.title,
                    step.status,
                    step.started_at,
                    step.completed_at,
                    step.input_ref,
                    step.output_ref,
                    step.error_text,
                    step.metadata_json,
                ),
            )
        return step

    def complete_step(
        self,
        step_id: str,
        *,
        status: str,
        completed_at: str | None = None,
        output_ref: str | None = None,
        error_text: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        with self._managed_connection() as conn:
            existing = conn.execute(
                "SELECT metadata_json FROM activity_steps WHERE step_id = ?",
                (step_id,),
            ).fetchone()
            merged_metadata = self._merge_metadata(
                self._load_json(existing["metadata_json"]) if existing else None,
                metadata,
            )
            conn.execute(
                """
                UPDATE activity_steps
                SET status = ?,
                    completed_at = ?,
                    output_ref = COALESCE(?, output_ref),
                    error_text = ?,
                    metadata_json = ?
                WHERE step_id = ?
                """,
                (
                    status,
                    completed_at or datetime.now().isoformat(),
                    output_ref,
                    error_text,
                    self._dump_json(merged_metadata),
                    step_id,
                ),
            )

    def attach_artifact(
        self,
        *,
        run_id: str,
        artifact_kind: str,
        title: str,
        step_id: str | None = None,
        path: str | None = None,
        mime_type: str | None = None,
        text_preview: str | None = None,
        metadata: dict | None = None,
    ) -> ActivityArtifact:
        artifact = ActivityArtifact(
            artifact_id=uuid.uuid4().hex,
            run_id=run_id,
            step_id=step_id,
            artifact_kind=artifact_kind,
            title=title,
            path=path,
            mime_type=mime_type,
            text_preview=text_preview,
            metadata_json=self._dump_json(metadata),
            created_at=datetime.now().isoformat(),
        )
        with self._managed_connection() as conn:
            conn.execute(
                """
                INSERT INTO activity_artifacts (
                    artifact_id, run_id, step_id, artifact_kind, title,
                    path, mime_type, text_preview, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact.artifact_id,
                    artifact.run_id,
                    artifact.step_id,
                    artifact.artifact_kind,
                    artifact.title,
                    artifact.path,
                    artifact.mime_type,
                    artifact.text_preview,
                    artifact.metadata_json,
                    artifact.created_at,
                ),
            )
        return artifact

    def link_entity(
        self,
        *,
        run_id: str,
        entity_type: str,
        entity_id: str,
        relation: str,
        metadata: dict | None = None,
    ) -> ActivityLink:
        link = ActivityLink(
            link_id=uuid.uuid4().hex,
            run_id=run_id,
            entity_type=entity_type,
            entity_id=entity_id,
            relation=relation,
            metadata_json=self._dump_json(metadata),
        )
        with self._managed_connection() as conn:
            conn.execute(
                """
                INSERT INTO activity_links (link_id, run_id, entity_type, entity_id, relation, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    link.link_id,
                    link.run_id,
                    link.entity_type,
                    link.entity_id,
                    link.relation,
                    link.metadata_json,
                ),
            )
        return link

    def add_tags(self, run_id: str, tags: list[str]) -> None:
        normalized = sorted({tag.strip().lower() for tag in tags if tag and tag.strip()})
        if not normalized:
            return
        now = datetime.now().isoformat()
        with self._managed_connection() as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO activity_tags (tag_id, run_id, tag, created_at)
                VALUES (?, ?, ?, ?)
                """,
                [(uuid.uuid4().hex, run_id, tag, now) for tag in normalized],
            )

    def list_runs(
        self,
        *,
        status: str | None = None,
        source_kind: str | None = None,
        trigger_kind: str | None = None,
        tag: str | None = None,
        query: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 50,
    ) -> list[ActivityRun]:
        sql = ["SELECT DISTINCT r.* FROM activity_runs r"]
        params: List[object] = []
        predicates: List[str] = []
        if tag:
            sql.append("JOIN activity_tags t ON t.run_id = r.run_id")
            predicates.append("t.tag = ?")
            params.append(tag.strip().lower())
        if status:
            predicates.append("r.status = ?")
            params.append(status)
        if source_kind:
            predicates.append("r.source_kind = ?")
            params.append(source_kind)
        if trigger_kind:
            predicates.append("r.trigger_kind = ?")
            params.append(trigger_kind)
        if date_from:
            predicates.append("r.created_at >= ?")
            params.append(date_from)
        if date_to:
            predicates.append("r.created_at <= ?")
            params.append(date_to)
        if query:
            predicates.append(
                "(LOWER(r.title) LIKE ? OR LOWER(r.summary) LIKE ? OR LOWER(r.output_text) LIKE ?)"
            )
            needle = f"%{query.lower()}%"
            params.extend([needle, needle, needle])
        if predicates:
            sql.append("WHERE " + " AND ".join(predicates))
        sql.append("ORDER BY r.created_at DESC LIMIT ?")
        params.append(limit)
        with self._managed_connection() as conn:
            rows = conn.execute(" ".join(sql), params).fetchall()
        return [self._run_from_row(row) for row in rows]

    def get_run_detail(self, run_id: str) -> Optional[ActivityRunDetail]:
        with self._managed_connection() as conn:
            run_row = conn.execute("SELECT * FROM activity_runs WHERE run_id = ?", (run_id,)).fetchone()
            if run_row is None:
                return None
            steps = [
                self._step_from_row(row)
                for row in conn.execute(
                    "SELECT * FROM activity_steps WHERE run_id = ? ORDER BY step_index ASC",
                    (run_id,),
                ).fetchall()
            ]
            artifacts = [
                self._artifact_from_row(row)
                for row in conn.execute(
                    "SELECT * FROM activity_artifacts WHERE run_id = ? ORDER BY created_at ASC",
                    (run_id,),
                ).fetchall()
            ]
            links = [
                self._link_from_row(row)
                for row in conn.execute(
                    "SELECT * FROM activity_links WHERE run_id = ? ORDER BY rowid ASC",
                    (run_id,),
                ).fetchall()
            ]
            tags = [
                row["tag"]
                for row in conn.execute(
                    "SELECT tag FROM activity_tags WHERE run_id = ? ORDER BY tag ASC",
                    (run_id,),
                ).fetchall()
            ]
        return ActivityRunDetail(
            run=self._run_from_row(run_row),
            steps=steps,
            artifacts=artifacts,
            links=links,
            tags=tags,
            traces=self._list_trace_links(run_id),
        )

    def search_runs(self, query: str, limit: int = 50) -> list[ActivityRun]:
        query = query.strip()
        if not query:
            return []
        runs = self.list_runs(query=query, limit=limit)
        if len(runs) >= limit:
            return runs
        with self._managed_connection() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT r.*
                FROM activity_runs r
                LEFT JOIN activity_tags t ON t.run_id = r.run_id
                LEFT JOIN activity_artifacts a ON a.run_id = r.run_id
                WHERE LOWER(COALESCE(t.tag, '')) LIKE ?
                   OR LOWER(COALESCE(a.text_preview, '')) LIKE ?
                   OR LOWER(COALESCE(a.title, '')) LIKE ?
                ORDER BY r.created_at DESC
                LIMIT ?
                """,
                (f"%{query.lower()}%", f"%{query.lower()}%", f"%{query.lower()}%", limit),
            ).fetchall()
        seen = {run.run_id for run in runs}
        for row in rows:
            if row["run_id"] not in seen:
                runs.append(self._run_from_row(row))
                seen.add(row["run_id"])
            if len(runs) >= limit:
                break
        return runs

    def summarize(self) -> dict:
        with self._managed_connection() as conn:
            completed = conn.execute(
                "SELECT COUNT(*) AS count FROM activity_runs WHERE status = 'completed'"
            ).fetchone()["count"]
            failed = conn.execute(
                "SELECT COUNT(*) AS count FROM activity_runs WHERE status = 'failed'"
            ).fetchone()["count"]
            pending = conn.execute(
                "SELECT COUNT(*) AS count FROM activity_runs WHERE status IN ('queued', 'running')"
            ).fetchone()["count"]
            research = conn.execute(
                "SELECT COUNT(*) AS count FROM activity_runs WHERE source_kind = 'proactive_research'"
            ).fetchone()["count"]
            tasks = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM activity_runs
                WHERE source_kind IN ('ambient_ai_task', 'transcript_task', 'simple_execution')
                """
            ).fetchone()["count"]
        return {
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "research": research,
            "tasks": tasks,
            "recent_completed": [self._run_to_dict(item) for item in self.list_runs(status="completed", limit=5)],
        }

    def get_artifact(self, artifact_id: str) -> Optional[ActivityArtifact]:
        with self._managed_connection() as conn:
            row = conn.execute(
                "SELECT * FROM activity_artifacts WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
        return self._artifact_from_row(row) if row else None

    def _list_trace_links(self, run_id: str) -> list[ActivityTraceLink]:
        conn = self._interaction_connection()
        if conn is None:
            return []
        try:
            rows = conn.execute(
                """
                SELECT interaction_id, created_at, source, model, response_text,
                       reasoning_text, tool_calls_json, error_text, duration_ms, metadata_json
                FROM interaction_logs
                ORDER BY created_at DESC
                """
            ).fetchall()
        finally:
            conn.close()
        traces: list[ActivityTraceLink] = []
        for row in rows:
            metadata = self._load_json(row["metadata_json"]) or {}
            if metadata.get("run_id") != run_id:
                continue
            traces.append(
                ActivityTraceLink(
                    interaction_id=row["interaction_id"],
                    created_at=row["created_at"],
                    source=row["source"],
                    model=row["model"],
                    response_text=row["response_text"],
                    reasoning_text=row["reasoning_text"],
                    tool_calls_json=row["tool_calls_json"],
                    error_text=row["error_text"],
                    duration_ms=row["duration_ms"],
                    metadata_json=row["metadata_json"],
                )
            )
        return traces

    def _run_from_row(self, row: sqlite3.Row) -> ActivityRun:
        return ActivityRun(
            run_id=row["run_id"],
            created_at=row["created_at"],
            completed_at=row["completed_at"],
            status=row["status"],
            source_kind=row["source_kind"],
            trigger_kind=row["trigger_kind"],
            title=row["title"],
            summary=row["summary"],
            output_text=row["output_text"],
            model=row["model"],
            session_id=row["session_id"],
            parent_run_id=row["parent_run_id"],
            priority=row["priority"],
            error_text=row["error_text"],
            metadata_json=row["metadata_json"],
        )

    def _step_from_row(self, row: sqlite3.Row) -> ActivityStep:
        return ActivityStep(
            step_id=row["step_id"],
            run_id=row["run_id"],
            step_index=row["step_index"],
            step_kind=row["step_kind"],
            title=row["title"],
            status=row["status"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            input_ref=row["input_ref"],
            output_ref=row["output_ref"],
            error_text=row["error_text"],
            metadata_json=row["metadata_json"],
        )

    def _artifact_from_row(self, row: sqlite3.Row) -> ActivityArtifact:
        return ActivityArtifact(
            artifact_id=row["artifact_id"],
            run_id=row["run_id"],
            step_id=row["step_id"],
            artifact_kind=row["artifact_kind"],
            title=row["title"],
            path=row["path"],
            mime_type=row["mime_type"],
            text_preview=row["text_preview"],
            metadata_json=row["metadata_json"],
            created_at=row["created_at"],
        )

    def _link_from_row(self, row: sqlite3.Row) -> ActivityLink:
        return ActivityLink(
            link_id=row["link_id"],
            run_id=row["run_id"],
            entity_type=row["entity_type"],
            entity_id=row["entity_id"],
            relation=row["relation"],
            metadata_json=row["metadata_json"],
        )

    def _run_to_dict(self, run: ActivityRun) -> dict:
        return {
            "run_id": run.run_id,
            "created_at": run.created_at,
            "status": run.status,
            "source_kind": run.source_kind,
            "title": run.title,
            "summary": run.summary,
        }

    def _dump_json(self, payload: dict | None) -> str | None:
        if not payload:
            return None
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _load_json(self, payload: str | None) -> dict | None:
        if not payload:
            return None
        try:
            loaded = json.loads(payload)
        except json.JSONDecodeError:
            return None
        return loaded if isinstance(loaded, dict) else None

    def _merge_metadata(self, base: dict | None, update: dict | None) -> dict | None:
        merged = dict(base or {})
        merged.update(update or {})
        return merged or None
