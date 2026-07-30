from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


@dataclass
class ArtifactCandidate:
    artifact_id: str
    title: str
    artifact_path: str
    short_summary: str
    detailed_summary: str
    last_ai_edited_at: str
    score: float = 0.0
    match_source: str = "lexical"

    def prompt_summary(self, *, summary_words: int) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "title": self.title,
            "artifact_path": self.artifact_path,
            "last_ai_edited_at": self.last_ai_edited_at,
            "short_summary": _truncate_words(self.short_summary or self.detailed_summary, summary_words),
            "score": round(self.score, 4),
            "match_source": self.match_source,
        }


def _now() -> str:
    return datetime.now().isoformat(timespec="microseconds")


def _truncate_words(text: str, limit: int) -> str:
    words = re.findall(r"\S+", text or "")
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit]).rstrip() + " ..."


def _safe_json(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return fallback


class ArtifactOrganizer:
    """Registry-backed artifact creator/merger for user-facing Markdown reports."""

    def __init__(
        self,
        artifact_root: str | Path,
        *,
        candidate_summary_words: int = 50,
        candidate_limit: int = 8,
        full_candidate_limit: int = 3,
        max_existing_artifact_chars: int = 50_000,
        semantic_memory: Any | None = None,
        semantic_sync_on_write: bool = True,
    ) -> None:
        self.artifact_root = Path(artifact_root)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.artifact_root / "artifacts.db"
        self.candidate_summary_words = max(10, int(candidate_summary_words))
        self.candidate_limit = max(1, int(candidate_limit))
        self.full_candidate_limit = max(1, int(full_candidate_limit))
        self.max_existing_artifact_chars = max(1000, int(max_existing_artifact_chars))
        self.semantic_memory = semantic_memory
        self.semantic_sync_on_write = bool(semantic_sync_on_write)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._init_db()
        self.backfill_registry()

    @contextmanager
    def _connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    artifact_path TEXT NOT NULL UNIQUE,
                    artifact_kind TEXT NOT NULL,
                    short_summary TEXT,
                    detailed_summary TEXT,
                    topics_json TEXT,
                    source_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    last_ai_edited_at TEXT NOT NULL,
                    last_source_ref TEXT,
                    content_hash TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    canonical_artifact_id TEXT,
                    archived_at TEXT
                )
                """
            )
            existing = {row[1] for row in conn.execute("PRAGMA table_info(artifacts)").fetchall()}
            migrations = {
                "status": "TEXT NOT NULL DEFAULT 'active'",
                "canonical_artifact_id": "TEXT",
                "archived_at": "TEXT",
            }
            for name, definition in migrations.items():
                if name not in existing:
                    conn.execute(f"ALTER TABLE artifacts ADD COLUMN {name} {definition}")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS artifact_maintenance_runs (
                    run_id TEXT PRIMARY KEY,
                    trigger_kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    scanned_count INTEGER NOT NULL DEFAULT 0,
                    candidate_pair_count INTEGER NOT NULL DEFAULT 0,
                    cluster_count INTEGER NOT NULL DEFAULT 0,
                    merged_cluster_count INTEGER NOT NULL DEFAULT 0,
                    archived_count INTEGER NOT NULL DEFAULT 0,
                    continuation_required INTEGER NOT NULL DEFAULT 0,
                    error_text TEXT
                );
                CREATE TABLE IF NOT EXISTS artifact_merge_history (
                    merge_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    canonical_artifact_id TEXT NOT NULL,
                    archived_artifact_ids_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    rationale TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )

    def backfill_registry(self) -> None:
        for path in sorted(self.artifact_root.glob("*.md")):
            if path.name.startswith("."):
                continue
            if self._get_by_path(path) is not None:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            title = self._extract_title(text) or path.stem.replace("_", " ")
            short_summary = self._extract_section(text, "Short Summary") or self._extract_section(text, "Summary")
            detailed_summary = self._extract_section(text, "Detailed Summary") or short_summary or self._plain_excerpt(text, 120)
            timestamp = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
            self._upsert_record(
                artifact_id=uuid.uuid4().hex,
                title=title,
                artifact_path=path,
                artifact_kind="note",
                short_summary=_truncate_words(short_summary, self.candidate_summary_words),
                detailed_summary=detailed_summary,
                topics=[],
                source_count=1,
                created_at=timestamp,
                last_ai_edited_at=timestamp,
                last_source_ref=None,
                content_hash=self._hash(text),
            )

    def candidates_for(self, *, title: str, summary: str, detailed_report: str) -> list[ArtifactCandidate]:
        query_terms = self._terms(" ".join([title, summary, detailed_report[:2000]]))
        rows = self._list_records()
        candidates_by_id: dict[str, ArtifactCandidate] = {}
        for candidate in self._semantic_candidates_for(
            title=title,
            summary=summary,
            detailed_report=detailed_report,
        ):
            candidates_by_id[candidate.artifact_id] = candidate
        for row in rows:
            haystack = " ".join(
                [
                    row["title"] or "",
                    row["short_summary"] or "",
                    row["detailed_summary"] or "",
                    row["topics_json"] or "",
                ]
            )
            score = self._lexical_score(query_terms, self._terms(haystack))
            if score <= 0:
                continue
            existing = candidates_by_id.get(row["artifact_id"])
            if existing is not None:
                existing.score = max(existing.score, score)
                existing.match_source = "semantic+lexical"
            else:
                candidates_by_id[row["artifact_id"]] = self._candidate_from_row(
                    row,
                    score=score,
                    match_source="lexical",
                )
        candidates = list(candidates_by_id.values())
        candidates.sort(key=lambda item: (item.score, item.last_ai_edited_at), reverse=True)
        return candidates[: self.candidate_limit]

    def build_existing_payload(self, candidates: Iterable[ArtifactCandidate]) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for candidate in list(candidates)[: self.full_candidate_limit]:
            path = self._resolve_artifact_path(candidate.artifact_path)
            content = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
            payload.append(
                {
                    **candidate.prompt_summary(summary_words=self.candidate_summary_words),
                    "current_content": content[: self.max_existing_artifact_chars],
                }
            )
        return payload

    def save_new(
        self,
        *,
        title: str,
        summary: str,
        detailed_report: str,
        source_ref: str,
    ) -> dict[str, Any]:
        timestamp = _now()
        path = self._unique_path(title)
        content = self._format_artifact(
            title=title,
            last_edit=timestamp,
            short_summary=summary,
            detailed_summary=summary,
            body=detailed_report,
            update_line=f"{timestamp} — Created from {source_ref}.",
        )
        path.write_text(content, encoding="utf-8")
        artifact_id = uuid.uuid4().hex
        self._upsert_record(
            artifact_id=artifact_id,
            title=title,
            artifact_path=path,
            artifact_kind="note",
            short_summary=_truncate_words(summary, self.candidate_summary_words),
            detailed_summary=summary,
            topics=[],
            source_count=1,
            created_at=timestamp,
            last_ai_edited_at=timestamp,
            last_source_ref=source_ref,
            content_hash=self._hash(content),
        )
        return {
            "artifact_id": artifact_id,
            "artifact_path": str(path),
            "artifact_action": "created",
            "artifact_reason": "No suitable existing artifact was selected.",
            "dedupe_notes": [],
        }

    def apply_decision(
        self,
        *,
        decision: dict[str, Any],
        fallback_title: str,
        fallback_summary: str,
        fallback_detailed_report: str,
        source_ref: str,
    ) -> dict[str, Any]:
        action = str(decision.get("action") or "").strip().lower()
        target_id = str(decision.get("target_artifact_id") or "").strip()
        final_title = str(decision.get("final_title") or fallback_title).strip() or fallback_title
        short_summary = str(decision.get("updated_short_summary") or fallback_summary).strip() or fallback_summary
        detailed_summary = str(decision.get("updated_detailed_summary") or short_summary).strip() or short_summary
        merged_content = str(decision.get("merged_content") or "").strip()
        dedupe_notes = _safe_json(decision.get("dedupe_notes"), [])
        if not isinstance(dedupe_notes, list):
            dedupe_notes = [str(dedupe_notes)]
        reason = str(decision.get("reason") or "").strip()

        if action != "merge_existing" or not target_id:
            result = self.save_new(
                title=final_title,
                summary=short_summary,
                detailed_report=fallback_detailed_report,
                source_ref=source_ref,
            )
            result["artifact_reason"] = reason or result["artifact_reason"]
            result["dedupe_notes"] = dedupe_notes
            return result

        row = self._get_by_id(target_id)
        if row is None or str(row["status"] or "active") != "active":
            result = self.save_new(
                title=final_title,
                summary=short_summary,
                detailed_report=fallback_detailed_report,
                source_ref=source_ref,
            )
            result["artifact_reason"] = f"Selected artifact {target_id} was not found; created a new artifact."
            return result

        timestamp = _now()
        path = self._resolve_artifact_path(row["artifact_path"])
        if not merged_content:
            previous = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
            merged_content = previous.rstrip() + "\n\n" + fallback_detailed_report
        content = self._ensure_standard_format(
            title=final_title,
            last_edit=timestamp,
            short_summary=short_summary,
            detailed_summary=detailed_summary,
            merged_content=merged_content,
            source_ref=source_ref,
        )
        path.write_text(content, encoding="utf-8")
        self._upsert_record(
            artifact_id=target_id,
            title=final_title,
            artifact_path=path,
            artifact_kind=row["artifact_kind"] or "note",
            short_summary=_truncate_words(short_summary, self.candidate_summary_words),
            detailed_summary=detailed_summary,
            topics=_safe_json(row["topics_json"], []),
            source_count=int(row["source_count"] or 0) + 1,
            created_at=row["created_at"],
            last_ai_edited_at=timestamp,
            last_source_ref=source_ref,
            content_hash=self._hash(content),
        )
        return {
            "artifact_id": target_id,
            "artifact_path": str(path),
            "artifact_action": "merged",
            "artifact_reason": reason or f"Merged into existing artifact: {row['title']}",
            "dedupe_notes": dedupe_notes,
        }

    def list_artifacts(self, *, status: str = "active", limit: int = 500) -> list[dict[str, Any]]:
        normalized = status if status in {"active", "archived"} else "active"
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM artifacts WHERE status = ? ORDER BY last_ai_edited_at DESC LIMIT ?",
                (normalized, max(1, int(limit))),
            ).fetchall()
        return [self._record_payload(row, include_content=False) for row in rows]

    def get_artifact(self, artifact_id: str, *, include_content: bool = True) -> dict[str, Any] | None:
        row = self._get_by_id(str(artifact_id))
        return self._record_payload(row, include_content=include_content) if row is not None else None

    def reconcile_and_reindex(self) -> int:
        """Register loose files and ensure every active registry row has a semantic chunk."""
        self.backfill_registry()
        rows = self._list_records()
        for row in rows:
            self._index_semantic_record(
                artifact_id=row["artifact_id"],
                title=row["title"],
                artifact_path=row["artifact_path"],
                artifact_kind=row["artifact_kind"],
                short_summary=row["short_summary"],
                detailed_summary=row["detailed_summary"],
                topics=_safe_json(row["topics_json"], []),
                source_count=row["source_count"],
                last_ai_edited_at=row["last_ai_edited_at"],
                sync=False,
            )
        if rows and self.semantic_memory is not None and hasattr(self.semantic_memory, "ensure_embeddings_synced"):
            self.semantic_memory.ensure_embeddings_synced(max_batches=None)
        return len(rows)

    def maintenance_status(self, *, min_changes: int, interval_hours: float) -> dict[str, Any]:
        with self._connection() as conn:
            last = conn.execute(
                "SELECT * FROM artifact_maintenance_runs ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
            last_success = conn.execute(
                "SELECT * FROM artifact_maintenance_runs WHERE status = 'completed' "
                "ORDER BY completed_at DESC LIMIT 1"
            ).fetchone()
            active_count = int(
                conn.execute("SELECT COUNT(*) FROM artifacts WHERE status = 'active'").fetchone()[0]
            )
            archived_count = int(
                conn.execute("SELECT COUNT(*) FROM artifacts WHERE status = 'archived'").fetchone()[0]
            )
            if last_success is None:
                changed_count = active_count
            else:
                changed_count = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM artifacts WHERE status = 'active' AND last_ai_edited_at > ?",
                        (last_success["completed_at"],),
                    ).fetchone()[0]
                )
        now = datetime.now()
        elapsed_hours = None
        if last_success is not None and last_success["completed_at"]:
            try:
                elapsed_hours = max(
                    0.0,
                    (now - datetime.fromisoformat(last_success["completed_at"])).total_seconds() / 3600.0,
                )
            except ValueError:
                elapsed_hours = None
        continuation = bool(last_success and last_success["continuation_required"])
        due_reasons: list[str] = []
        if last_success is None:
            due_reasons.append("initial_scan")
        if changed_count >= max(1, int(min_changes)):
            due_reasons.append("artifact_changes")
        if elapsed_hours is not None and elapsed_hours >= max(1.0, float(interval_hours)):
            due_reasons.append("daily_interval")
        if continuation:
            due_reasons.append("continuation")
        return {
            "active_count": active_count,
            "archived_count": archived_count,
            "changed_count": changed_count,
            "due": bool(due_reasons),
            "due_reasons": due_reasons,
            "last_run": dict(last) if last is not None else None,
            "last_success": dict(last_success) if last_success is not None else None,
        }

    def list_maintenance_history(self, *, limit: int = 50) -> dict[str, Any]:
        with self._connection() as conn:
            runs = conn.execute(
                "SELECT * FROM artifact_maintenance_runs ORDER BY started_at DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
            merges = conn.execute(
                "SELECT * FROM artifact_merge_history ORDER BY created_at DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        return {"runs": [dict(row) for row in runs], "merges": [dict(row) for row in merges]}

    def start_maintenance_run(self, trigger_kind: str) -> str:
        run_id = uuid.uuid4().hex
        with self._connection() as conn:
            conn.execute(
                "INSERT INTO artifact_maintenance_runs "
                "(run_id, trigger_kind, status, started_at) VALUES (?, ?, 'running', ?)",
                (run_id, str(trigger_kind), _now()),
            )
        return run_id

    def finish_maintenance_run(self, run_id: str, *, status: str, **metrics: Any) -> None:
        allowed = {
            "scanned_count", "candidate_pair_count", "cluster_count",
            "merged_cluster_count", "archived_count", "continuation_required", "error_text",
        }
        updates = {key: value for key, value in metrics.items() if key in allowed}
        assignments = ["status = ?", "completed_at = ?"]
        params: list[Any] = [str(status), _now()]
        for key, value in updates.items():
            assignments.append(f"{key} = ?")
            params.append(int(bool(value)) if key == "continuation_required" else value)
        params.append(str(run_id))
        with self._connection() as conn:
            conn.execute(
                f"UPDATE artifact_maintenance_runs SET {', '.join(assignments)} WHERE run_id = ?",
                params,
            )

    def consolidate_cluster(
        self,
        *,
        run_id: str,
        canonical_artifact_id: str,
        artifact_ids: list[str],
        final_title: str,
        short_summary: str,
        detailed_summary: str,
        merged_content: str,
        confidence: float,
        rationale: str,
        archive_dir: str = "archived",
    ) -> dict[str, Any]:
        ordered_ids = list(dict.fromkeys(str(item) for item in artifact_ids))
        if canonical_artifact_id not in ordered_ids or len(ordered_ids) < 2:
            raise ValueError("A maintenance cluster requires a canonical artifact and a duplicate.")
        rows = [self._get_by_id(item) for item in ordered_ids]
        if any(row is None or str(row["status"] or "active") != "active" for row in rows):
            raise ValueError("Maintenance cluster contains a missing or inactive artifact.")
        row_by_id = {str(row["artifact_id"]): row for row in rows if row is not None}
        canonical = row_by_id[canonical_artifact_id]
        duplicates = [row_by_id[item] for item in ordered_ids if item != canonical_artifact_id]
        canonical_path = self._resolve_artifact_path(canonical["artifact_path"])
        previous_canonical = canonical_path.read_text(encoding="utf-8", errors="replace")
        timestamp = _now()
        formatted = self._ensure_standard_format(
            title=final_title,
            last_edit=timestamp,
            short_summary=short_summary,
            detailed_summary=detailed_summary,
            merged_content=merged_content,
            source_ref=f"artifact-maintenance/{run_id}",
        )
        if len(formatted.strip()) < 200:
            raise ValueError("Maintenance merge output was too short to replace the canonical artifact.")
        archive_root = (self.artifact_root / archive_dir).resolve(strict=False)
        archive_root.relative_to(self.artifact_root.resolve(strict=False))
        archive_root.mkdir(parents=True, exist_ok=True)
        moved: list[tuple[Path, Path]] = []
        temp_path = canonical_path.with_name(f".{canonical_path.name}.{run_id}.tmp")
        archived_rows: list[tuple[sqlite3.Row, Path]] = []
        try:
            temp_path.write_text(formatted, encoding="utf-8")
            for duplicate in duplicates:
                source = self._resolve_artifact_path(duplicate["artifact_path"])
                destination = self._unique_archive_path(archive_root, source.name)
                source.replace(destination)
                moved.append((source, destination))
                archived_rows.append((duplicate, destination))
            temp_path.replace(canonical_path)
            total_sources = sum(max(1, int(row["source_count"] or 0)) for row in rows if row is not None)
            merge_id = uuid.uuid4().hex
            with self._connection() as conn:
                conn.execute(
                    "UPDATE artifacts SET title=?, short_summary=?, detailed_summary=?, source_count=?, "
                    "last_ai_edited_at=?, last_source_ref=?, content_hash=?, status='active', "
                    "canonical_artifact_id=NULL, archived_at=NULL WHERE artifact_id=?",
                    (
                        final_title, _truncate_words(short_summary, self.candidate_summary_words),
                        detailed_summary, total_sources, timestamp,
                        f"artifact-maintenance/{run_id}", self._hash(formatted), canonical_artifact_id,
                    ),
                )
                for duplicate, destination in archived_rows:
                    conn.execute(
                        "UPDATE artifacts SET artifact_path=?, status='archived', "
                        "canonical_artifact_id=?, archived_at=? WHERE artifact_id=?",
                        (str(destination), canonical_artifact_id, timestamp, duplicate["artifact_id"]),
                    )
                conn.execute(
                    "INSERT INTO artifact_merge_history "
                    "(merge_id, run_id, canonical_artifact_id, archived_artifact_ids_json, confidence, rationale, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        merge_id, run_id, canonical_artifact_id,
                        json.dumps([row["artifact_id"] for row in duplicates]),
                        float(confidence), str(rationale or ""), timestamp,
                    ),
                )
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            canonical_path.write_text(previous_canonical, encoding="utf-8")
            for source, destination in reversed(moved):
                if destination.exists():
                    destination.replace(source)
            raise
        canonical_payload = self.get_artifact(canonical_artifact_id, include_content=False) or {}
        self._index_semantic_record(
            artifact_id=canonical_artifact_id,
            title=final_title,
            artifact_path=canonical_path,
            artifact_kind=canonical["artifact_kind"],
            short_summary=short_summary,
            detailed_summary=detailed_summary,
            topics=_safe_json(canonical["topics_json"], []),
            source_count=canonical_payload.get("source_count", 1),
            last_ai_edited_at=timestamp,
        )
        memory = getattr(self.semantic_memory, "memory", None)
        if memory is not None and hasattr(memory, "delete_semantic_chunk"):
            for duplicate in duplicates:
                memory.delete_semantic_chunk(f"artifact:{duplicate['artifact_id']}")
        return {
            "canonical_artifact_id": canonical_artifact_id,
            "archived_artifact_ids": [row["artifact_id"] for row in duplicates],
            "archived_count": len(duplicates),
            "artifact_path": str(canonical_path),
        }

    def _ensure_standard_format(
        self,
        *,
        title: str,
        last_edit: str,
        short_summary: str,
        detailed_summary: str,
        merged_content: str,
        source_ref: str,
    ) -> str:
        body = merged_content.strip()
        previous_log = self._extract_section(body, "Update Log")
        if body.startswith("# "):
            body = re.sub(r"^# .+?(?:\n|$)", "", body, count=1).strip()
        body = re.sub(r"(?is)^_Last AI edit:.*?_\s*", "", body).strip()
        if "## Content" in body:
            body = self._extract_section(body, "Content") or body
        update_lines = []
        if previous_log:
            update_lines.extend(
                line.strip()
                for line in previous_log.splitlines()
                if line.strip().startswith("- ")
            )
        update_lines.append(f"- {last_edit} — Merged new information from {source_ref}.")
        return self._format_artifact(
            title=title,
            last_edit=last_edit,
            short_summary=short_summary,
            detailed_summary=detailed_summary,
            body=body,
            update_line="\n".join(update_lines),
        )

    def _format_artifact(
        self,
        *,
        title: str,
        last_edit: str,
        short_summary: str,
        detailed_summary: str,
        body: str,
        update_line: str,
    ) -> str:
        return "\n".join(
            [
                f"# {title}",
                "",
                f"_Last AI edit: {last_edit}_",
                "",
                "## Short Summary",
                _truncate_words(short_summary, self.candidate_summary_words),
                "",
                "## Detailed Summary",
                detailed_summary.strip(),
                "",
                "## Content",
                body.strip(),
                "",
                "## Update Log",
                update_line if update_line.strip().startswith("- ") else f"- {update_line}",
                "",
            ]
        )

    def _get_by_path(self, path: Path) -> sqlite3.Row | None:
        with self._connection() as conn:
            return conn.execute("SELECT * FROM artifacts WHERE artifact_path = ?", (str(path),)).fetchone()

    def _get_by_id(self, artifact_id: str) -> sqlite3.Row | None:
        with self._connection() as conn:
            return conn.execute("SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,)).fetchone()

    def _list_records(self) -> list[sqlite3.Row]:
        with self._connection() as conn:
            return conn.execute(
                "SELECT * FROM artifacts WHERE status = 'active' ORDER BY last_ai_edited_at DESC"
            ).fetchall()

    def _upsert_record(self, **values: Any) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO artifacts (
                    artifact_id, title, artifact_path, artifact_kind, short_summary,
                    detailed_summary, topics_json, source_count, created_at,
                    last_ai_edited_at, last_source_ref, content_hash, status,
                    canonical_artifact_id, archived_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    values["artifact_id"],
                    values["title"],
                    str(values["artifact_path"]),
                    values["artifact_kind"],
                    values.get("short_summary"),
                    values.get("detailed_summary"),
                    json.dumps(values.get("topics") or [], ensure_ascii=False),
                    int(values.get("source_count") or 0),
                    values["created_at"],
                    values["last_ai_edited_at"],
                    values.get("last_source_ref"),
                    values.get("content_hash"),
                    values.get("status", "active"),
                    values.get("canonical_artifact_id"),
                    values.get("archived_at"),
                ),
            )
        self._index_semantic_record(**values)

    def _candidate_from_row(self, row: sqlite3.Row, *, score: float, match_source: str = "lexical") -> ArtifactCandidate:
        return ArtifactCandidate(
            artifact_id=row["artifact_id"],
            title=row["title"],
            artifact_path=row["artifact_path"],
            short_summary=row["short_summary"] or "",
            detailed_summary=row["detailed_summary"] or "",
            last_ai_edited_at=row["last_ai_edited_at"],
            score=score,
            match_source=match_source,
        )

    def _semantic_candidates_for(
        self,
        *,
        title: str,
        summary: str,
        detailed_report: str,
    ) -> list[ArtifactCandidate]:
        semantic_memory = self.semantic_memory
        if semantic_memory is None or not getattr(semantic_memory, "is_enabled", lambda: False)():
            return []
        query = "\n".join(
            [
                f"Title: {title}",
                f"Summary: {summary}",
                "Content:",
                detailed_report[:4000],
            ]
        ).strip()
        try:
            results = semantic_memory.retrieve(
                query=query,
                limit=max(self.candidate_limit * 5, 30),
                rerank_limit=max(self.candidate_limit, 8),
                source_types=["artifact"],
            )
        except Exception as exc:
            self.logger.warning("Artifact semantic retrieval failed: %s", exc)
            return []
        candidates: list[ArtifactCandidate] = []
        for index, result in enumerate(results):
            artifact_id = str(getattr(result.chunk, "source_id", "") or "").strip()
            if not artifact_id:
                continue
            row = self._get_by_id(artifact_id)
            if row is None or str(row["status"] or "active") != "active":
                continue
            semantic_score = result.rerank_score
            if semantic_score is None:
                semantic_score = result.vector_score
            try:
                score = 1.0 + float(semantic_score)
            except (TypeError, ValueError):
                score = 1.0 - (index * 0.01)
            candidates.append(
                self._candidate_from_row(
                    row,
                    score=score,
                    match_source="semantic",
                )
            )
        return candidates

    def _index_semantic_record(self, *, sync: bool = True, **values: Any) -> None:
        semantic_memory = self.semantic_memory
        if semantic_memory is None or not getattr(semantic_memory, "is_enabled", lambda: False)():
            return
        memory = getattr(semantic_memory, "memory", None)
        if memory is None or not hasattr(memory, "upsert_semantic_chunk"):
            return
        artifact_id = str(values["artifact_id"])
        artifact_path = str(values["artifact_path"])
        topics = values.get("topics") or []
        content = "\n".join(
            [
                f"Artifact title: {values.get('title') or ''}",
                f"Short summary: {values.get('short_summary') or ''}",
                f"Detailed summary: {values.get('detailed_summary') or ''}",
                f"Topics: {', '.join(str(topic) for topic in topics)}",
            ]
        )
        metadata = {
            "artifact_id": artifact_id,
            "artifact_path": artifact_path,
            "artifact_kind": values.get("artifact_kind") or "note",
            "title": values.get("title") or "",
            "last_ai_edited_at": values.get("last_ai_edited_at") or "",
            "source_count": int(values.get("source_count") or 0),
        }
        try:
            memory.upsert_semantic_chunk(
                source_type="artifact",
                source_id=artifact_id,
                source_ref=artifact_path,
                content=content,
                metadata_json=json.dumps(metadata, ensure_ascii=False),
            )
            if sync and self.semantic_sync_on_write and hasattr(semantic_memory, "ensure_embeddings_synced"):
                semantic_memory.ensure_embeddings_synced(max_batches=1)
        except Exception as exc:
            self.logger.warning("Artifact semantic indexing failed for %s: %s", artifact_id, exc)

    def _resolve_artifact_path(self, value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = self.artifact_root / path
        path = path.resolve(strict=False)
        root = self.artifact_root.resolve(strict=False)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Artifact path escapes artifact root: {path}") from exc
        return path

    def _unique_path(self, title: str) -> Path:
        safe = self._sanitize_artifact_name(title)
        candidate = self.artifact_root / f"{safe}.md"
        suffix = 1
        while candidate.exists():
            candidate = self.artifact_root / f"{safe}_{suffix}.md"
            suffix += 1
        return candidate

    def _record_payload(self, row: sqlite3.Row, *, include_content: bool) -> dict[str, Any]:
        path = self._resolve_artifact_path(row["artifact_path"])
        payload = {
            "artifact_id": row["artifact_id"],
            "title": row["title"],
            "artifact_path": str(path),
            "artifact_filename": path.name,
            "artifact_kind": row["artifact_kind"],
            "short_summary": row["short_summary"] or "",
            "detailed_summary": row["detailed_summary"] or "",
            "topics": _safe_json(row["topics_json"], []),
            "source_count": int(row["source_count"] or 0),
            "created_at": row["created_at"],
            "last_ai_edited_at": row["last_ai_edited_at"],
            "last_source_ref": row["last_source_ref"],
            "status": row["status"] or "active",
            "canonical_artifact_id": row["canonical_artifact_id"],
            "archived_at": row["archived_at"],
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
        }
        if include_content:
            payload["content"] = (
                path.read_text(encoding="utf-8", errors="replace")[: self.max_existing_artifact_chars]
                if path.exists() else ""
            )
        return payload

    @staticmethod
    def _unique_archive_path(archive_root: Path, filename: str) -> Path:
        candidate = archive_root / filename
        suffix = 1
        while candidate.exists():
            candidate = archive_root / f"{Path(filename).stem}_{suffix}{Path(filename).suffix}"
            suffix += 1
        return candidate

    @staticmethod
    def _sanitize_artifact_name(title: str) -> str:
        cleaned = re.sub(r"[^\w\s-]", "", title, flags=re.UNICODE)
        cleaned = re.sub(r"\s+", "_", cleaned.strip())
        return cleaned[:80] or "artifact"

    @staticmethod
    def _extract_title(text: str) -> str:
        match = re.search(r"^#\s+(.+)$", text or "", flags=re.MULTILINE)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _extract_section(text: str, heading: str) -> str:
        pattern = rf"(?ims)^##\s+{re.escape(heading)}\s*\n(.*?)(?=^##\s+|\Z)"
        match = re.search(pattern, text or "")
        return match.group(1).strip() if match else ""

    @staticmethod
    def _plain_excerpt(text: str, words: int) -> str:
        cleaned = re.sub(r"[#*_`>\[\]()-]+", " ", text or "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return _truncate_words(cleaned, words)

    @staticmethod
    def _terms(text: str) -> set[str]:
        stop = {"the", "and", "for", "with", "this", "that", "from", "into", "about", "what", "when", "where"}
        return {term for term in re.findall(r"[a-zA-Z0-9]{3,}", (text or "").lower()) if term not in stop}

    @staticmethod
    def _lexical_score(query_terms: set[str], artifact_terms: set[str]) -> float:
        if not query_terms or not artifact_terms:
            return 0.0
        overlap = query_terms & artifact_terms
        return len(overlap) / max(1, min(len(query_terms), len(artifact_terms)))

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()
