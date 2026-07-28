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
    return datetime.now().isoformat(timespec="seconds")


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
                    content_hash TEXT
                )
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
        if row is None:
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
            return conn.execute("SELECT * FROM artifacts ORDER BY last_ai_edited_at DESC").fetchall()

    def _upsert_record(self, **values: Any) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO artifacts (
                    artifact_id, title, artifact_path, artifact_kind, short_summary,
                    detailed_summary, topics_json, source_count, created_at,
                    last_ai_edited_at, last_source_ref, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            if row is None:
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

    def _index_semantic_record(self, **values: Any) -> None:
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
            if self.semantic_sync_on_write and hasattr(semantic_memory, "ensure_embeddings_synced"):
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
