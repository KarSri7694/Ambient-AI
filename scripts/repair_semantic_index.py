"""Remove non-semantic legacy records and rebuild a compact semantic corpus.

This is a one-time repair tool for databases created before Ambient separated
final LLM JSON from intermediate tool/browser output. It intentionally operates
on stored data instead of silently truncating requests at retrieval time.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import sqlite_vec
except ImportError:  # pragma: no cover - only matters when applying to a vec0 database
    sqlite_vec = None


RAW_SOURCE_TYPES = {"delegated_task_result"}
TOOL_OR_SYSTEM_MARKERS = (
    "<function=",
    "<tool_call",
    "</tool_call>",
    '"tool_calls"',
    '"tool_call_id"',
    '"role": "system"',
    '"role":"system"',
    "system prompt",
)


def parse_metadata(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def is_final_json_temporal(source_type: str, metadata: dict[str, Any]) -> bool:
    return source_type == "temporal_event" and metadata.get("semantic_origin") == "final_llm_json"


def removal_reason(source_type: str, content: str, metadata: dict[str, Any]) -> str | None:
    """Return why a legacy chunk is not permitted in vector retrieval."""
    if source_type in RAW_SOURCE_TYPES:
        return "raw_delegated_tool_result"
    if source_type == "temporal_event" and not is_final_json_temporal(source_type, metadata):
        return "legacy_temporal_record_not_marked_final_llm_json"
    lowered = content.lower()
    if any(marker in lowered for marker in TOOL_OR_SYSTEM_MARKERS):
        return "tool_or_system_prompt_payload"
    return None


def canonical_text(value: str, *, max_chars: int) -> str:
    """Create a compact factual document for vector storage, preserving words."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    # Stop at a natural word boundary. The detailed source remains in its own
    # durable store; this is intentionally only the retrieval representation.
    content_limit = max_chars - 4
    cutoff = text.rfind(" ", 0, content_limit)
    cutoff = cutoff if cutoff >= content_limit // 2 else content_limit
    return text[:cutoff].rstrip(" ,;:-") + " ..."


def connect_database(path: Path) -> sqlite3.Connection:
    """Open the database with sqlite-vec loaded so vec0 rows can be deleted."""
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    if sqlite_vec is not None:
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
    return connection


def repair_database(path: Path, *, max_chars: int, apply: bool) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Semantic database not found: {path}")
    if max_chars < 256:
        raise ValueError("max_chars must be at least 256")

    backup_path = path.with_suffix(path.suffix + f".pre-semantic-repair-{datetime.now():%Y%m%d-%H%M%S}.bak")
    connection = connect_database(path)
    try:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "semantic_memory_chunks" not in tables:
            raise RuntimeError("semantic_memory_chunks does not exist in this database")
        has_vectors = "semantic_memory_embeddings" in tables
        if apply and has_vectors and sqlite_vec is None:
            raise RuntimeError(
                "sqlite-vec is required to repair this database. Run this script with Ambient AI's virtual environment."
            )
        rows = connection.execute(
            "SELECT rowid, chunk_id, source_type, content, metadata_json FROM semantic_memory_chunks ORDER BY rowid"
        ).fetchall()
        removals: list[tuple[int, str]] = []
        rewrites: list[tuple[str, str, int]] = []
        retained = 0
        for row in rows:
            metadata = parse_metadata(row["metadata_json"])
            reason = removal_reason(row["source_type"], row["content"], metadata)
            if reason:
                removals.append((int(row["rowid"]), reason))
                continue
            compacted = canonical_text(row["content"], max_chars=max_chars)
            metadata["semantic_document_kind"] = "canonical_factual"
            metadata["semantic_rebuilt_at"] = datetime.now().isoformat()
            rewrites.append((compacted, json.dumps(metadata, ensure_ascii=False), int(row["rowid"])))
            retained += 1

        result = {
            "database": str(path),
            "apply": apply,
            "max_chars": max_chars,
            "total_chunks": len(rows),
            "retained_chunks": retained,
            "removed_chunks": len(removals),
            "removal_reasons": {reason: sum(1 for _, item_reason in removals if item_reason == reason)
                                for reason in sorted({reason for _, reason in removals})},
            "vectors_reset": retained if has_vectors else 0,
            "backup": str(backup_path) if apply else None,
        }
        if not apply:
            return result

        shutil.copy2(path, backup_path)
        connection.execute("BEGIN IMMEDIATE")
        if has_vectors:
            connection.executemany(
                "DELETE FROM semantic_memory_embeddings WHERE rowid = ?",
                [(rowid,) for rowid, _ in removals] + [(rowid,) for _, _, rowid in rewrites],
            )
        connection.executemany("DELETE FROM semantic_memory_chunks WHERE rowid = ?", [(rowid,) for rowid, _ in removals])
        connection.executemany(
            "UPDATE semantic_memory_chunks SET content = ?, metadata_json = ?, updated_at = CURRENT_TIMESTAMP WHERE rowid = ?",
            rewrites,
        )
        connection.commit()
        return result
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True, help="Path to memory.db")
    parser.add_argument("--max-chars", type=int, default=1200, help="Maximum stored retrieval-document length")
    parser.add_argument("--apply", action="store_true", help="Apply changes; omit for a dry-run report")
    args = parser.parse_args()
    print(json.dumps(repair_database(args.db, max_chars=args.max_chars, apply=args.apply), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
