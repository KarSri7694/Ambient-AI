import importlib.util
import json
import sqlite3
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "repair_semantic_index.py"
SPEC = importlib.util.spec_from_file_location("repair_semantic_index", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_legacy_tool_and_unmarked_temporal_records_are_rejected():
    assert MODULE.removal_reason("delegated_task_result", "use this scraped page", {}) == "raw_delegated_tool_result"
    assert (
        MODULE.removal_reason("temporal_event", "normal event", {})
        == "legacy_temporal_record_not_marked_final_llm_json"
    )
    assert MODULE.removal_reason("artifact", '<function=fs_read>{}', {}) == "tool_or_system_prompt_payload"
    assert MODULE.removal_reason(
        "temporal_event", "{\"summary\": \"final\"}", {"semantic_origin": "final_llm_json"}
    ) is None


def test_canonical_text_preserves_words_and_respects_limit():
    value = "alpha " * 500
    compacted = MODULE.canonical_text(value, max_chars=1200)
    assert len(compacted) <= 1202
    assert compacted.endswith("...")
    assert "alpha" in compacted


def test_repair_removes_raw_legacy_data_and_compacts_retained_documents(tmp_path):
    database = tmp_path / "memory.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE semantic_memory_chunks (
                chunk_id TEXT PRIMARY KEY, source_type TEXT, source_id TEXT,
                source_ref TEXT, speaker_id TEXT, content TEXT, metadata_json TEXT,
                created_at TEXT, updated_at TEXT
            )
            """
        )
        records = [
            ("delegated_task_result:x", "delegated_task_result", "x", "raw tool result"),
            ("temporal_event:y", "temporal_event", "y", "legacy temporal data"),
            ("artifact:z", "artifact", "z", "fact " * 500),
        ]
        for chunk_id, source_type, source_id, content in records:
            connection.execute(
                "INSERT INTO semantic_memory_chunks VALUES (?, ?, ?, '', NULL, ?, ?, '', '')",
                (chunk_id, source_type, source_id, content, json.dumps({})),
            )

    report = MODULE.repair_database(database, max_chars=256, apply=True)
    assert report["removed_chunks"] == 2
    assert report["retained_chunks"] == 1
    with sqlite3.connect(database) as connection:
        rows = connection.execute("SELECT source_type, content, metadata_json FROM semantic_memory_chunks").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "artifact"
    assert len(rows[0][1]) <= 256
    assert json.loads(rows[0][2])["semantic_document_kind"] == "canonical_factual"
