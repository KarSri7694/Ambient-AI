import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from application.services.temporal_memory_service import TemporalMemoryService
from core.models import SemanticMemoryChunk, SemanticMemoryResult
from infrastructure.adapter.LlamaCppSemanticAdapter import LlamaCppSemanticAdapter
from infrastructure.adapter.SQLiteMemoryAdapter import SQLiteMemoryAdapter


class _SemanticResults:
    def __init__(self, memory):
        self.memory = memory
        self.calls = []

    def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        rows = self.memory.get_temporal_events(limit=100)
        return [
            SemanticMemoryResult(
                chunk=SemanticMemoryChunk(
                    chunk_id=f"temporal_event:{row.temporal_event_id}",
                    source_type="temporal_event",
                    source_id=row.temporal_event_id,
                    source_ref=row.source_ref,
                    speaker_id=None,
                    content=row.content,
                    metadata_json=json.dumps(
                        {
                            "temporal_event_id": row.temporal_event_id,
                            "thread_id": row.thread_id,
                            "occurred_at": row.occurred_at,
                        }
                    ),
                ),
                vector_score=1.0,
                rerank_score=1.0,
            )
            for row in rows
        ]


def _event(event_id, occurred_at, payload):
    return SimpleNamespace(
        event_id=event_id,
        event_type="visual_context_changed",
        source_kind="passive_observer",
        source_ref=f"capture://{event_id}",
        occurred_at=occurred_at,
        payload_json=json.dumps(payload),
        confidence=0.8,
    )


def test_work_retrieval_is_temporal_post_ranked_and_chronological(tmp_path):
    memory = SQLiteMemoryAdapter(str(tmp_path / "memory.db"), str(tmp_path / "memory"))
    semantic = _SemanticResults(memory)
    service = TemporalMemoryService(memory=memory, semantic_memory=semantic)
    first = service.record_ambient_event(
        _event(
            "one",
            "2026-08-02T09:00:00",
            {"activity": "Comparing Qwen embedding models for a local retrieval system", "app_name": "Browser"},
        )
    )
    second = service.record_ambient_event(
        _event(
            "two",
            "2026-08-02T10:00:00",
            {"activity": "Validated the Qwen embedding integration in the local retrieval system", "app_name": "VS Code"},
        )
    )

    context = service.build_context(query_text="Qwen embedding retrieval integration", current_event=second)

    assert semantic.calls[-1]["query_instruction"] == service.work_retrieval_instruction
    assert context["active_thread"]["thread_id"] == first.thread_id == second.thread_id
    assert [row["temporal_event_id"] for row in context["timeline"]] == [
        first.temporal_event_id,
        second.temporal_event_id,
    ]
    assert context["active_thread"]["state"] == "completed"
    assert "do not repeat" in context["suppression_hint"].lower()


def test_old_temporal_detail_is_compacted_into_searchable_summary(tmp_path):
    memory = SQLiteMemoryAdapter(str(tmp_path / "memory.db"), str(tmp_path / "memory"))
    service = TemporalMemoryService(memory=memory, detailed_retention_days=30)
    old = (datetime.now() - timedelta(days=31)).replace(microsecond=0).isoformat()
    service.record_ambient_event(
        _event("old", old, {"activity": "Investigated a dependency compatibility issue"})
    )

    assert service.consolidate_expired() == 1
    rows = memory.get_temporal_events(limit=20)
    assert any(row.source_type == "temporal_summary" for row in rows)
    assert any(row.state == "consolidated" for row in rows)


def test_qwen_instruction_is_applied_to_query_not_document():
    adapter = LlamaCppSemanticAdapter(
        embedding_base_url="http://example.invalid",
        embedding_model="QwenEmbedding",
    )
    captured = []
    adapter.embed_texts = lambda texts: (captured.append(texts), [[0.1, 0.2]])[1]

    result = adapter.embed_query("current task", instruction="Retrieve relevant work evidence")

    assert result == [0.1, 0.2]
    assert captured == [["Instruct: Retrieve relevant work evidence\nQuery: current task"]]
