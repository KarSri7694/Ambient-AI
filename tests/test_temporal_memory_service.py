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
    now = datetime.now().replace(microsecond=0)
    first = service.record_ambient_event(
        _event(
            "one",
            (now - timedelta(minutes=20)).isoformat(),
            {"activity": "Comparing Qwen embedding models for a local retrieval system", "app_name": "Browser"},
        )
    )
    second = service.record_ambient_event(
        _event(
            "two",
            (now - timedelta(minutes=10)).isoformat(),
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


def test_temporal_records_are_not_embedded_unless_explicitly_promoted(tmp_path):
    memory = SQLiteMemoryAdapter(str(tmp_path / "memory.db"), str(tmp_path / "memory"))
    service = TemporalMemoryService(memory=memory)

    service.record_ambient_event(_event("tool", "2026-08-02T09:00:00", {"summary": "raw scraped page"}))
    assert memory.get_chunks_missing_embeddings(limit=20) == []

    service.record_ambient_event(
        _event("final", "2026-08-02T10:00:00", {"summary": "structured final result"}),
        semantic_index=True,
    )
    chunks = memory.get_chunks_missing_embeddings(limit=20)
    assert len(chunks) == 1
    assert chunks[0].source_type == "temporal_event"
    assert "structured final result" in chunks[0].content


def test_ambiguous_visual_event_never_injects_prior_thread_history(tmp_path):
    memory = SQLiteMemoryAdapter(str(tmp_path / "memory.db"), str(tmp_path / "memory"))
    service = TemporalMemoryService(memory=memory)
    service.record_ambient_event(
        _event("seed", "2026-08-02T10:00:00", {
            "activity": "Editing the ambient AI temporal service", "project_root": "D:/projects/ambient_ai",
        })
    )
    ambiguous = service.record_enriched_visual_event(
        _event("capture", "2026-08-02T10:05:00", {
            "activity": "Reading a generic browser page", "domain": "github.com", "analysis_status": "model",
            "continuation_relation": "unclear",
        })
    )

    assert ambiguous.thread_id is None
    context = service.build_context(query_text="generic browser page", current_event=ambiguous)
    assert context["ambiguous"] is True
    assert context["timeline"] == []


def test_specific_anchor_routes_but_generic_domain_does_not(tmp_path):
    memory = SQLiteMemoryAdapter(str(tmp_path / "memory.db"), str(tmp_path / "memory"))
    service = TemporalMemoryService(memory=memory)
    seed = service.record_ambient_event(
        _event("seed", "2026-08-02T10:00:00", {
            "activity": "Implement temporal routing", "project_root": "D:/projects/ambient_ai",
            "domain": "github.com",
        })
    )
    resolved = service.record_enriched_visual_event(
        _event("same-project", "2026-08-02T10:05:00", {
            "activity": "Implement temporal routing", "project_root": "D:/projects/ambient_ai",
            "analysis_status": "model", "continuation_relation": "continues",
        })
    )
    generic = service.record_enriched_visual_event(
        _event("generic-domain", "2026-08-02T10:06:00", {
            "activity": "Reading github", "domain": "github.com", "analysis_status": "model",
            "continuation_relation": "continues",
        })
    )

    assert resolved.thread_id == seed.thread_id
    assert generic.thread_id is None
