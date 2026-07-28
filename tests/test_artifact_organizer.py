import json
import sys
from pathlib import Path
import asyncio

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from application.services.artifact_organizer_service import ArtifactOrganizer
from application.services.llm_interaction_service import LLMInteractionService
from core.models import SemanticMemoryChunk, SemanticMemoryResult


class _Delta:
    def __init__(self, content):
        self.content = content
        self.reasoning_content = None
        self.tool_calls = None


class _Choice:
    def __init__(self, content):
        self.delta = _Delta(content)


class _Chunk:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _SequencedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def chat_completion_stream(self, *args, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)

        async def _gen():
            yield _Chunk(response)

        return _gen()


class _NoTools:
    async def get_all_tools(self):
        return []


class _FakeSemanticBackingStore:
    def __init__(self):
        self.upserts = []

    def upsert_semantic_chunk(self, **kwargs):
        self.upserts.append(kwargs)


class _FakeSemanticMemory:
    def __init__(self, artifact_ids=None):
        self.memory = _FakeSemanticBackingStore()
        self.artifact_ids = list(artifact_ids or [])
        self.queries = []
        self.sync_kwargs = []
        self.synced = 0

    def is_enabled(self):
        return True

    def ensure_embeddings_synced(self, **kwargs):
        self.sync_kwargs.append(kwargs)
        self.synced += 1
        return 0

    def retrieve(self, **kwargs):
        self.queries.append(kwargs)
        return [
            SemanticMemoryResult(
                chunk=SemanticMemoryChunk(
                    chunk_id=f"artifact:{artifact_id}",
                    source_type="artifact",
                    source_id=artifact_id,
                    source_ref=f"artifact-{index}.md",
                    speaker_id=None,
                    content="semantic artifact summary",
                ),
                vector_score=0.9 - (index * 0.01),
                rerank_score=0.8 - (index * 0.01),
            )
            for index, artifact_id in enumerate(self.artifact_ids)
        ]


def test_artifact_organizer_creates_standard_artifact(tmp_path):
    organizer = ArtifactOrganizer(tmp_path, candidate_summary_words=8)

    result = organizer.save_new(
        title="NPTEL LLM Lecture",
        summary="Lecture about LLM functions and token prediction.",
        detailed_report="The lecture explains how LLMs predict tokens from context.",
        source_ref="test/one",
    )

    path = Path(result["artifact_path"])
    assert result["artifact_action"] == "created"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "# NPTEL LLM Lecture" in text
    assert "_Last AI edit:" in text
    assert "## Short Summary" in text
    assert "## Detailed Summary" in text
    assert "## Content" in text
    assert "## Update Log" in text


def test_artifact_organizer_ranks_related_candidate(tmp_path):
    organizer = ArtifactOrganizer(tmp_path, candidate_summary_words=50)
    organizer.save_new(
        title="NPTEL LLM Lecture",
        summary="Lecture notes about LLM functions, token prediction, attention, and context windows.",
        detailed_report="Initial LLM lecture notes.",
        source_ref="test/one",
    )

    candidates = organizer.candidates_for(
        title="NPTEL lecture attention",
        summary="The same lecture now explains attention and context windows.",
        detailed_report="Attention lets the model relate tokens across the context window.",
    )

    assert candidates
    assert candidates[0].title == "NPTEL LLM Lecture"


def test_artifact_organizer_indexes_artifacts_into_semantic_memory(tmp_path):
    semantic_memory = _FakeSemanticMemory()
    organizer = ArtifactOrganizer(tmp_path, semantic_memory=semantic_memory)

    result = organizer.save_new(
        title="NPTEL LLM Lecture",
        summary="Lecture about LLM functions and token prediction.",
        detailed_report="The lecture explains how LLMs predict tokens from context.",
        source_ref="test/semantic",
    )

    assert semantic_memory.memory.upserts
    upsert = semantic_memory.memory.upserts[-1]
    assert upsert["source_type"] == "artifact"
    assert upsert["source_id"] == result["artifact_id"]
    assert "NPTEL LLM Lecture" in upsert["content"]
    assert semantic_memory.synced >= 1
    assert semantic_memory.sync_kwargs[-1] == {"max_batches": 1}


def test_artifact_organizer_uses_semantic_candidates_when_lexical_overlap_is_weak(tmp_path):
    semantic_memory = _FakeSemanticMemory()
    organizer = ArtifactOrganizer(tmp_path, semantic_memory=semantic_memory)
    created = organizer.save_new(
        title="NPTEL LLM Lecture",
        summary="Lecture notes about transformers and autoregressive language models.",
        detailed_report="The lecture discusses token prediction and attention.",
        source_ref="test/one",
    )
    semantic_memory.artifact_ids = [created["artifact_id"]]

    candidates = organizer.candidates_for(
        title="Class video notes",
        summary="The screen explains neural sequence predictors.",
        detailed_report="The new screenshot uses different wording but is the same lecture thread.",
    )

    assert candidates
    assert candidates[0].artifact_id == created["artifact_id"]
    assert "semantic" in candidates[0].match_source
    assert semantic_memory.queries[-1]["source_types"] == ["artifact"]


def test_artifact_organizer_merges_existing_and_updates_registry(tmp_path):
    organizer = ArtifactOrganizer(tmp_path, candidate_summary_words=50)
    created = organizer.save_new(
        title="NPTEL LLM Lecture",
        summary="Lecture notes about LLM functions.",
        detailed_report="LLMs predict next tokens.",
        source_ref="test/one",
    )

    decision = {
        "action": "merge_existing",
        "target_artifact_id": created["artifact_id"],
        "final_title": "NPTEL LLM Lecture",
        "updated_short_summary": "Lecture notes about LLM functions, attention, and context windows.",
        "updated_detailed_summary": "The artifact covers LLM token prediction, attention, and context windows.",
        "merged_content": "LLMs predict next tokens.\n\nAttention helps tokens interact across long context windows.",
        "dedupe_notes": ["Skipped duplicate token prediction explanation."],
        "reason": "Same NPTEL lecture topic.",
    }
    result = organizer.apply_decision(
        decision=decision,
        fallback_title="NPTEL lecture attention",
        fallback_summary="Attention and context windows.",
        fallback_detailed_report="Attention helps long context.",
        source_ref="test/two",
    )

    assert result["artifact_action"] == "merged"
    assert result["artifact_path"] == created["artifact_path"]
    text = Path(result["artifact_path"]).read_text(encoding="utf-8")
    assert "Attention helps tokens interact" in text
    assert "Skipped duplicate" not in text
    assert "test/two" in text


def test_artifact_organizer_backfills_existing_markdown(tmp_path):
    existing = tmp_path / "Existing.md"
    existing.write_text(
        "# Existing Notes\n\n## Summary\nShort old summary.\n\n## Detailed Report\nOld body.\n",
        encoding="utf-8",
    )

    organizer = ArtifactOrganizer(tmp_path)
    candidates = organizer.candidates_for(
        title="Existing notes",
        summary="old summary",
        detailed_report="old body",
    )

    assert candidates
    assert candidates[0].title == "Existing Notes"


def test_llm_interaction_service_merges_same_topic_artifact(tmp_path):
    first_report = json.dumps({
        "title": "NPTEL LLM Lecture",
        "summary": "Lecture notes about LLM functions.",
        "detailed_report": "LLMs predict next tokens.",
    })
    second_report = json.dumps({
        "title": "NPTEL LLM Lecture",
        "summary": "Lecture notes about attention in LLMs.",
        "detailed_report": "Attention helps tokens interact across context windows.",
    })
    merge_decision = json.dumps({
        "action": "merge_existing",
        "target_artifact_id": "",
        "final_title": "NPTEL LLM Lecture",
        "updated_short_summary": "Lecture notes about LLM token prediction and attention.",
        "updated_detailed_summary": "The artifact covers token prediction and attention across context windows.",
        "merged_content": "LLMs predict next tokens.\n\nAttention helps tokens interact across context windows.",
        "dedupe_notes": ["Skipped duplicate lecture title."],
        "reason": "Same NPTEL LLM lecture.",
    })
    llm = _SequencedLLM([first_report, second_report, merge_decision])
    service = LLMInteractionService(
        llm_provider=llm,
        tool_bridge=_NoTools(),
        reporter_model="reporter",
        artifact_root=str(tmp_path),
        artifact_organizer_enabled=True,
    )

    first = asyncio.run(service._build_user_report(
        model="model",
        user_input="first",
        final_response="first",
        tools_used=[],
        source_name="test",
    ))
    candidates = service.artifact_organizer.candidates_for(
        title="NPTEL LLM Lecture",
        summary="Lecture notes about attention in LLMs.",
        detailed_report="Attention helps tokens interact across context windows.",
    )
    assert candidates
    decision = json.loads(merge_decision)
    decision["target_artifact_id"] = candidates[0].artifact_id
    llm.responses[-1] = json.dumps(decision)
    second = asyncio.run(service._build_user_report(
        model="model",
        user_input="second",
        final_response="second",
        tools_used=[],
        source_name="test",
    ))

    assert first["artifact_path"] == second["artifact_path"]
    assert second["artifact_action"] == "merged"
    text = Path(second["artifact_path"]).read_text(encoding="utf-8")
    assert "Attention helps tokens interact" in text
