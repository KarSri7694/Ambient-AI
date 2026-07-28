import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from application.services.user_context_service import UserContextService
from core.models import SemanticMemoryChunk, SemanticMemoryResult


class _Memory:
    def __init__(self):
        self.user_info = "User is preparing Ambient AI for an AMD ROCm hackathon."
        self.working_memory = "User is currently improving proactive artifact organization."
        self.recent_context = "Legacy context: user was viewing a lecture."

    def get_user_info(self):
        return self.user_info

    def get_working_memory(self):
        return self.working_memory

    def get_recent_context(self):
        return self.recent_context


class _SemanticMemory:
    def __init__(self):
        self.retrieve_calls = []

    def retrieve(self, **kwargs):
        self.retrieve_calls.append(kwargs)
        return [
            SemanticMemoryResult(
                chunk=SemanticMemoryChunk(
                    chunk_id="user_info_note:1",
                    source_type="user_info_note",
                    source_id="1",
                    source_ref="user_info_note",
                    speaker_id=None,
                    content="User prefers local GPU inference and measurable benchmarks.",
                    metadata_json=json.dumps({"category": "preference"}),
                ),
                vector_score=0.8,
                rerank_score=0.9,
            )
        ]

    def format_context(self, results):
        return [
            {
                "source_type": result.chunk.source_type,
                "source_ref": result.chunk.source_ref,
                "content": result.chunk.content,
                "metadata": json.loads(result.chunk.metadata_json),
            }
            for result in results
        ]


def test_user_context_includes_biodata_working_memory_legacy_and_relevant_semantic_memory():
    semantic = _SemanticMemory()
    service = UserContextService(
        memory=_Memory(),
        semantic_memory=semantic,
        stable_profile_chars=200,
        working_memory_chars=200,
        semantic_limit=4,
        prompt_context_chars=2000,
    )

    text = service.build_prompt_context(query_text="ROCm benchmark artifact", include_semantic=True)

    assert "preparing Ambient AI" in text
    assert "proactive artifact organization" in text
    assert "local GPU inference" in text
    assert "Legacy context" in text
    assert semantic.retrieve_calls[-1]["source_types"] == [
        "user_info_note",
        "working_memory_note",
        "visual_user_fact",
        "memory_fact",
        "open_loop",
    ]


def test_user_context_respects_prompt_limit_and_can_disable_legacy_context():
    memory = _Memory()
    memory.user_info = "profile " * 100
    memory.working_memory = "memory " * 100
    service = UserContextService(
        memory=memory,
        include_recent_context_legacy=False,
        prompt_context_chars=120,
    )

    text = service.build_prompt_context(include_semantic=False)

    assert len(text) <= 135
    assert "Legacy context" not in text
    assert "[truncated]" in text
