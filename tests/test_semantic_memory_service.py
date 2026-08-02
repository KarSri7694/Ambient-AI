import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from application.services.semantic_memory_service import SemanticMemoryService
from core.models import SemanticMemoryChunk, SemanticMemoryResult


class _FakeMemory:
    def __init__(self):
        self.missing_batches = []
        self.updated = []
        self.deleted = []
        self.upserted = []
        self.vector_search_calls = []

    def get_chunks_missing_embeddings(self, limit=100):
        if self.missing_batches:
            return self.missing_batches.pop(0)
        return []

    def update_embedding(self, chunk_id, embedding):
        self.updated.append((chunk_id, embedding))

    def delete_semantic_chunk(self, chunk_id):
        self.deleted.append(chunk_id)

    def upsert_semantic_chunk(self, **kwargs):
        self.upserted.append(kwargs)

    def vector_search(self, query_embedding, *, limit=30, speaker_ids=None, source_types=None):
        self.vector_search_calls.append(
            {
                "query_embedding": query_embedding,
                "limit": limit,
                "speaker_ids": speaker_ids,
                "source_types": source_types,
            }
        )
        return [
            SemanticMemoryResult(
                chunk=SemanticMemoryChunk(
                    chunk_id="artifact:one",
                    source_type="artifact",
                    source_id="one",
                    source_ref="one.md",
                    speaker_id=None,
                    content="artifact one",
                ),
                vector_score=-0.1,
            )
        ]


class _FakeAdapter:
    def __init__(self):
        self.embed_calls = []
        self.rerank_calls = []

    def is_enabled(self):
        return True

    def embed_texts(self, texts):
        self.embed_calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]

    def rerank(self, *, query, documents, top_n=None):
        self.rerank_calls.append({"query": query, "documents": documents, "top_n": top_n})
        return [{"index": 0, "score": 0.75}]


def _chunk(chunk_id: str, content: str = "content"):
    return SemanticMemoryChunk(
        chunk_id=chunk_id,
        source_type="artifact",
        source_id=chunk_id.split(":", 1)[-1],
        source_ref="source",
        speaker_id=None,
        content=content,
    )


def test_ensure_embeddings_synced_can_be_bounded():
    memory = _FakeMemory()
    memory.missing_batches = [
        [_chunk("artifact:one", "one")],
        [_chunk("artifact:two", "two")],
    ]
    service = SemanticMemoryService(memory=memory, semantic_adapter=_FakeAdapter(), sync_batch_size=1)

    synced = service.ensure_embeddings_synced(max_batches=1)

    assert synced == 1
    assert memory.updated == [("artifact:one", [1.0, 0.0])]
    assert len(memory.missing_batches) == 1


def test_retrieve_passes_source_types_into_vector_search_and_uses_bounded_sync():
    memory = _FakeMemory()
    memory.missing_batches = [
        [_chunk("artifact:pending", "pending")],
        [_chunk("artifact:later", "later")],
    ]
    adapter = _FakeAdapter()
    service = SemanticMemoryService(memory=memory, semantic_adapter=adapter, sync_batch_size=1)

    results = service.retrieve(query="same lecture", source_types=["artifact"], limit=8, rerank_limit=3)

    assert results
    assert memory.vector_search_calls[-1]["source_types"] == ["artifact"]
    assert memory.vector_search_calls[-1]["limit"] == 8
    assert len(memory.updated) == 1
    assert len(memory.missing_batches) == 1
    assert adapter.rerank_calls[-1]["top_n"] == 1


def test_sync_removes_tool_payloads_and_persists_compact_factual_documents():
    memory = _FakeMemory()
    unsafe = _chunk("artifact:unsafe", '<function=fs_read>{"path":"secret"}')
    oversized = _chunk("artifact:oversized", "fact " * 500)
    memory.missing_batches = [[unsafe, oversized]]
    adapter = _FakeAdapter()
    service = SemanticMemoryService(memory=memory, semantic_adapter=adapter, sync_batch_size=2)

    assert service.ensure_embeddings_synced() == 1
    assert memory.deleted == ["artifact:unsafe"]
    assert len(memory.upserted) == 1
    assert len(memory.upserted[0]["content"]) <= service.MAX_DOCUMENT_CHARS
    assert adapter.embed_calls == [[memory.upserted[0]["content"]]]
