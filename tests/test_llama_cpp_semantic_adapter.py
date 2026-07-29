import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from infrastructure.adapter.LlamaCppSemanticAdapter import LlamaCppSemanticAdapter


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _FakeGuard:
    def __init__(self):
        self.operations = []

    def semantic_operation(self, operation):
        self.operations.append(operation)
        return self

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):
        return False


def test_embedding_call_runs_inside_semantic_guard(monkeypatch):
    guard = _FakeGuard()
    adapter = LlamaCppSemanticAdapter(
        embedding_base_url="http://semantic",
        embedding_model="EmbeddingModel",
        semantic_model_guard=guard,
    )

    def fake_post(url, json, timeout):
        assert url == "http://semantic/v1/embeddings"
        assert json["model"] == "EmbeddingModel"
        return _FakeResponse({"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    monkeypatch.setattr("infrastructure.adapter.LlamaCppSemanticAdapter.requests.post", fake_post)

    assert adapter.embed_texts(["hello"]) == [[1.0, 0.0]]
    assert guard.operations == ["embedding"]


def test_rerank_call_runs_inside_semantic_guard(monkeypatch):
    guard = _FakeGuard()
    adapter = LlamaCppSemanticAdapter(
        embedding_base_url="http://semantic",
        embedding_model="EmbeddingModel",
        reranker_base_url="http://semantic",
        reranker_model="RerankerModel",
        semantic_model_guard=guard,
    )

    def fake_post(url, json, timeout):
        assert url == "http://semantic/v1/rerank"
        assert json["model"] == "RerankerModel"
        assert json["documents"] == ["rocm note", "bread note"]
        return _FakeResponse({"results": [{"index": 0, "relevance_score": 0.9}]})

    monkeypatch.setattr("infrastructure.adapter.LlamaCppSemanticAdapter.requests.post", fake_post)

    results = adapter.rerank(query="rocm", documents=["rocm note", "bread note"], top_n=1)

    assert results == [{"index": 0, "score": 0.9, "document": "rocm note"}]
    assert guard.operations == ["rerank"]
