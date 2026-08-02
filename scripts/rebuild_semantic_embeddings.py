#!/usr/bin/env python
"""Embed every pending semantic-memory document using Ambient's configured server."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from application.services.semantic_memory_service import SemanticMemoryService
from config import CONFIG
from infrastructure.adapter.LlamaCppSemanticAdapter import LlamaCppSemanticAdapter
from infrastructure.adapter.SQLiteMemoryAdapter import SQLiteMemoryAdapter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True, help="Path to memory.db")
    parser.add_argument("--memory-root", type=Path, required=True, help="Ambient memory directory")
    parser.add_argument(
        "--verify-query",
        default="recent user work and unresolved blockers",
        help="Run a post-rebuild retrieval and rerank check using this query",
    )
    args = parser.parse_args()

    adapter = LlamaCppSemanticAdapter(
        embedding_base_url=CONFIG.get_str("semantic_memory", "embedding_api_base_url", "http://127.0.0.1:8081"),
        embedding_model=CONFIG.get_model("embedding_model", "", section="semantic_memory"),
        reranker_base_url=CONFIG.get_str("semantic_memory", "reranker_api_base_url", "http://127.0.0.1:8081"),
        reranker_model=CONFIG.get_model("reranker_model", "", section="semantic_memory"),
        timeout_seconds=CONFIG.get_float("semantic_memory", "timeout_seconds", 60.0),
    )
    memory = SQLiteMemoryAdapter(str(args.db), str(args.memory_root))
    service = SemanticMemoryService(
        memory=memory,
        semantic_adapter=adapter,
        sync_batch_size=CONFIG.get_int("semantic_memory", "sync_batch_size", 32),
        vector_limit=CONFIG.get_int("semantic_memory", "vector_limit", 12),
        rerank_limit=CONFIG.get_int("semantic_memory", "rerank_limit", 6),
    )
    if not service.is_enabled():
        raise RuntimeError("Semantic embedding is not configured or enabled.")
    synced = service.ensure_embeddings_synced()
    remaining = len(memory.get_chunks_missing_embeddings(limit=1))
    print(f"Embedded {synced} canonical semantic documents. Remaining: {remaining}.")
    if remaining:
        return 1
    results = service.retrieve(query=args.verify_query, limit=6, rerank_limit=6, sync_max_batches=0)
    if not results:
        raise RuntimeError("No retrieval results returned after rebuild.")
    print(json.dumps(
        [
            {
                "source_type": item.chunk.source_type,
                "content_length": len(item.chunk.content),
                "rerank_score": item.rerank_score,
            }
            for item in results
        ],
        indent=2,
    ))
    if adapter.reranker_model and all(item.rerank_score is None for item in results):
        raise RuntimeError("Reranker is configured but returned no rerank scores.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
