#!/usr/bin/env python
"""Check whether Ambient AI semantic memory is working end-to-end.

This script uses the real project SemanticMemoryService against a temporary
SQLite memory database. It does not modify the user's production memory.

It verifies:
- embedding endpoint returns finite, non-zero vectors
- pending semantic chunks can be embedded and stored
- vector retrieval returns relevant chunks
- reranker endpoint is callable through SemanticMemoryService
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import requests


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from application.services.semantic_memory_service import SemanticMemoryService
from config import CONFIG
from infrastructure.adapter.LlamaCppSemanticAdapter import LlamaCppSemanticAdapter
from infrastructure.adapter.SQLiteMemoryAdapter import SQLiteMemoryAdapter


def _diag(values: list[float]) -> dict:
    finite_values: list[float] = []
    bad = 0
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            bad += 1
            continue
        if math.isfinite(number):
            finite_values.append(number)
        else:
            bad += 1
    return {
        "length": len(values),
        "finite": len(finite_values),
        "bad": bad,
        "nonzero": sum(1 for value in finite_values if abs(value) > 1e-12),
        "norm": math.sqrt(sum(value * value for value in finite_values)) if finite_values else 0.0,
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Ambient AI semantic memory service.")
    parser.add_argument("--keep-temp", action="store_true", help="Do not delete the temporary test database.")
    parser.add_argument("--query", default="Which note is about Radeon ROCm inference optimization?")
    args = parser.parse_args()

    enabled = CONFIG.get_bool("semantic_memory", "enabled", False)
    embedding_base_url = CONFIG.get_str("semantic_memory", "embedding_api_base_url", "http://127.0.0.1:8081")
    embedding_model = CONFIG.get_model("embedding_model", "", section="semantic_memory")
    reranker_base_url = CONFIG.get_str("semantic_memory", "reranker_api_base_url", embedding_base_url)
    reranker_model = CONFIG.get_model("reranker_model", "", section="semantic_memory")
    timeout_seconds = CONFIG.get_float("semantic_memory", "timeout_seconds", 60.0)
    vector_limit = CONFIG.get_int("semantic_memory", "vector_limit", 12)
    rerank_limit = CONFIG.get_int("semantic_memory", "rerank_limit", 6)
    sync_batch_size = CONFIG.get_int("semantic_memory", "sync_batch_size", 32)

    print("Semantic memory config:")
    print(json.dumps(
        {
            "enabled": enabled,
            "embedding_base_url": embedding_base_url,
            "embedding_model": embedding_model,
            "reranker_base_url": reranker_base_url,
            "reranker_model": reranker_model,
            "timeout_seconds": timeout_seconds,
            "vector_limit": vector_limit,
            "rerank_limit": rerank_limit,
            "sync_batch_size": sync_batch_size,
        },
        indent=2,
    ))

    _require(enabled, "semantic_memory.enabled is false")
    _require(bool(embedding_model), "semantic_memory.embedding_model is blank")

    adapter = LlamaCppSemanticAdapter(
        embedding_base_url=embedding_base_url,
        embedding_model=embedding_model,
        reranker_base_url=reranker_base_url,
        reranker_model=reranker_model,
        timeout_seconds=timeout_seconds,
    )

    print("\n1. Checking raw embedding endpoint...")
    probe = adapter.embed_texts(["Ambient AI semantic memory health check."])
    _require(bool(probe), "embedding endpoint returned no vectors")
    probe_diag = _diag(probe[0])
    print(json.dumps(probe_diag, indent=2))
    _require(probe_diag["bad"] == 0, "embedding vector contains non-finite/non-numeric values")
    _require(probe_diag["nonzero"] > 0, "embedding vector is all-zero")

    temp_dir = Path(tempfile.mkdtemp(prefix="ambient-semantic-check-"))
    try:
        memory = SQLiteMemoryAdapter(
            db_path=str(temp_dir / "database" / "memory.db"),
            memory_root=str(temp_dir / "memory"),
        )
        service = SemanticMemoryService(
            memory=memory,
            semantic_adapter=adapter,
            sync_batch_size=sync_batch_size,
            vector_limit=vector_limit,
            rerank_limit=rerank_limit,
        )

        print("\n2. Inserting temporary semantic chunks...")
        fixtures = [
            (
                "test:rocm",
                "ROCm inference optimization on Radeon GPUs benefits from Flash Attention, tuned batch size, tuned ubatch size, and MTP draft depth one.",
            ),
            (
                "test:cooking",
                "A sourdough starter needs regular feeding with flour and water before baking bread.",
            ),
            (
                "test:calendar",
                "Calendar planning notes should preserve dates, deadlines, and reminders clearly.",
            ),
        ]
        for chunk_id, content in fixtures:
            memory.upsert_semantic_chunk(
                source_type="semantic_health_check",
                source_id=chunk_id,
                source_ref="scripts/check_semantic_service.py",
                content=content,
                metadata_json=json.dumps({"temporary": True}),
            )

        print("\n3. Syncing embeddings through SemanticMemoryService...")
        synced = service.ensure_embeddings_synced(max_batches=3)
        print(f"synced={synced}")
        _require(synced == len(fixtures), f"expected {len(fixtures)} embedded chunks, got {synced}")

        print("\n4. Retrieving through SemanticMemoryService...")
        results = service.retrieve(
            query=args.query,
            limit=3,
            rerank_limit=3,
            source_types=["semantic_health_check"],
            sync_max_batches=0,
        )
        formatted = service.format_context(results)
        print(json.dumps(formatted, indent=2, ensure_ascii=False))
        _require(bool(results), "retrieval returned no results")
        top = results[0]
        _require(
            top.chunk.source_id == "test:rocm",
            f"top result was {top.chunk.source_id!r}, expected 'test:rocm'",
        )
        _require(
            top.rerank_score is not None or not reranker_model,
            "reranker_model is configured but no rerank_score was attached",
        )

        print("\nPASS: semantic memory embedding, vector search, and reranking are working.")
        return 0
    finally:
        if args.keep_temp:
            print(f"\nKept temp directory: {temp_dir}")
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except requests.RequestException as exc:
        print(f"\nFAIL: semantic endpoint request failed: {exc}", file=sys.stderr)
        print(
            "Start the semantic server, for example:\n"
            "  llama-server --models-preset d:\\Projects\\ambient_ai\\models_preset.ini --port 8081",
            file=sys.stderr,
        )
        raise SystemExit(1)
    except Exception as exc:
        print(f"\nFAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
