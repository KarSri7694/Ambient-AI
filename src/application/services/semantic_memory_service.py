import json
import logging
import math
from typing import List, Optional

from application.ports.memory_port import MemoryPort
from core.models import SemanticMemoryResult
from infrastructure.adapter.LlamaCppSemanticAdapter import LlamaCppSemanticAdapter


class SemanticMemoryService:
    """Keeps semantic embeddings in sync and retrieves relevant memory snippets."""

    def __init__(
        self,
        *,
        memory: MemoryPort,
        semantic_adapter: Optional[LlamaCppSemanticAdapter],
        sync_batch_size: int = 32,
        vector_limit: int = 12,
        rerank_limit: int = 6,
        logger: Optional[logging.Logger] = None,
    ):
        self.memory = memory
        self.semantic_adapter = semantic_adapter
        self.sync_batch_size = max(1, int(sync_batch_size))
        self.vector_limit = max(1, int(vector_limit))
        self.rerank_limit = max(1, int(rerank_limit))
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def is_enabled(self) -> bool:
        return self.semantic_adapter is not None and self.semantic_adapter.is_enabled()

    def ensure_embeddings_synced(self, *, max_batches: Optional[int] = None) -> int:
        if not self.is_enabled():
            return 0
        synced = 0
        batches = 0
        while True:
            if max_batches is not None and batches >= max(0, int(max_batches)):
                break
            chunks = self.memory.get_chunks_missing_embeddings(limit=self.sync_batch_size)
            if not chunks:
                break
            batches += 1
            texts = [chunk.content for chunk in chunks]
            try:
                embeddings = self.semantic_adapter.embed_texts(texts)
            except Exception as exc:
                self.logger.warning("Failed to sync semantic embeddings: %s", exc)
                break
            if len(embeddings) != len(chunks):
                self.logger.warning(
                    "Embedding count mismatch while syncing semantic memory: expected %s got %s.",
                    len(chunks),
                    len(embeddings),
                )
                break
            for chunk, embedding in zip(chunks, embeddings):
                if not embedding:
                    continue
                if not self._is_valid_embedding(embedding):
                    self.logger.warning(
                        "Skipping invalid semantic embedding for chunk %s: %s",
                        chunk.chunk_id,
                        self._embedding_diagnostics(embedding),
                    )
                    continue
                try:
                    self.memory.update_embedding(chunk.chunk_id, embedding)
                except (TypeError, ValueError) as exc:
                    self.logger.warning(
                        "Skipping invalid semantic embedding for chunk %s: %s",
                        chunk.chunk_id,
                        exc,
                    )
                    continue
                synced += 1
        return synced

    def retrieve(
        self,
        *,
        query: str,
        limit: Optional[int] = None,
        rerank_limit: Optional[int] = None,
        source_types: Optional[List[str]] = None,
        sync_max_batches: Optional[int] = 1,
        query_instruction: str = "",
    ) -> List[SemanticMemoryResult]:
        if not self.is_enabled():
            return []
        normalized_query = str(query).strip()
        if not normalized_query:
            return []
        self.ensure_embeddings_synced(max_batches=sync_max_batches)
        try:
            if query_instruction and hasattr(self.semantic_adapter, "embed_query"):
                query_embedding = [
                    self.semantic_adapter.embed_query(
                        normalized_query, instruction=query_instruction
                    )
                ]
            else:
                query_embedding = self.semantic_adapter.embed_texts([normalized_query])
        except Exception as exc:
            self.logger.warning("Semantic query embedding failed: %s", exc)
            return []
        if not query_embedding:
            return []
        if not self._is_valid_embedding(query_embedding[0]):
            self.logger.warning(
                "Semantic query embedding was invalid; skipping retrieval: %s",
                self._embedding_diagnostics(query_embedding[0]),
            )
            return []
        allowed_source_types = [str(item).strip() for item in (source_types or []) if str(item).strip()]
        results = self.memory.vector_search(
            query_embedding[0],
            limit=limit or self.vector_limit,
            source_types=allowed_source_types or None,
        )
        if allowed_source_types:
            allowed = set(allowed_source_types)
            results = [result for result in results if result.chunk.source_type in allowed]
        if not results:
            return []
        rerank_count = min(rerank_limit or self.rerank_limit, len(results))
        reranked = self.semantic_adapter.rerank(
            query=normalized_query,
            documents=[result.chunk.content for result in results],
            top_n=rerank_count,
        )
        if not reranked:
            return results[:rerank_count]
        ordered = []
        for item in reranked:
            index = int(item.get("index", -1))
            if index < 0 or index >= len(results):
                continue
            result = results[index]
            ordered.append(
                SemanticMemoryResult(
                    chunk=result.chunk,
                    vector_score=result.vector_score,
                    rerank_score=float(item.get("score", 0.0)),
                )
            )
        return ordered

    def format_context(self, results: List[SemanticMemoryResult]) -> List[dict]:
        formatted: List[dict] = []
        for result in results:
            metadata = self._parse_metadata(result.chunk.metadata_json)
            formatted.append(
                {
                    "source_type": result.chunk.source_type,
                    "source_ref": result.chunk.source_ref,
                    "content": result.chunk.content,
                    "vector_score": result.vector_score,
                    "rerank_score": result.rerank_score,
                    "metadata": metadata,
                }
            )
        return formatted

    def _parse_metadata(self, raw: str) -> dict:
        try:
            payload = json.loads(raw or "{}")
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _is_valid_embedding(self, embedding: List[float]) -> bool:
        if not embedding:
            return False
        try:
            values = [float(value) for value in embedding]
        except (TypeError, ValueError):
            return False
        if not all(math.isfinite(value) for value in values):
            return False
        return any(abs(value) > 1e-12 for value in values)

    def _embedding_diagnostics(self, embedding: List[float]) -> str:
        try:
            values = [float(value) for value in embedding]
        except (TypeError, ValueError):
            return f"length={len(embedding) if embedding is not None else 0}, non_numeric=true"
        finite_values = [value for value in values if math.isfinite(value)]
        nonzero_count = sum(1 for value in finite_values if abs(value) > 1e-12)
        norm = math.sqrt(sum(value * value for value in finite_values)) if finite_values else 0.0
        return (
            f"length={len(values)}, finite={len(finite_values)}, "
            f"nonzero={nonzero_count}, norm={norm:.6g}"
        )
