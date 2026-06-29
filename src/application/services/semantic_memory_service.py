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

    def ensure_embeddings_synced(self) -> int:
        if not self.is_enabled():
            return 0
        synced = 0
        while True:
            chunks = self.memory.get_chunks_missing_embeddings(limit=self.sync_batch_size)
            if not chunks:
                break
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
                        "Skipping invalid semantic embedding for chunk %s.",
                        chunk.chunk_id,
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
    ) -> List[SemanticMemoryResult]:
        if not self.is_enabled():
            return []
        normalized_query = str(query).strip()
        if not normalized_query:
            return []
        self.ensure_embeddings_synced()
        try:
            query_embedding = self.semantic_adapter.embed_texts([normalized_query])
        except Exception as exc:
            self.logger.warning("Semantic query embedding failed: %s", exc)
            return []
        if not query_embedding:
            return []
        if not self._is_valid_embedding(query_embedding[0]):
            self.logger.warning("Semantic query embedding was invalid; skipping retrieval.")
            return []
        results = self.memory.vector_search(
            query_embedding[0],
            limit=limit or self.vector_limit,
        )
        if source_types:
            allowed = {str(item).strip() for item in source_types if str(item).strip()}
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
        ordered: List[SemanticMemoryResult] = []
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
            return all(math.isfinite(float(value)) for value in embedding)
        except (TypeError, ValueError):
            return False
