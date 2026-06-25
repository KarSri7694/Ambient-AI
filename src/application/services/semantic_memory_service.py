import logging
from dataclasses import replace
from typing import List, Optional

from application.ports.semantic_memory_port import (
    RerankerPort,
    SemanticMemoryPort,
    TextEmbeddingPort,
)
from core.models import SemanticMemoryResult


class SemanticMemoryService:
    """Embed, retrieve, rerank, and format relevant semantic memory."""

    def __init__(
        self,
        *,
        memory: SemanticMemoryPort,
        embedder: TextEmbeddingPort,
        reranker: RerankerPort | None = None,
        logger: logging.Logger | None = None,
    ):
        self.memory = memory
        self.embedder = embedder
        self.reranker = reranker
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    async def retrieve(
        self,
        query: str,
        *,
        speaker_ids: Optional[List[str]] = None,
        candidate_limit: int = 30,
        result_limit: int = 8,
        refresh_limit: int = 100,
    ) -> List[SemanticMemoryResult]:
        query = " ".join(query.split())
        if not query:
            return []
        await self.refresh_missing_embeddings(limit=refresh_limit)
        try:
            query_embedding = (await self.embedder.embed_texts([query]))[0]
        except Exception as exc:
            self.logger.warning("Semantic query embedding failed: %s", exc)
            return []

        candidates = self.memory.vector_search(
            query_embedding,
            limit=candidate_limit,
            speaker_ids=speaker_ids,
        )
        if not candidates:
            return []

        if self.reranker is None:
            return candidates[:result_limit]

        try:
            scores = await self.reranker.rerank(
                query=query,
                documents=[candidate.chunk.content for candidate in candidates],
            )
        except Exception as exc:
            self.logger.warning("Semantic reranking failed; using vector order: %s", exc)
            return candidates[:result_limit]

        reranked = [
            replace(candidate, rerank_score=score)
            for candidate, score in zip(candidates, scores)
        ]
        reranked.sort(
            key=lambda result: (
                result.rerank_score if result.rerank_score is not None else result.vector_score
            ),
            reverse=True,
        )
        return reranked[:result_limit]

    async def refresh_missing_embeddings(self, *, limit: int = 100) -> int:
        chunks = self.memory.get_chunks_missing_embeddings(limit=limit)
        if not chunks:
            return 0
        try:
            embeddings = await self.embedder.embed_texts([chunk.content for chunk in chunks])
        except Exception as exc:
            self.logger.warning("Semantic chunk embedding refresh failed: %s", exc)
            return 0
        updated = 0
        for chunk, embedding in zip(chunks, embeddings):
            self.memory.update_embedding(chunk.chunk_id, embedding)
            updated += 1
        return updated

    def format_prompt_section(self, results: List[SemanticMemoryResult]) -> str:
        if not results:
            return ""
        lines = ["## Relevant Semantic Memory"]
        for result in results:
            chunk = result.chunk
            score = (
                result.rerank_score
                if result.rerank_score is not None
                else result.vector_score
            )
            source = chunk.source_ref or chunk.source_id
            lines.append(
                f"- [{chunk.source_type} | score {score:.3f} | {source}] {chunk.content}"
            )
        return "\n".join(lines)
