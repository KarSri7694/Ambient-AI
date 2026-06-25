from abc import ABC, abstractmethod
from typing import List, Optional

from core.models import SemanticMemoryChunk, SemanticMemoryResult


class TextEmbeddingPort(ABC):
    """Port for local text embedding generation."""

    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        pass


class RerankerPort(ABC):
    """Port for cross-encoder style query/document reranking."""

    @abstractmethod
    async def rerank(self, query: str, documents: List[str]) -> List[float]:
        pass


class SemanticMemoryPort(ABC):
    """Port for storing and querying semantic memory chunks."""

    @abstractmethod
    def upsert_chunk(
        self,
        *,
        source_type: str,
        source_id: str,
        source_ref: str,
        content: str,
        speaker_id: Optional[str] = None,
        metadata_json: str = "{}",
    ) -> SemanticMemoryChunk:
        pass

    @abstractmethod
    def get_chunks_missing_embeddings(self, limit: int = 100) -> List[SemanticMemoryChunk]:
        pass

    @abstractmethod
    def update_embedding(self, chunk_id: str, embedding: List[float]) -> None:
        pass

    @abstractmethod
    def vector_search(
        self,
        query_embedding: List[float],
        *,
        limit: int = 30,
        speaker_ids: Optional[List[str]] = None,
    ) -> List[SemanticMemoryResult]:
        pass
