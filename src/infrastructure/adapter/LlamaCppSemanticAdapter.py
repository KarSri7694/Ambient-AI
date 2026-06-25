import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

import requests

from application.ports.semantic_memory_port import RerankerPort, TextEmbeddingPort
from infrastructure.adapter.llamaCppAdapter import LlamaCppAdapter


class _TemporaryLlamaModel:
    """Run a short operation with a temporary llama.cpp model loaded."""

    def __init__(
        self,
        *,
        llm: LlamaCppAdapter,
        model_path: str,
        messages_provider: Callable[[], List[Dict[str, Any]]] | None = None,
        logger: logging.Logger | None = None,
    ):
        self.llm = llm
        self.model_path = model_path
        self.messages_provider = messages_provider
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    async def run(self, operation):
        previous_model = self.llm.get_current_model()
        saved_state = None
        messages = self.messages_provider() if self.messages_provider else []
        if previous_model is not None:
            saved_state = await self.llm.save_and_unload(messages)
        try:
            await self.llm.load_model(self.model_path)
            return await operation()
        finally:
            await self.llm.unload_model()
            if previous_model is None:
                return
            if saved_state is not None:
                await self.llm.load_and_restore()
            else:
                self.logger.warning(
                    "Temporary model operation could not save KV state; reloading previous model without restore."
                )
                await self.llm.load_model(previous_model)


class LlamaCppEmbeddingAdapter(TextEmbeddingPort):
    """Text embeddings through llama.cpp's OpenAI-compatible embedding endpoint."""

    def __init__(
        self,
        *,
        llm: LlamaCppAdapter,
        model_path: str,
        messages_provider: Callable[[], List[Dict[str, Any]]] | None = None,
    ):
        self.llm = llm
        self.model_path = model_path
        self.temporary_model = _TemporaryLlamaModel(
            llm=llm,
            model_path=model_path,
            messages_provider=messages_provider,
            logger=logging.getLogger(self.__class__.__name__),
        )

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        async def _operation():
            return await asyncio.to_thread(self._request_embeddings, texts)

        return await self.temporary_model.run(_operation)

    def _request_embeddings(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(
            f"{self.llm.api_uri_v1}/embeddings",
            json={"model": self.model_path, "input": texts},
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", [])
        embeddings: List[List[float]] = []
        for item in data:
            embedding = item.get("embedding", [])
            embeddings.append([float(value) for value in embedding])
        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"Expected {len(texts)} embeddings, received {len(embeddings)}."
            )
        return embeddings


class LlamaCppRerankerAdapter(RerankerPort):
    """Jina GGUF reranking through llama.cpp rerank-compatible endpoints."""

    def __init__(
        self,
        *,
        llm: LlamaCppAdapter,
        model_path: str,
        messages_provider: Callable[[], List[Dict[str, Any]]] | None = None,
    ):
        self.llm = llm
        self.model_path = model_path
        self.temporary_model = _TemporaryLlamaModel(
            llm=llm,
            model_path=model_path,
            messages_provider=messages_provider,
            logger=logging.getLogger(self.__class__.__name__),
        )

    async def rerank(self, query: str, documents: List[str]) -> List[float]:
        if not documents:
            return []

        async def _operation():
            return await asyncio.to_thread(self._request_rerank, query, documents)

        return await self.temporary_model.run(_operation)

    def _request_rerank(self, query: str, documents: List[str]) -> List[float]:
        payload = {
            "model": self.model_path,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
        }
        errors: List[str] = []
        for endpoint in ("/v1/rerank", "/rerank", "/v1/reranking", "/reranking"):
            try:
                response = requests.post(
                    f"{self.llm.base_url}{endpoint}",
                    json=payload,
                    timeout=120,
                )
                if response.status_code == 404:
                    errors.append(f"{endpoint}: 404")
                    continue
                response.raise_for_status()
                return self._parse_scores(response.json(), len(documents))
            except requests.RequestException as exc:
                errors.append(f"{endpoint}: {exc}")
        raise RuntimeError("llama.cpp rerank endpoint failed: " + "; ".join(errors))

    def _parse_scores(self, payload: Dict[str, Any], expected_count: int) -> List[float]:
        raw_results = payload.get("results", payload.get("data", []))
        scores_by_index: Dict[int, float] = {}
        if isinstance(raw_results, list):
            for position, item in enumerate(raw_results):
                if not isinstance(item, dict):
                    continue
                index = int(item.get("index", position))
                score = item.get("relevance_score", item.get("score", item.get("logit")))
                if score is not None:
                    scores_by_index[index] = float(score)
        if scores_by_index:
            return [scores_by_index.get(index, 0.0) for index in range(expected_count)]
        scores = payload.get("scores")
        if isinstance(scores, list):
            return [float(score) for score in scores[:expected_count]]
        raise RuntimeError("Rerank response did not contain scores.")
