import logging
import math
import re
from typing import Any, Dict, List, Optional

import requests


class LlamaCppSemanticAdapter:
    """Small client for llama.cpp-compatible embedding and rerank endpoints."""

    def __init__(
        self,
        *,
        embedding_base_url: str,
        embedding_model: str,
        reranker_base_url: Optional[str] = None,
        reranker_model: Optional[str] = None,
        timeout_seconds: float = 30.0,
    ):
        self.embedding_base_url = embedding_base_url.rstrip("/")
        self.embedding_model = embedding_model.strip()
        self.reranker_base_url = (reranker_base_url or embedding_base_url).rstrip("/")
        self.reranker_model = (reranker_model or "").strip()
        self.timeout_seconds = timeout_seconds
        self.logger = logging.getLogger(self.__class__.__name__)

    def is_enabled(self) -> bool:
        return bool(self.embedding_model)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        normalized = [str(text).strip() for text in texts if str(text).strip()]
        if not normalized or not self.embedding_model:
            return []
        response = requests.post(
            f"{self.embedding_base_url}/v1/embeddings",
            json={"model": self.embedding_model, "input": normalized},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", [])
        ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
        return [list(item.get("embedding") or []) for item in ordered]

    def rerank(
        self,
        *,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        normalized_documents = [str(document).strip() for document in documents]
        if not normalized_documents:
            return []
        if self.reranker_model:
            try:
                response = requests.post(
                    f"{self.reranker_base_url}/v1/rerank",
                    json={
                        "model": self.reranker_model,
                        "query": query,
                        "documents": normalized_documents,
                        "top_n": top_n or len(normalized_documents),
                    },
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                payload = response.json()
                results = payload.get("results") or payload.get("data") or []
                parsed: List[Dict[str, Any]] = []
                for item in results:
                    index = int(item.get("index", item.get("document_index", -1)))
                    if index < 0 or index >= len(normalized_documents):
                        continue
                    parsed.append(
                        {
                            "index": index,
                            "score": float(item.get("relevance_score", item.get("score", 0.0))),
                            "document": normalized_documents[index],
                        }
                    )
                if parsed:
                    return parsed
            except Exception as exc:
                self.logger.warning("Rerank endpoint unavailable, using lexical fallback: %s", exc)
        return self._fallback_rerank(query=query, documents=normalized_documents, top_n=top_n)

    def _fallback_rerank(
        self,
        *,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        query_terms = self._tokenize(query)
        scored: List[Dict[str, Any]] = []
        for index, document in enumerate(documents):
            document_terms = self._tokenize(document)
            overlap = len(query_terms & document_terms)
            length_penalty = math.log(len(document_terms) + 2, 2)
            score = overlap / length_penalty if length_penalty else 0.0
            scored.append({"index": index, "score": score, "document": document})
        scored.sort(key=lambda item: item["score"], reverse=True)
        if top_n is not None:
            scored = scored[:top_n]
        return scored

    def _tokenize(self, text: str) -> set[str]:
        return {token for token in re.findall(r"[A-Za-z0-9_]+", (text or "").lower()) if token}
