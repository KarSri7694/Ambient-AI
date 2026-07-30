import logging
import re
from collections.abc import Callable
from typing import Any


class DdgsSearchService:
    """Validated, normalized wrapper around the DDGS text metasearch API."""

    VALID_SAFESEARCH = {"on", "moderate", "off"}
    VALID_TIMELIMITS = {"", "d", "w", "m", "y"}
    BACKEND_PATTERN = re.compile(r"^[a-z0-9_-]+(?:\s*,\s*[a-z0-9_-]+)*$", re.IGNORECASE)

    def __init__(
        self,
        *,
        timeout_seconds: float = 10.0,
        proxy: str | None = None,
        ddgs_factory: Callable[..., Any] | None = None,
        logger: logging.Logger | None = None,
    ):
        self.timeout_seconds = max(1, min(120, int(timeout_seconds)))
        self.proxy = str(proxy or "").strip() or None
        self.ddgs_factory = ddgs_factory
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def search_text(
        self,
        *,
        query: str,
        max_results: int = 15,
        region: str = "us-en",
        safesearch: str = "moderate",
        timelimit: str = "",
        page: int = 1,
        backend: str = "auto",
    ) -> dict[str, Any]:
        query = str(query or "").strip()
        region = str(region or "us-en").strip().lower()
        safesearch = str(safesearch or "moderate").strip().lower()
        timelimit = str(timelimit or "").strip().lower()
        backend = str(backend or "auto").strip().lower()

        error = self._validate(
            query=query,
            max_results=max_results,
            region=region,
            safesearch=safesearch,
            timelimit=timelimit,
            page=page,
            backend=backend,
        )
        if error:
            return self._error_payload(query=query, error="invalid_request", detail=error)

        try:
            factory = self.ddgs_factory or self._load_ddgs
            client = factory(proxy=self.proxy, timeout=self.timeout_seconds)
            raw_results = client.text(
                query,
                region=region,
                safesearch=safesearch,
                timelimit=timelimit or None,
                max_results=int(max_results),
                page=int(page),
                backend=backend,
            )
            results = self._normalize_results(raw_results, limit=int(max_results))
            return {
                "ok": True,
                "provider": "ddgs",
                "query": query,
                "region": region,
                "safesearch": safesearch,
                "timelimit": timelimit or None,
                "backend": backend,
                "page": int(page),
                "count": len(results),
                "results": results,
            }
        except ModuleNotFoundError:
            self.logger.warning("DDGS search requested but the ddgs package is not installed.")
            return self._error_payload(
                query=query,
                error="dependency_unavailable",
                detail="Install the project's requirements to enable DDGS search.",
            )
        except Exception as exc:
            self.logger.warning("DDGS search failed (%s).", type(exc).__name__)
            return self._error_payload(
                query=query,
                error="search_failed",
                detail=f"DDGS search failed ({type(exc).__name__}). Try again or choose another backend.",
            )

    @staticmethod
    def _load_ddgs(**kwargs):
        from ddgs import DDGS

        return DDGS(**kwargs)

    @classmethod
    def _validate(
        cls,
        *,
        query: str,
        max_results: int,
        region: str,
        safesearch: str,
        timelimit: str,
        page: int,
        backend: str,
    ) -> str | None:
        if not query:
            return "query must not be empty"
        if len(query) > 1000:
            return "query must be 1000 characters or fewer"
        if not isinstance(max_results, int) or isinstance(max_results, bool) or not 1 <= max_results <= 20:
            return "max_results must be an integer from 1 to 20"
        if not isinstance(page, int) or isinstance(page, bool) or not 1 <= page <= 10:
            return "page must be an integer from 1 to 10"
        if safesearch not in cls.VALID_SAFESEARCH:
            return "safesearch must be on, moderate, or off"
        if timelimit not in cls.VALID_TIMELIMITS:
            return "timelimit must be empty, d, w, m, or y"
        if not region or len(region) > 20 or not re.fullmatch(r"[a-z0-9-]+", region):
            return "region must be a DDGS region such as us-en or wt-wt"
        if not backend or not cls.BACKEND_PATTERN.fullmatch(backend):
            return "backend must be one backend or a comma-delimited backend list"
        return None

    @staticmethod
    def _normalize_results(raw_results, *, limit: int) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for raw in list(raw_results or []):
            if not isinstance(raw, dict):
                continue
            url = str(raw.get("href") or raw.get("url") or raw.get("link") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            normalized.append(
                {
                    "title": str(raw.get("title") or "Untitled result").strip(),
                    "url": url,
                    "snippet": str(
                        raw.get("body") or raw.get("snippet") or raw.get("description") or ""
                    ).strip(),
                    "source": str(raw.get("source") or raw.get("provider") or "").strip() or None,
                }
            )
            if len(normalized) >= limit:
                break
        return normalized

    @staticmethod
    def _error_payload(*, query: str, error: str, detail: str) -> dict[str, Any]:
        return {
            "ok": False,
            "provider": "ddgs",
            "query": query,
            "count": 0,
            "results": [],
            "error": error,
            "detail": detail,
        }
