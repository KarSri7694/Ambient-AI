from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from application.services.ddgs_search_service import DdgsSearchService
from core.models import AmbientEvent, ApprovalGrant, DelegatedTask


class BrowserApprovalFallbackService:
    """Fallback expired browser-use approvals to DDGS search without browser control."""

    def __init__(
        self,
        *,
        autonomy_store: Any,
        search_service: DdgsSearchService | None = None,
        enabled: bool = True,
        max_results: int = 8,
        region: str = "us-en",
        safesearch: str = "moderate",
        backend: str = "auto",
        logger: logging.Logger | None = None,
    ) -> None:
        self.autonomy_store = autonomy_store
        self.search_service = search_service or DdgsSearchService()
        self.enabled = bool(enabled)
        self.max_results = max(1, min(int(max_results), 20))
        self.region = str(region or "us-en").strip() or "us-en"
        self.safesearch = str(safesearch or "moderate").strip() or "moderate"
        self.backend = str(backend or "auto").strip() or "auto"
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def run_once(self) -> dict[str, Any]:
        if not self.enabled or self.autonomy_store is None:
            return {"ran": False, "reason": "disabled"}
        if not all(
            hasattr(self.autonomy_store, name)
            for name in (
                "list_approvals",
                "expire_approval",
                "get_delegated_task_by_approval",
                "update_delegated_task",
                "enqueue_event",
            )
        ):
            return {"ran": False, "reason": "store_missing_required_methods"}

        now = datetime.now(timezone.utc)
        processed: list[str] = []
        skipped: list[str] = []
        for approval in self.autonomy_store.list_approvals(status="pending", limit=500):
            if approval.capability != "browser.use":
                continue
            if not self._expired(approval, now):
                continue
            delegated = self.autonomy_store.get_delegated_task_by_approval(approval.approval_id)
            if delegated is None:
                self.autonomy_store.expire_approval(approval.approval_id)
                skipped.append(approval.approval_id)
                continue
            try:
                self._fallback_one(approval=approval, delegated=delegated)
                processed.append(approval.approval_id)
            except Exception:
                self.logger.exception("Browser approval DDGS fallback failed for %s.", approval.approval_id)
                skipped.append(approval.approval_id)
        return {"ran": True, "processed": processed, "skipped": skipped}

    def _fallback_one(self, *, approval: ApprovalGrant, delegated: DelegatedTask) -> None:
        self.autonomy_store.expire_approval(approval.approval_id)
        query = self._query_from_task(delegated.task)
        search = self.search_service.search_text(
            query=query,
            max_results=self.max_results,
            region=self.region,
            safesearch=self.safesearch,
            backend=self.backend,
        )
        result_payload = {
            "status": "completed" if search.get("ok") else "failed",
            "summary": (
                "Browser-use approval timed out, so Ambient AI used DDGS web search instead."
                if search.get("ok")
                else "Browser-use approval timed out and DDGS fallback search failed."
            ),
            "details": self._format_details(query=query, search=search),
            "fallback": {
                "kind": "ddgs_web_search_after_browser_approval_timeout",
                "approval_id": approval.approval_id,
                "query": query,
                "search": search,
            },
            "actions_performed": ["Expired pending browser-use approval", "Ran DDGS web search fallback"],
            "sources": [
                {"title": item.get("title"), "url": item.get("url"), "snippet": item.get("snippet")}
                for item in search.get("results", [])
            ],
            "blockers": [] if search.get("ok") else [str(search.get("detail") or search.get("error") or "DDGS fallback failed")],
        }
        completion_payload = {
            "delegation_id": delegated.delegation_id,
            "approval_id": delegated.approval_id,
            "capability": delegated.capability,
            "task": delegated.task,
            "reason": delegated.reason,
            "expected_result": delegated.expected_result,
            "continuation_instruction": delegated.continuation_instruction,
            "origin_kind": delegated.origin_kind,
            "origin": self._safe_json(delegated.origin_json),
            "result": result_payload,
        }
        now = datetime.now(timezone.utc).isoformat()
        event = AmbientEvent(
            event_id=uuid.uuid4().hex,
            event_type="delegated_action_completed",
            source_kind="delegated_task",
            source_ref=delegated.delegation_id,
            occurred_at=now,
            payload_json=json.dumps(completion_payload, ensure_ascii=False),
            confidence=1.0,
            privacy_label="public",
            fingerprint=hashlib.sha256(
                f"browser_approval_timeout_ddgs|{delegated.delegation_id}".encode("utf-8")
            ).hexdigest(),
            priority=1.0,
            available_at=now,
        )
        stored = self.autonomy_store.enqueue_event(event)
        self.autonomy_store.update_delegated_task(
            delegated.delegation_id,
            status=str(result_payload["status"]),
            result_json=json.dumps(result_payload, ensure_ascii=False),
            error_text="" if search.get("ok") else str(search.get("detail") or ""),
            continuation_event_id=stored.event_id,
            mark_completed=True,
        )
        if hasattr(self.autonomy_store, "audit"):
            self.autonomy_store.audit(
                "ambient_agent",
                "browser_approval_timeout.ddgs_fallback",
                approval.approval_id,
                {"delegation_id": delegated.delegation_id, "query": query, "ok": bool(search.get("ok"))},
            )

    @staticmethod
    def _expired(approval: ApprovalGrant, now: datetime) -> bool:
        try:
            expires_at = datetime.fromisoformat(approval.expires_at)
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            return expires_at <= now
        except (TypeError, ValueError):
            return True

    @staticmethod
    def _query_from_task(task: str) -> str:
        value = str(task or "").strip()
        return value[:1000] or "web search"

    @staticmethod
    def _format_details(*, query: str, search: dict[str, Any]) -> str:
        if not search.get("ok"):
            return f"DDGS query: {query}\n\nFallback failed: {search.get('detail') or search.get('error')}"
        lines = [f"DDGS query: {query}", ""]
        for index, item in enumerate(search.get("results", []), start=1):
            lines.append(f"{index}. {item.get('title') or 'Untitled'}")
            lines.append(f"   URL: {item.get('url') or ''}")
            snippet = str(item.get("snippet") or "").strip()
            if snippet:
                lines.append(f"   Snippet: {snippet}")
        return "\n".join(lines)

    @staticmethod
    def _safe_json(value: str) -> dict[str, Any]:
        try:
            parsed = json.loads(value or "{}")
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
