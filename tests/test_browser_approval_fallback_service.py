import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from application.services.browser_approval_fallback_service import BrowserApprovalFallbackService
from core.models import ApprovalGrant, DelegatedTask
from infrastructure.adapter.SQLiteAutonomyAdapter import SQLiteAutonomyAdapter


class _Search:
    def __init__(self):
        self.calls = []

    def search_text(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "ok": True,
            "query": kwargs["query"],
            "count": 1,
            "results": [{
                "title": "Example result",
                "url": "https://example.com/result",
                "snippet": "Useful result.",
            }],
        }


def _approval(*, approval_id: str, expires_at: datetime, status: str = "pending") -> ApprovalGrant:
    return ApprovalGrant(
        approval_id=approval_id,
        capability="browser.use",
        action_fingerprint=f"fingerprint-{approval_id}",
        constraints_json="{}",
        status=status,
        created_at=(expires_at - timedelta(minutes=30)).isoformat(),
        expires_at=expires_at.isoformat(),
        approver="pending",
    )


def _delegated(*, approval_id: str, delegation_id: str = "delegation-1") -> DelegatedTask:
    return DelegatedTask(
        delegation_id=delegation_id,
        approval_id=approval_id,
        capability="browser.use",
        task="Find Ambient AI browser-use model results",
        reason="Need current public web information",
        expected_result="Search results",
        continuation_instruction="Use the fallback search results to answer the user.",
        origin_kind="direct_chat",
        origin_json=json.dumps({"chat_session_id": "session-1", "chat_message_id": "message-1"}),
        parent_model="main-model",
        status="awaiting_approval",
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )


def test_expired_browser_approval_falls_back_to_ddgs_and_enqueues_continuation(tmp_path):
    store = SQLiteAutonomyAdapter(str(tmp_path / "autonomy.db"))
    approval = _approval(
        approval_id="approval-expired",
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    delegated = _delegated(approval_id=approval.approval_id)
    store.create_approval(approval)
    store.create_delegated_task(delegated)
    search = _Search()
    service = BrowserApprovalFallbackService(
        autonomy_store=store,
        search_service=search,
        max_results=3,
    )

    result = service.run_once()

    assert result["processed"] == ["approval-expired"]
    assert search.calls[0]["query"] == delegated.task
    assert search.calls[0]["max_results"] == 3
    assert store.get_approval(approval.approval_id).status == "expired"
    updated = store.get_delegated_task(delegated.delegation_id)
    assert updated.status == "completed"
    payload = json.loads(updated.result_json)
    assert payload["fallback"]["kind"] == "ddgs_web_search_after_browser_approval_timeout"
    event = store.claim_next_event(lease_seconds=30)
    assert event is not None
    assert event.event_type == "delegated_action_completed"


def test_denied_browser_approval_does_not_fallback(tmp_path):
    store = SQLiteAutonomyAdapter(str(tmp_path / "autonomy.db"))
    approval = _approval(
        approval_id="approval-denied",
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
        status="denied",
    )
    delegated = _delegated(approval_id=approval.approval_id)
    store.create_approval(approval)
    store.create_delegated_task(delegated)
    search = _Search()
    service = BrowserApprovalFallbackService(autonomy_store=store, search_service=search)

    result = service.run_once()

    assert result["processed"] == []
    assert search.calls == []
    assert store.get_approval(approval.approval_id).status == "denied"
    assert store.get_delegated_task(delegated.delegation_id).status == "awaiting_approval"
