import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from application.services.autonomy_coordinator_service import AutonomyCoordinatorService
from application.services.capability_policy_service import CapabilityPolicyService
from application.services.capture_control_service import CaptureControlService
from application.services.opportunity_judgment_service import OpportunityJudgmentService
from core.models import AmbientEvent
from infrastructure.adapter.SQLiteAutonomyAdapter import SQLiteAutonomyAdapter
from infrastructure.plain_capture_store import PlainCaptureStore
from infrastructure.runtime_log_server import RuntimeLogBuffer, create_runtime_log_app


class _JudgmentProvider:
    async def chat_completion_stream(self, **kwargs):
        async def stream():
            payload = {
                "classification": "opportunity",
                "title": "Observed hackathon",
                "goal": "Research tracks, personalize project ideas, and verify the deadline.",
                "rationale": "The user is reviewing a time-sensitive hackathon page.",
                "expected_value": 0.9,
                "urgency": 0.8,
                "confidence": 0.9,
                "cost_of_wrong": 0.1,
                "personalization_benefit": 0.9,
                "evidence_gaps": ["Verify the deadline", "Find the official tracks"],
            }
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=json.dumps(payload)))]
            )

        return stream()


def _event(event_id: str = "event-1", fingerprint: str = "fingerprint-1") -> AmbientEvent:
    now = datetime.now(timezone.utc).isoformat()
    return AmbientEvent(
        event_id=event_id,
        event_type="visual_context_changed",
        source_kind="passive_observer",
        source_ref="observation-1",
        occurred_at=now,
        payload_json=json.dumps(
            {
                "url": "https://example.org/hackathon",
                "title": "AI Hackathon",
                "summary": "Applications close Friday",
            }
        ),
        confidence=0.9,
        privacy_label="sensitive_visual",
        fingerprint=fingerprint,
        priority=0.8,
        available_at=now,
    )


def test_event_store_deduplicates_leases_and_recovers_expired_work(tmp_path):
    store = SQLiteAutonomyAdapter(str(tmp_path / "autonomy.db"))
    first = store.enqueue_event(_event())
    duplicate = store.enqueue_event(_event(event_id="event-2"))
    assert duplicate.event_id == first.event_id

    claimed = store.claim_next_event(lease_seconds=1)
    assert claimed is not None and claimed.status == "leased"
    assert store.claim_next_event(lease_seconds=1) is None

    with store._connect() as conn:
        conn.execute(
            "UPDATE ambient_events SET lease_expires_at=? WHERE event_id=?",
            ((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(), claimed.event_id),
        )
    reclaimed = store.claim_next_event(lease_seconds=30)
    assert reclaimed is not None and reclaimed.event_id == claimed.event_id
    store.complete_event(reclaimed.event_id)
    assert store.claim_next_event() is None


def test_shadow_coordinator_judges_active_context_without_idle_trigger(tmp_path):
    store = SQLiteAutonomyAdapter(str(tmp_path / "autonomy.db"))
    provider = _JudgmentProvider()
    policy = CapabilityPolicyService(store=store)
    coordinator = AutonomyCoordinatorService(
        store=store,
        judgment=OpportunityJudgmentService(llm_provider=provider),
        policy=policy,
        mode="shadow",
    )
    store.enqueue_event(_event())

    result = asyncio.run(
        coordinator.process_next(
            model="test-model",
            llm_service=SimpleNamespace(),
            personalization_context="User likes local AI and Python projects.",
        )
    )

    assert result["outcome"] == "shadow"
    inbox = store.list_inbox_items()
    assert len(inbox) == 1
    assert "Research tracks" in inbox[0].summary
    assert "No tools or external actions" in inbox[0].detailed_report


def test_policy_blocks_inferred_shell_and_requires_approval_for_browser_mutation(tmp_path):
    store = SQLiteAutonomyAdapter(str(tmp_path / "autonomy.db"))
    policy = CapabilityPolicyService(store=store)

    denied = policy.evaluate(
        tool_name="powershell_terminal",
        arguments={"command": "Get-ChildItem"},
        source="autonomy_investigation",
        confidence=0.99,
    )
    browser = policy.evaluate(
        tool_name="use_browser",
        arguments={"task": "Submit the application form"},
        source="autonomy_investigation",
        confidence=0.99,
    )
    low_confidence_reminder = policy.evaluate(
        tool_name="add_task",
        arguments={"content": "Hackathon deadline"},
        source="autonomy_investigation",
        confidence=0.70,
    )

    assert denied.decision == "deny"
    assert browser.requires_approval is True
    assert low_confidence_reminder.requires_approval is True


def test_automatic_reminders_require_verified_sources_and_shadow_calibration(tmp_path):
    store = SQLiteAutonomyAdapter(str(tmp_path / "autonomy.db"))
    policy = CapabilityPolicyService(store=store)
    arguments = {
        "content": "Hackathon deadline",
        "due_datetime": "2026-08-07T17:00:00+05:30",
        "source_url": "https://example.org/hackathon/rules",
        "source_verified": True,
    }

    gated = policy.evaluate(
        tool_name="add_task",
        arguments=arguments,
        source="autonomy_investigation",
        confidence=0.95,
        evidence_context={
            "observed_text": "Official deadline: 2026-08-07. https://example.org/hackathon/rules"
        },
    )
    assert gated.requires_approval is True
    assert gated.matched_rule == "calibration_gate"

    for index in range(20):
        store.record_calibration_outcome(
            "assistance.reminder", correct=True, source_ref=f"shadow-{index}"
        )

    allowed = policy.evaluate(
        tool_name="add_task",
        arguments=arguments,
        source="autonomy_investigation",
        confidence=0.95,
        evidence_context={
            "observed_text": "Official deadline: 2026-08-07. https://example.org/hackathon/rules"
        },
    )
    assert allowed.decision == "auto_reversible"
    assert allowed.requires_approval is False


def test_loopback_control_api_works_without_login():
    capture = CaptureControlService()
    app = create_runtime_log_app(
        RuntimeLogBuffer(),
        capture_control=capture,
    )
    client = TestClient(app)

    assert client.get("/api/logs").status_code == 200
    paused = client.post("/api/privacy/capture/pause")
    assert paused.status_code == 200
    assert capture.is_paused() is True
    exclusions = client.put(
        "/api/privacy/capture/exclusions",
        json={"apps": ["1Password"], "domains": ["bank.example"]},
    )
    assert exclusions.status_code == 200
    assert capture.is_excluded(app_name="1password.exe") is True
    assert capture.is_excluded(domain="login.bank.example") is True


def test_plain_capture_store_keeps_normal_readable_files(tmp_path):
    store = PlainCaptureStore(str(tmp_path / "captures"))
    raw = tmp_path / "screen.png"
    raw.write_bytes(b"plain screenshot bytes")

    uri = store.store_file(str(raw), kind="screenshot", delete_source=True)
    stored_files = list((tmp_path / "captures" / "screenshot").glob("*.png"))

    assert uri.startswith("capture://")
    assert not raw.exists()
    assert len(stored_files) == 1
    assert stored_files[0].read_bytes() == b"plain screenshot bytes"
    with store.materialize(uri) as materialized:
        assert Path(materialized) == stored_files[0]
    assert store.storage_status()["storage_mode"] == "plain"
