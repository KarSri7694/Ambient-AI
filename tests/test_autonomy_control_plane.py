import asyncio
import json
import sys
from dataclasses import replace
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
from application.services.llm_interaction_service import InteractionSuspended
from core.models import AmbientEvent, ApprovalGrant, DelegatedTask, VisualObservation
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


class _CapturingJudgment:
    def __init__(self):
        self.personalization_contexts = []

    async def judge(self, *, event, model, personalization_context=""):
        self.personalization_contexts.append(personalization_context)
        return None


class _UserContext:
    def __init__(self):
        self.queries = []

    def build_prompt_context(self, *, query_text="", include_semantic=True, max_chars=None):
        self.queries.append(
            {
                "query_text": query_text,
                "include_semantic": include_semantic,
                "max_chars": max_chars,
            }
        )
        return "User profile: user is building Ambient AI for ROCm hackathon."


class _TemporalVLMContext:
    def __init__(self):
        self.recorded = []
        self.queries = []

    def record_ambient_event(self, event, outcome=""):
        self.recorded.append((event, outcome))
        return SimpleNamespace(thread_id="temporal-thread")

    def build_context(self, *, query_text, current_event=None):
        self.queries.append(query_text)
        return {"active_thread": {"thread_id": "temporal-thread"}}

    def build_prompt_context(self, **_kwargs):
        return "## Temporal work context\n- The user is comparing local embedding models."


class _SuspendingInvestigationService:
    def reset_context(self):
        return None

    def available_tool_definitions(self):
        return [{"type": "function", "function": {"name": "use_browser", "parameters": {}}}]

    async def run_interaction(self, **kwargs):
        now = datetime.now(timezone.utc).isoformat()
        approval = ApprovalGrant(
            approval_id="a" * 32, capability="browser.use", action_fingerprint="fingerprint",
            constraints_json="{}", status="pending", created_at=now,
            expires_at=(datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat(),
            approver="pending",
        )
        task = DelegatedTask(
            delegation_id="d" * 32, approval_id=approval.approval_id,
            capability="browser.use", task="Research the official hackathon deadline",
            reason="Current information is needed", expected_result="Verified deadline",
            continuation_instruction="Complete the original report", origin_kind="autonomy",
            origin_json="{}", parent_model="test-model", status="awaiting_approval",
            created_at=now, updated_at=now,
        )
        raise InteractionSuspended(approval=approval, delegated_task=task, tool_call_id="call-1")


class _NeverJudgeQueuedTask:
    async def judge(self, **_kwargs):
        raise AssertionError("queued background tasks must not be judged a second time")

    @staticmethod
    def qualifies_for_enrichment(candidate):
        return candidate.expected_value >= 0.6


class _CompletingInvestigationService:
    def __init__(self):
        self.calls = []

    def reset_context(self):
        return None

    def available_tool_definitions(self):
        return []

    async def run_interaction(self, **kwargs):
        self.calls.append(kwargs)
        return "Completed the queued ROCm benchmark comparison."


class _FastVisualObserver:
    def __init__(self, memory=None, needs_deep_analysis=False):
        self.calls = []
        self.memory = memory
        self.needs_deep_analysis = needs_deep_analysis

    async def process_screenshot(self, **kwargs):
        self.calls.append(kwargs)
        observation = VisualObservation(
            observation_id=kwargs["observation_id"],
            screenshot_path=kwargs["persisted_screenshot_path"],
            created_at=kwargs["captured_at"],
            app_name="Browser",
            page_hint="Example page",
            summary="A product comparison is visible.",
            detailed_description="Two products and their prices are shown.",
            inferred_user_activity="Comparing products",
            confidence=0.75,
            analysis_status="model",
            analysis_latency_ms=4200,
            analysis_model="fast-vlm",
            source_capture_event_id=kwargs["source_capture_event_id"],
            needs_deep_analysis=self.needs_deep_analysis,
        )
        if self.memory is not None:
            self.memory.append_visual_observation(observation)
        return observation


class _VisualMemory:
    def __init__(self):
        self.items = {}

    def append_visual_observation(self, observation):
        self.items[observation.observation_id] = observation

    def get_visual_observation(self, observation_id):
        return self.items.get(observation_id)


class _DeepVisualObserver:
    def __init__(self, memory):
        self.memory = memory
        self.full_model = "deep-vlm"
        self.calls = []

    async def process_screenshot(self, **kwargs):
        self.calls.append(kwargs)
        existing = self.memory.get_visual_observation(kwargs["observation_id"])
        return VisualObservation(
            **{
                **existing.__dict__,
                "summary": "A detailed product comparison with prices is visible.",
                "detailed_description": "The larger model extracted both product names and exact prices.",
                "analysis_model": "deep-vlm",
                "analysis_latency_ms": 12000,
                "needs_deep_analysis": False,
            }
        )


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


def test_event_store_excludes_visual_captures_from_downstream_claims(tmp_path):
    store = SQLiteAutonomyAdapter(str(tmp_path / "autonomy.db"))
    downstream = store.enqueue_event(_event(event_id="downstream-event"))
    visual = replace(
        _event(event_id="visual-event"),
        event_type="lightweight_visual_capture",
        source_kind="screen_capture",
        source_ref="capture://00000000000000000000000000000001",
        fingerprint="visual-capture-fingerprint",
        priority=0.95,
    )
    store.enqueue_event(visual)

    assert store.has_ready_events(exclude_event_types=["lightweight_visual_capture"]) is True
    claimed = store.claim_next_event(exclude_event_types=["lightweight_visual_capture"])
    assert claimed is not None
    assert claimed.event_id == downstream.event_id
    assert store.claim_next_event(event_types=["lightweight_visual_capture"]).event_id == visual.event_id


def test_event_store_claims_offset_timestamp_as_same_instant(tmp_path):
    store = SQLiteAutonomyAdapter(str(tmp_path / "autonomy.db"))
    india = timezone(timedelta(hours=5, minutes=30))
    local_timestamp = (datetime.now(india) - timedelta(minutes=1)).isoformat()
    event = AmbientEvent(
        event_id="offset-approval-event",
        event_type="approval_granted",
        source_kind="local_approval",
        source_ref="approval-1",
        occurred_at=local_timestamp,
        payload_json="{}",
        confidence=1.0,
        privacy_label="private",
        fingerprint="offset-approval-fingerprint",
        priority=1.0,
        available_at=local_timestamp,
    )

    stored = store.enqueue_event(event)
    assert stored.available_at.endswith("+00:00")

    # Reproduce a row written by the older local-time API. SQLite must compare
    # the instant represented by the offset, not the timestamp text itself.
    with store._connect() as conn:
        conn.execute(
            "UPDATE ambient_events SET occurred_at=?, available_at=? WHERE event_id=?",
            (local_timestamp, local_timestamp, event.event_id),
        )

    assert store.has_ready_events() is True
    claimed = store.claim_next_event()
    assert claimed is not None
    assert claimed.event_id == event.event_id


def test_event_store_repairs_future_screen_timestamps_from_legacy_local_clock(tmp_path):
    db_path = tmp_path / "autonomy.db"
    store = SQLiteAutonomyAdapter(str(db_path))
    future = (datetime.now(timezone.utc) + timedelta(hours=5)).isoformat()
    event = AmbientEvent(
        event_id="future-screen-event",
        event_type="lightweight_visual_capture",
        source_kind="screen_capture",
        source_ref="capture://00000000000000000000000000000001",
        occurred_at=future,
        payload_json="{}",
        confidence=0.5,
        privacy_label="sensitive_visual",
        fingerprint="future-screen-fingerprint",
        status="pending",
        priority=0.5,
        available_at=future,
    )
    store.enqueue_event(event)
    assert store.has_ready_events() is False

    restarted = SQLiteAutonomyAdapter(str(db_path))
    assert restarted.recovered_future_capture_timestamps == 1
    assert restarted.has_ready_events() is True
    claimed = restarted.claim_next_event()
    assert claimed is not None
    assert datetime.fromisoformat(claimed.occurred_at) <= datetime.now(timezone.utc)


def test_visual_perception_completes_capture_before_judgment(tmp_path):
    store = SQLiteAutonomyAdapter(str(tmp_path / "autonomy.db"))
    capture_store = PlainCaptureStore(str(tmp_path / "captures"))
    screenshot = tmp_path / "screen.png"
    screenshot.write_bytes(b"test-image")
    screenshot_ref = capture_store.store_file(str(screenshot), kind="screenshot", delete_source=False)
    observer = _FastVisualObserver()
    user_context = _UserContext()
    temporal_context = _TemporalVLMContext()
    coordinator = AutonomyCoordinatorService(
        store=store,
        judgment=_CapturingJudgment(),
        policy=CapabilityPolicyService(store=store),
        mode="shadow",
        capture_store=capture_store,
        visual_observer=observer,
        visual_model="fast-vlm",
        user_context_service=user_context,
        temporal_memory_service=temporal_context,
    )
    captured_at = datetime.now(timezone.utc).isoformat()
    capture_event = coordinator.enqueue_lightweight_visual(
        screenshot_ref=screenshot_ref,
        captured_at=captured_at,
        context={"app_name": "Browser", "window_title": "Example", "accessible_text": "Products"},
        similarity_score=0.5,
    )

    result = asyncio.run(coordinator.process_next_visual())

    assert result["outcome"] == "perception_completed"
    assert result["analysis_latency_ms"] == 4200
    assert observer.calls[0]["source_capture_event_id"] == capture_event.event_id
    assert "comparing local embedding models" in observer.calls[0]["temporal_context"]
    assert temporal_context.queries
    assert user_context.queries == []
    counts = store.event_counts()
    assert counts["processed"] == 1
    assert counts["pending"] == 1
    downstream = store.claim_next_event(event_types=["visual_context_changed"])
    assert downstream is not None
    assert json.loads(downstream.payload_json)["analysis_model"] == "fast-vlm"


def test_visual_observations_batch_until_configured_size(tmp_path):
    store = SQLiteAutonomyAdapter(str(tmp_path / "autonomy.db"))
    coordinator = AutonomyCoordinatorService(
        store=store,
        judgment=_CapturingJudgment(),
        policy=CapabilityPolicyService(store=store),
        mode="shadow",
        visual_context_batch_size=3,
        visual_context_batch_max_wait_seconds=60,
    )
    now = datetime.now(timezone.utc).isoformat()
    for index in range(2):
        observation = VisualObservation(
            observation_id=f"obs-{index}",
            screenshot_path=f"capture://{index}",
            created_at=now,
            app_name="Browser",
            summary=f"Screen {index}",
            detailed_description="A useful page is visible.",
            inferred_user_activity="Researching",
            confidence=0.7,
            raw_payload_json=json.dumps({"salience": "medium"}),
            analysis_model="fast-vlm",
        )
        event = coordinator._enqueue_visual_batch_or_single(observation, {})
        assert event.event_type == "visual_context_batch_pending"

    assert store.claim_next_event(event_types=["visual_context_batch_changed"]) is None

    observation = VisualObservation(
        observation_id="obs-2",
        screenshot_path="capture://2",
        created_at=now,
        app_name="Browser",
        summary="Screen 2",
        detailed_description="A useful page is visible.",
        inferred_user_activity="Researching",
        confidence=0.7,
        raw_payload_json=json.dumps({"salience": "medium"}),
        analysis_model="fast-vlm",
    )
    event = coordinator._enqueue_visual_batch_or_single(observation, {})

    assert event.event_type == "visual_context_batch_changed"
    payload = json.loads(event.payload_json)
    assert payload["flush_reason"] == "size"
    assert payload["observation_ids"] == ["obs-0", "obs-1", "obs-2"]


def test_visual_batch_timeout_is_processed_as_batch_event(tmp_path):
    store = SQLiteAutonomyAdapter(str(tmp_path / "autonomy.db"))
    coordinator = AutonomyCoordinatorService(
        store=store,
        judgment=_CapturingJudgment(),
        policy=CapabilityPolicyService(store=store),
        mode="shadow",
        visual_context_batch_size=5,
        visual_context_batch_max_wait_seconds=60,
    )
    old = (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()
    observation = VisualObservation(
        observation_id="old-obs",
        screenshot_path="capture://old",
        created_at=old,
        app_name="Browser",
        summary="Old screen",
        confidence=0.7,
        raw_payload_json=json.dumps({"salience": "medium"}),
        analysis_model="fast-vlm",
    )
    event = coordinator._enqueue_visual_batch_or_single(observation, {})

    result = asyncio.run(
        coordinator.process_next(model="model", llm_service=None, personalization_context="")
    )

    assert result["event_type"] == "visual_context_batch_changed"
    assert result["processed"] is True


def test_high_salience_visual_flushes_batch_immediately(tmp_path):
    store = SQLiteAutonomyAdapter(str(tmp_path / "autonomy.db"))
    coordinator = AutonomyCoordinatorService(
        store=store,
        judgment=_CapturingJudgment(),
        policy=CapabilityPolicyService(store=store),
        mode="shadow",
        visual_context_batch_size=5,
        visual_context_batch_flush_high_salience=True,
    )
    observation = VisualObservation(
        observation_id="urgent-obs",
        screenshot_path="capture://urgent",
        created_at=datetime.now(timezone.utc).isoformat(),
        app_name="Browser",
        summary="Urgent deadline visible",
        confidence=0.8,
        raw_payload_json=json.dumps({"salience": "high"}),
        analysis_model="fast-vlm",
    )
    event = coordinator._enqueue_visual_batch_or_single(observation, {})

    assert event.event_type == "visual_context_batch_changed"
    assert json.loads(event.payload_json)["flush_reason"] == "high_salience"


def test_high_salience_visual_is_deep_enriched_asynchronously(tmp_path):
    store = SQLiteAutonomyAdapter(str(tmp_path / "autonomy.db"))
    capture_store = PlainCaptureStore(str(tmp_path / "captures"))
    screenshot = tmp_path / "screen.png"
    screenshot.write_bytes(b"test-image")
    screenshot_ref = capture_store.store_file(str(screenshot), kind="screenshot", delete_source=False)
    memory = _VisualMemory()
    fast = _FastVisualObserver(memory=memory, needs_deep_analysis=True)
    deep = _DeepVisualObserver(memory)
    coordinator = AutonomyCoordinatorService(
        store=store,
        judgment=_CapturingJudgment(),
        policy=CapabilityPolicyService(store=store),
        mode="shadow",
        capture_store=capture_store,
        visual_observer=fast,
        deep_visual_observer=deep,
        visual_model="fast-vlm",
    )
    coordinator.enqueue_lightweight_visual(
        screenshot_ref=screenshot_ref,
        captured_at=datetime.now(timezone.utc).isoformat(),
        context={"app_name": "Browser", "window_title": "Checkout warning"},
        similarity_score=0.4,
    )

    perception = asyncio.run(coordinator.process_next_visual())
    deep_result = asyncio.run(
        coordinator.process_next(model="deep-vlm", llm_service=None, personalization_context="profile")
    )

    assert perception["outcome"] == "perception_completed"
    assert deep_result["outcome"] == "deep_enriched"
    assert deep.calls[0]["force_full_analysis"] is True
    downstream = store.claim_next_event(event_types=["visual_context_changed"])
    assert downstream is not None
    downstream_payload = json.loads(downstream.payload_json)
    assert downstream_payload["analysis_status"] == "deep_enriched"
    assert downstream_payload["analysis_model"] == "deep-vlm"


def test_visual_capture_backlog_keeps_only_recent_context_items(tmp_path):
    store = SQLiteAutonomyAdapter(str(tmp_path / "autonomy.db"))
    coordinator = AutonomyCoordinatorService(
        store=store,
        judgment=_CapturingJudgment(),
        policy=CapabilityPolicyService(store=store),
        mode="shadow",
        max_pending_visual_per_context=2,
    )
    for index in range(3):
        coordinator.enqueue_lightweight_visual(
            screenshot_ref=f"capture://{'a' * 31}{index}",
            captured_at=(datetime.now(timezone.utc) + timedelta(seconds=index)).isoformat(),
            context={"app_name": "Browser", "accessible_text": f"state {index}"},
            similarity_score=0.5,
        )

    counts = store.event_counts()
    assert counts["pending"] == 2
    assert counts["ignored"] == 1


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


def test_coordinator_builds_event_specific_personalization_context(tmp_path):
    store = SQLiteAutonomyAdapter(str(tmp_path / "autonomy.db"))
    judgment = _CapturingJudgment()
    user_context = _UserContext()
    coordinator = AutonomyCoordinatorService(
        store=store,
        judgment=judgment,
        policy=CapabilityPolicyService(store=store),
        mode="shadow",
        user_context_service=user_context,
    )
    store.enqueue_event(_event())

    result = asyncio.run(
        coordinator.process_next(
            model="test-model",
            llm_service=SimpleNamespace(),
            personalization_context="stale recent context",
        )
    )

    assert result["outcome"] == "ignored"
    assert "ROCm hackathon" in judgment.personalization_contexts[-1]
    assert "Applications close Friday" in user_context.queries[-1]["query_text"]
    assert user_context.queries[-1]["include_semantic"] is True


def test_feedback_is_durable_and_injected_as_preference_evidence(tmp_path):
    store = SQLiteAutonomyAdapter(str(tmp_path / "autonomy.db"))
    policy = CapabilityPolicyService(store=store)
    first = AutonomyCoordinatorService(
        store=store,
        judgment=OpportunityJudgmentService(llm_provider=_JudgmentProvider()),
        policy=policy,
        mode="shadow",
    )
    store.enqueue_event(_event())
    assert asyncio.run(first.process_next(
        model="test-model", llm_service=SimpleNamespace(), personalization_context=""
    ))["outcome"] == "shadow"
    inbox = store.list_inbox_items()[0]
    assert store.record_feedback(inbox.inbox_id, "too_intrusive") is True
    audit = store.list_feedback_for_inbox(inbox.inbox_id)
    assert audit[0]["feedback"] == "too_intrusive"

    judgment = _CapturingJudgment()
    coordinator = AutonomyCoordinatorService(
        store=store,
        judgment=judgment,
        policy=policy,
        mode="shadow",
    )
    store.enqueue_event(_event("event-2", "fingerprint-2"))
    assert asyncio.run(coordinator.process_next(
        model="test-model", llm_service=SimpleNamespace(), personalization_context="base profile"
    ))["outcome"] == "ignored"
    context = judgment.personalization_contexts[-1]
    assert "Explicit proactive feedback" in context
    assert "too_intrusive" in context
    assert "never authorization" in context


def test_active_investigation_becomes_awaiting_approval_without_retry(tmp_path):
    store = SQLiteAutonomyAdapter(str(tmp_path / "autonomy.db"))
    coordinator = AutonomyCoordinatorService(
        store=store,
        judgment=OpportunityJudgmentService(llm_provider=_JudgmentProvider()),
        policy=CapabilityPolicyService(store=store),
        mode="active",
    )
    store.enqueue_event(_event())

    result = asyncio.run(coordinator.process_next(
        model="test-model",
        llm_service=_SuspendingInvestigationService(),
        personalization_context="User likes local AI.",
    ))

    assert result["outcome"] == "awaiting_approval"
    assert store.event_counts()["processed"] == 1
    assert store.event_counts().get("pending", 0) == 0
    inbox = store.list_inbox_items()
    assert len(inbox) == 1
    assert inbox[0].status == "awaiting_approval"


def test_untimed_queued_task_executes_through_active_coordinator(tmp_path):
    store = SQLiteAutonomyAdapter(str(tmp_path / "autonomy.db"))
    service = _CompletingInvestigationService()
    coordinator = AutonomyCoordinatorService(
        store=store,
        judgment=_NeverJudgeQueuedTask(),
        policy=CapabilityPolicyService(store=store),
        mode="active",
    )
    event = coordinator.enqueue_background_task(
        task_id=17,
        description="Compare the completed ROCm benchmark configurations.",
        priority="medium",
        metadata_json=json.dumps(
            {"source": "reflection_service", "reason": "Benchmarking has finished."}
        ),
    )

    result = asyncio.run(
        coordinator.process_next(
            model="test-model",
            llm_service=service,
            personalization_context="User is optimizing llama.cpp on Radeon.",
        )
    )

    assert event.event_type == "queued_background_task"
    assert result["outcome"] == "completed"
    assert len(service.calls) == 1
    assert "Compare the completed ROCm benchmark" in service.calls[0]["user_input"]
    inbox = store.list_inbox_items()
    assert len(inbox) == 1
    assert inbox[0].status == "completed"
    assert "Completed the queued ROCm" in inbox[0].detailed_report


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


def test_expired_approval_cannot_enqueue_execution(tmp_path):
    store = SQLiteAutonomyAdapter(str(tmp_path / "autonomy.db"))
    now = datetime.now(timezone.utc)
    store.create_approval(
        ApprovalGrant(
            approval_id="expired-approval", capability="browser.use",
            action_fingerprint="expired-fingerprint", constraints_json="{}",
            status="pending", created_at=(now - timedelta(hours=2)).isoformat(),
            expires_at=(now - timedelta(hours=1)).isoformat(), approver="pending",
        )
    )
    client = TestClient(
        create_runtime_log_app(RuntimeLogBuffer(), autonomy_store=store)
    )

    response = client.post(
        "/api/autonomy/approvals/expired-approval/decision", json={"approved": True}
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "approval_expired"
    assert store.get_approval("expired-approval").status == "expired"
    assert store.event_counts().get("pending", 0) == 0


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
