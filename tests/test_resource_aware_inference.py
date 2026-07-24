import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from application.services.autonomy_coordinator_service import AutonomyCoordinatorService
from application.services.resource_governor_service import (
    ModelResidencyManager,
    ResourceGovernorService,
    ResourceUnavailableError,
)
from core.models import AmbientEvent, InferenceRequest, ResourceSnapshot
from infrastructure.adapter.SQLiteAutonomyAdapter import SQLiteAutonomyAdapter
from infrastructure.runtime_log_server import RuntimeLogBuffer, create_runtime_log_app


class _MutableMonitor:
    def __init__(self, *, available_ram_mb=12_000, free_vram_mb=5_000):
        self.available_ram_mb = available_ram_mb
        self.free_vram_mb = free_vram_mb
        self.gpu_available = free_vram_mb is not None

    def snapshot(self, *, user_idle=False, force=False):
        total_ram_mb = 16_000
        return ResourceSnapshot(
            captured_at=datetime.now(timezone.utc).isoformat(),
            total_ram_mb=total_ram_mb,
            available_ram_mb=self.available_ram_mb,
            available_ram_percent=self.available_ram_mb / total_ram_mb * 100,
            total_vram_mb=6_144 if self.gpu_available else None,
            free_vram_mb=self.free_vram_mb,
            gpu_telemetry_available=self.gpu_available,
            user_idle=user_idle,
        )


class _ModelProvider:
    def __init__(self):
        self.current = None
        self.loads = []
        self.unloads = []
        self.saved = []

    def get_current_model(self):
        return self.current

    async def load_model(self, model_name):
        self.loads.append(model_name)
        self.current = model_name

    async def unload_model(self):
        self.unloads.append(self.current)
        self.current = None

    async def save_and_unload(self, messages):
        self.saved.append(self.current)
        self.current = None
        return Path("saved-state.bin")

    async def load_and_restore(self):
        self.current = self.saved[-1]
        return Path("saved-state.bin")


def test_active_use_requires_strict_headroom_and_capture_only_defers_background():
    monitor = _MutableMonitor(available_ram_mb=1_500, free_vram_mb=5_000)
    governor = ResourceGovernorService(monitor=monitor, preset="balanced")
    request = InferenceRequest(
        workload="research",
        model_name="unknown-model",
        background=True,
        user_active=True,
    )

    decision = governor.evaluate(request)
    assert decision.allowed is False
    assert "RAM" in decision.reason

    monitor.available_ram_mb = 3_000
    assert governor.evaluate(request).allowed is True

    governor.set_preset("capture_only")
    decision = governor.evaluate(request)
    assert decision.allowed is False
    assert "capture-only" in decision.reason


def test_model_admission_uses_live_vram_without_per_model_estimates():
    monitor = _MutableMonitor(available_ram_mb=3_000, free_vram_mb=5_000)
    governor = ResourceGovernorService(monitor=monitor, preset="balanced")

    active = governor.evaluate(
        InferenceRequest("judgment", "never-profiled-model", background=True, user_active=True)
    )
    assert active.allowed is True
    assert "VRAM" in active.reason

    monitor.available_ram_mb = 2_300
    idle = governor.evaluate(
        InferenceRequest("judgment", "another-unknown-model", background=True, user_active=False),
        force_snapshot=True,
    )
    assert idle.allowed is True
    assert governor.is_critical(idle.snapshot) is False

    monitor.free_vram_mb = 300
    denied = governor.evaluate(
        InferenceRequest("judgment", "another-unknown-model", background=True, user_active=False),
        force_snapshot=True,
    )
    assert denied.allowed is False
    assert "VRAM" in denied.reason


def test_configured_critical_ram_caps_preset_host_ram_reserve():
    monitor = _MutableMonitor(available_ram_mb=335, free_vram_mb=5_994)
    governor = ResourceGovernorService(
        monitor=monitor,
        preset="balanced",
        critical_ram_mb=200,
        critical_ram_percent=3,
    )
    request = InferenceRequest(
        workload="ambient_inference_batch",
        model_name="ambient-model",
        background=True,
        user_active=True,
    )

    decision = governor.evaluate(request)

    assert governor.critical_ram_mb == 200
    assert decision.allowed is True


def test_configured_critical_ram_is_reported_as_admission_reserve():
    monitor = _MutableMonitor(available_ram_mb=199, free_vram_mb=5_994)
    governor = ResourceGovernorService(
        monitor=monitor,
        preset="balanced",
        critical_ram_mb=200,
        critical_ram_percent=3,
    )

    decision = governor.evaluate(
        InferenceRequest(
            workload="ambient_inference_batch",
            model_name="ambient-model",
            background=True,
            user_active=True,
        )
    )

    assert decision.allowed is False
    assert "200 MB is reserved" in decision.reason


def test_configured_critical_vram_caps_preset_reserve_after_model_load():
    monitor = _MutableMonitor(available_ram_mb=4_000, free_vram_mb=5_994)
    provider = _ModelProvider()
    governor = ResourceGovernorService(
        monitor=monitor,
        preset="balanced",
        critical_ram_mb=200,
        critical_ram_percent=3,
        critical_vram_mb=512,
    )
    manager = ModelResidencyManager(provider=provider, governor=governor)

    original_load = provider.load_model

    async def load_and_leave_604_mb_vram(model_name):
        await original_load(model_name)
        monitor.free_vram_mb = 604

    provider.load_model = load_and_leave_604_mb_vram

    decision = asyncio.run(
        manager.load_model("chat-model", role="direct_chat", user_active=True)
    )

    assert decision.allowed is True
    assert provider.current == "chat-model"
    assert provider.unloads == []
    assert governor.status()["thresholds"]["critical_vram_mb"] == 512


def test_resource_deferral_is_visible_and_audit_failure_is_nonfatal(caplog):
    monitor = _MutableMonitor(available_ram_mb=1_500, free_vram_mb=5_000)

    def broken_audit(*_args):
        raise OSError("audit database unavailable")

    governor = ResourceGovernorService(
        monitor=monitor,
        preset="balanced",
        audit=broken_audit,
    )
    request = InferenceRequest(
        workload="ambient_inference_batch",
        model_name="unknown-research-model",
        background=True,
        user_active=True,
    )

    with caplog.at_level(logging.WARNING):
        lease = governor.request_lease(request)

    assert lease.acquired is False
    assert "Capture remains active" in caplog.text
    assert "Resource audit failed" in caplog.text


def test_missing_gpu_telemetry_denies_heavy_work_but_allows_declared_light_work():
    monitor = _MutableMonitor(available_ram_mb=14_000, free_vram_mb=None)
    governor = ResourceGovernorService(monitor=monitor)
    gpu_request = InferenceRequest("vision", "vision-model", True, True)
    assert governor.evaluate(gpu_request).allowed is False

    light_request = InferenceRequest("local_artifact", "none", True, True, heavy=False)
    assert governor.evaluate(light_request).allowed is True


def test_tiny_model_evicts_under_pressure_and_reloads_after_recovery():
    monitor = _MutableMonitor(available_ram_mb=12_000, free_vram_mb=5_000)
    provider = _ModelProvider()
    governor = ResourceGovernorService(monitor=monitor)
    manager = ModelResidencyManager(
        provider=provider,
        governor=governor,
        lightweight_chat_model="tiny-q4",
        recovery_stable_seconds=0,
        transition_cooldown_seconds=0,
    )

    assert asyncio.run(manager.ensure_lightweight_resident(user_active=True, startup=True)) is True
    assert provider.current == "tiny-q4"

    monitor.available_ram_mb = 1_500
    assert asyncio.run(manager.evict_if_critical()) is True
    assert provider.current is None

    monitor.available_ram_mb = 12_000
    assert asyncio.run(manager.ensure_lightweight_resident(user_active=True)) is True
    assert provider.loads == ["tiny-q4", "tiny-q4"]


def test_low_memory_session_never_calls_model_provider():
    monitor = _MutableMonitor(available_ram_mb=1_000, free_vram_mb=5_000)
    provider = _ModelProvider()
    governor = ResourceGovernorService(monitor=monitor)
    manager = ModelResidencyManager(
        provider=provider,
        governor=governor,
    )

    decision = asyncio.run(
        manager.load_model("vision-model", role="vision", background=True, user_active=True)
    )
    assert decision.allowed is False
    assert provider.loads == []


def test_actual_post_load_pressure_unloads_unknown_model():
    monitor = _MutableMonitor(available_ram_mb=4_000, free_vram_mb=5_000)
    provider = _ModelProvider()
    governor = ResourceGovernorService(monitor=monitor)
    manager = ModelResidencyManager(provider=provider, governor=governor)
    original_load = provider.load_model

    async def load_and_consume_resources(model_name):
        await original_load(model_name)
        monitor.free_vram_mb = 200

    provider.load_model = load_and_consume_resources
    decision = asyncio.run(
        manager.load_model("previously-unknown-model", role="vision", background=True, user_active=True)
    )

    assert decision.allowed is False
    assert provider.current is None
    assert provider.unloads == ["previously-unknown-model"]


def test_resident_model_is_reused_and_uses_post_load_floor():
    monitor = _MutableMonitor(available_ram_mb=3_000, free_vram_mb=5_000)
    provider = _ModelProvider()
    governor = ResourceGovernorService(monitor=monitor)
    manager = ModelResidencyManager(provider=provider, governor=governor)
    governor.set_residency_status_provider(manager.status)
    original_load = provider.load_model

    async def load_and_consume_host_ram(model_name):
        await original_load(model_name)
        monitor.available_ram_mb = 1_500

    provider.load_model = load_and_consume_host_ram
    first = asyncio.run(
        manager.load_model("ambient-vision", role="research", background=True, user_active=True)
    )
    second = asyncio.run(
        manager.load_model("ambient-vision", role="research", background=True, user_active=True)
    )

    assert first.allowed is True
    assert second.allowed is True
    assert provider.loads == ["ambient-vision"]
    assert provider.unloads == []
    assert governor.evaluate(
        InferenceRequest("ambient", "different-model", True, True), force_snapshot=True
    ).allowed is False


def test_saved_model_restore_is_resource_gated():
    monitor = _MutableMonitor(available_ram_mb=14_000, free_vram_mb=5_000)
    provider = _ModelProvider()
    provider.current = "vision-model"
    governor = ResourceGovernorService(monitor=monitor)
    manager = ModelResidencyManager(
        provider=provider,
        governor=governor,
    )
    asyncio.run(manager.save_and_unload([]))
    monitor.available_ram_mb = 1_000
    monitor.free_vram_mb = 5_000

    try:
        asyncio.run(manager.load_and_restore(background=True, user_active=True))
    except ResourceUnavailableError as exc:
        assert "RAM" in exc.decision.reason
    else:
        raise AssertionError("unsafe saved-model restore was allowed")
    assert provider.current is None


def test_resource_deferral_does_not_consume_attempts_or_dead_letter(tmp_path):
    store = SQLiteAutonomyAdapter(str(tmp_path / "autonomy.db"))
    now = datetime.now(timezone.utc).isoformat()
    event = AmbientEvent(
        event_id="deferred-event",
        event_type="lightweight_visual_capture",
        source_kind="screen_capture",
        source_ref="capture://00000000000000000000000000000001",
        occurred_at=now,
        payload_json="{}",
        confidence=0.55,
        privacy_label="sensitive_visual",
        fingerprint="deferred-fingerprint",
        available_at=now,
    )
    store.enqueue_event(event)

    for _ in range(5):
        claimed = store.claim_next_event()
        assert claimed is not None
        store.defer_event(claimed.event_id, reason="RAM pressure", delay_seconds=1)
        with store._connect() as conn:
            conn.execute(
                "UPDATE ambient_events SET available_at=? WHERE event_id=?",
                ((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(), claimed.event_id),
            )

    reclaimed = store.claim_next_event()
    assert reclaimed is not None
    assert reclaimed.attempt_count == 1
    store.defer_event(reclaimed.event_id, reason="RAM pressure", delay_seconds=1)
    assert store.event_counts()["resource_deferred"] == 1


def test_lightweight_visual_capture_is_durable_without_model_inference(tmp_path):
    store = SQLiteAutonomyAdapter(str(tmp_path / "autonomy.db"))
    coordinator = AutonomyCoordinatorService(
        store=store,
        judgment=object(),
        policy=object(),
        mode="shadow",
    )
    captured_at = datetime.now(timezone.utc).isoformat()

    saved = coordinator.enqueue_lightweight_visual(
        screenshot_ref="capture://00000000000000000000000000000001",
        captured_at=captured_at,
        context={
            "app_name": "browser.exe",
            "window_title": "Useful article",
            "url": "https://example.test/article",
            "accessible_text": "Article heading",
        },
    )

    assert saved.event_type == "lightweight_visual_capture"
    assert saved.status == "pending"
    payload = json.loads(saved.payload_json)
    assert payload["screenshot_ref"] == "capture://00000000000000000000000000000001"
    assert payload["capture_mode"] == "lightweight"


def test_loopback_resource_status_and_preset_controls():
    governor = ResourceGovernorService(monitor=_MutableMonitor(), preset="balanced")
    app = create_runtime_log_app(
        RuntimeLogBuffer(),
        resource_governor=governor,
    )
    client = TestClient(app)

    status = client.get("/api/runtime/resources")
    assert status.status_code == 200
    assert status.json()["snapshot"]["available_ram_mb"] == 12_000
    changed = client.put(
        "/api/runtime/resource-policy",
        json={"preset": "capture_only"},
    )
    assert changed.status_code == 200
    assert changed.json()["preset"] == "capture_only"


def test_chat_preempts_background_batch_between_events():
    coordinator = AutonomyCoordinatorService(
        store=object(), judgment=object(), policy=object(), mode="shadow"
    )
    processed = []

    async def process_next(**kwargs):
        processed.append(kwargs)
        return {"processed": True, "outcome": "completed"}

    coordinator.process_next = process_next
    result = asyncio.run(
        coordinator.process_batch(
            model="judgment",
            llm_service=object(),
            personalization_context="",
            max_events=8,
            max_seconds=90,
            should_preempt=lambda: len(processed) >= 1,
        )
    )

    assert result["count"] == 1
    assert result["preempted"] is True
