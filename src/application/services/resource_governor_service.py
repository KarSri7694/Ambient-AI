import asyncio
import logging
import threading
import time
from dataclasses import asdict
from typing import Any, Callable, Optional

from application.ports.resource_monitor_port import ResourceMonitorPort
from core.models import (
    InferenceRequest,
    ResourceDecision,
    ResourceSnapshot,
)


RESOURCE_PRESETS = {
    "capture_only": {
        "active_ram_mb": 4096,
        "active_ram_percent": 30.0,
        "idle_ram_mb": 2048,
        "idle_ram_percent": 20.0,
        "active_vram_mb": 768,
        "idle_vram_mb": 512,
        "gpu_active_ram_mb": 2048,
        "gpu_idle_ram_mb": 1536,
        "post_load_active_ram_mb": 1024,
        "post_load_idle_ram_mb": 768,
        "background_enabled": False,
    },
    "balanced": {
        "active_ram_mb": 4096,
        "active_ram_percent": 30.0,
        "idle_ram_mb": 2048,
        "idle_ram_percent": 20.0,
        "active_vram_mb": 768,
        "idle_vram_mb": 512,
        "gpu_active_ram_mb": 2048,
        "gpu_idle_ram_mb": 1536,
        "post_load_active_ram_mb": 1024,
        "post_load_idle_ram_mb": 768,
        "background_enabled": True,
    },
    "aggressive": {
        "active_ram_mb": 3072,
        "active_ram_percent": 22.0,
        "idle_ram_mb": 1536,
        "idle_ram_percent": 15.0,
        "active_vram_mb": 512,
        "idle_vram_mb": 384,
        "gpu_active_ram_mb": 1536,
        "gpu_idle_ram_mb": 1024,
        "post_load_active_ram_mb": 768,
        "post_load_idle_ram_mb": 512,
        "background_enabled": True,
    },
}


class ResourceUnavailableError(RuntimeError):
    def __init__(self, decision: ResourceDecision):
        self.decision = decision
        super().__init__(decision.reason)


class InferenceLease:
    def __init__(self, governor: "ResourceGovernorService", request: InferenceRequest, decision: ResourceDecision):
        self.governor = governor
        self.request = request
        self.decision = decision
        self.acquired = decision.allowed

    def __enter__(self) -> "InferenceLease":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.acquired:
            self.governor.release(self.request, error=str(exc) if exc else None)
            self.acquired = False


class ResourceGovernorService:
    """Decides whether one bounded model workload fits without harming laptop use."""

    def __init__(
        self,
        *,
        monitor: ResourceMonitorPort,
        preset: str = "balanced",
        critical_ram_mb: int = 2048,
        critical_ram_percent: float = 15.0,
        critical_vram_mb: int = 512,
        audit: Optional[Callable[[str, str, str, dict], None]] = None,
        logger: logging.Logger | None = None,
    ):
        self.monitor = monitor
        self._preset = preset if preset in RESOURCE_PRESETS else "balanced"
        # An explicit runtime threshold must be allowed to lower the preset's
        # host-RAM reserve.  The previous 512 MB clamp also meant values such as
        # 200 MB could never take effect even though config.py loaded them.
        self.critical_ram_mb = max(1, int(critical_ram_mb))
        self.critical_ram_percent = max(1.0, float(critical_ram_percent))
        self.critical_vram_mb = max(1, int(critical_vram_mb))
        self.audit = audit
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._lease_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._active_request: Optional[InferenceRequest] = None
        self._active_started_at: Optional[float] = None
        self._active_start_snapshot: Optional[ResourceSnapshot] = None
        self._last_decision: Optional[ResourceDecision] = None
        self._deferred_count = 0
        self._residency_status_provider: Optional[Callable[[], dict[str, Any]]] = None
        self._rocm_profile_provider: Optional[Callable[[str], dict[str, Any] | None]] = None
        self._last_deferral_log_at = 0.0
        self._last_deferral_reason = ""

    @property
    def preset(self) -> str:
        with self._state_lock:
            return self._preset

    def set_preset(self, preset: str) -> str:
        if preset not in RESOURCE_PRESETS:
            raise ValueError("preset must be capture_only, balanced, or aggressive")
        with self._state_lock:
            self._preset = preset
        self._audit("resource.preset_changed", preset, {})
        return preset

    def set_residency_status_provider(self, provider: Callable[[], dict[str, Any]]) -> None:
        self._residency_status_provider = provider

    def set_rocm_profile_provider(self, provider: Callable[[str], dict[str, Any] | None]) -> None:
        self._rocm_profile_provider = provider

    def _host_ram_reserve(self, preset_reserve_mb: int) -> int:
        """Apply the configured RAM threshold as an upper bound on a preset."""
        return min(int(preset_reserve_mb), self.critical_ram_mb)

    def _vram_reserve(self, preset_reserve_mb: int) -> int:
        """Apply the configured VRAM threshold as an upper bound on a preset."""
        return min(int(preset_reserve_mb), self.critical_vram_mb)

    def evaluate(
        self,
        request: InferenceRequest,
        *,
        force_snapshot: bool = False,
        post_load: bool = False,
    ) -> ResourceDecision:
        snapshot = self.monitor.snapshot(user_idle=not request.user_active, force=force_snapshot)
        preset = RESOURCE_PRESETS[self.preset]
        loaded_model = None
        if self._residency_status_provider is not None:
            try:
                loaded_model = self._residency_status_provider().get("loaded_model")
            except Exception:
                self.logger.exception("Unable to read model residency status during resource evaluation.")
        use_post_load_floor = post_load or loaded_model == request.model_name
        reason = "live RAM and VRAM headroom are sufficient"
        allowed = True
        if request.background and not preset["background_enabled"]:
            allowed, reason = False, "capture-only preset defers background inference"
        elif snapshot.total_ram_mb <= 0:
            allowed, reason = False, "RAM telemetry is unavailable"
        elif request.heavy and not snapshot.gpu_telemetry_available:
            allowed, reason = False, "GPU telemetry is unavailable for model inference"
        else:
            if snapshot.gpu_telemetry_available:
                preset_vram_reserve = (
                    preset["active_vram_mb"] if request.user_active else preset["idle_vram_mb"]
                )
                vram_reserve = self._vram_reserve(preset_vram_reserve)
                if use_post_load_floor:
                    preset_ram_reserve = (
                        preset["post_load_active_ram_mb"]
                        if request.user_active
                        else preset["post_load_idle_ram_mb"]
                    )
                else:
                    preset_ram_reserve = (
                        preset["gpu_active_ram_mb"] if request.user_active else preset["gpu_idle_ram_mb"]
                    )
                gpu_ram_reserve = self._host_ram_reserve(preset_ram_reserve)
                if snapshot.free_vram_mb is None or snapshot.free_vram_mb < int(vram_reserve):
                    allowed = False
                    reason = f"only {snapshot.free_vram_mb or 0} MB VRAM is free; {vram_reserve} MB is reserved"
                elif (
                    not use_post_load_floor
                    and snapshot.gpu_backend == "amd_rocm"
                    and self._measured_rocm_vram_requirement_mb(request.model_name, int(vram_reserve)) is not None
                    and snapshot.free_vram_mb < self._measured_rocm_vram_requirement_mb(request.model_name, int(vram_reserve))
                ):
                    required_vram = self._measured_rocm_vram_requirement_mb(request.model_name, int(vram_reserve)) or vram_reserve
                    allowed = False
                    reason = (
                        f"only {snapshot.free_vram_mb} MB VRAM is free; tuned ROCm profile for "
                        f"{request.model_name} needs {required_vram} MB including reserve"
                    )
                elif snapshot.available_ram_mb < int(gpu_ram_reserve):
                    allowed = False
                    reason = (
                        f"only {snapshot.available_ram_mb} MB host RAM is free; "
                        f"{gpu_ram_reserve} MB is reserved"
                    )
            else:
                preset_ram_reserve = (
                    preset["active_ram_mb"] if request.user_active else preset["idle_ram_mb"]
                )
                ram_reserve_mb = self._host_ram_reserve(preset_ram_reserve)
                ram_reserve_percent = (
                    preset["active_ram_percent"] if request.user_active else preset["idle_ram_percent"]
                )
                required_percent_mb = int(snapshot.total_ram_mb * ram_reserve_percent / 100.0)
                required_ram = max(int(ram_reserve_mb), required_percent_mb)
                if snapshot.available_ram_mb < required_ram:
                    allowed = False
                    reason = f"only {snapshot.available_ram_mb} MB RAM is free; {required_ram} MB is reserved"
        decision = ResourceDecision(
            allowed=allowed,
            reason=reason,
            preset=self.preset,
            snapshot=snapshot,
            available_ram_mb=snapshot.available_ram_mb,
            free_vram_mb=snapshot.free_vram_mb,
        )
        with self._state_lock:
            self._last_decision = decision
            if not allowed:
                self._deferred_count += 1
        return decision

    def verify_after_load(self, request: InferenceRequest) -> ResourceDecision:
        """Verify measured post-load headroom and reject unsafe residency."""
        decision = self.evaluate(request, force_snapshot=True, post_load=True)
        self._audit(
            "resource.post_load_verified" if decision.allowed else "resource.post_load_unsafe",
            request.model_name,
            self._decision_details(request, decision),
        )
        return decision

    def request_lease(self, request: InferenceRequest) -> InferenceLease:
        decision = self.evaluate(request, force_snapshot=True)
        if not decision.allowed:
            self._log_deferral(request, decision)
            self._audit("resource.inference_deferred", request.workload, self._decision_details(request, decision))
            return InferenceLease(self, request, decision)
        if not self._lease_lock.acquire(blocking=False):
            busy = ResourceDecision(
                allowed=False,
                reason="another model workload owns the inference lease",
                preset=decision.preset,
                snapshot=decision.snapshot,
                available_ram_mb=decision.available_ram_mb,
                free_vram_mb=decision.free_vram_mb,
            )
            with self._state_lock:
                self._last_decision = busy
                self._deferred_count += 1
            return InferenceLease(self, request, busy)
        with self._state_lock:
            self._active_request = request
            self._active_started_at = time.monotonic()
            self._active_start_snapshot = decision.snapshot
        self._audit("resource.lease_acquired", request.workload, self._decision_details(request, decision))
        return InferenceLease(self, request, decision)

    def release(self, request: InferenceRequest, *, error: str | None = None) -> None:
        with self._state_lock:
            started_at = self._active_started_at
            start_snapshot = self._active_start_snapshot
            self._active_request = None
            self._active_started_at = None
            self._active_start_snapshot = None
        if self._lease_lock.locked():
            self._lease_lock.release()
        end_snapshot = self.monitor.snapshot(user_idle=not request.user_active, force=True)
        observed_available_ram = [
            item.available_ram_mb for item in (start_snapshot, end_snapshot) if item is not None
        ]
        observed_free_vram = [
            item.free_vram_mb for item in (start_snapshot, end_snapshot)
            if item is not None and item.free_vram_mb is not None
        ]
        self._audit(
            "resource.lease_released",
            request.workload,
            {
                "error": error,
                "duration_seconds": (time.monotonic() - started_at) if started_at is not None else None,
                "minimum_observed_available_ram_mb": min(observed_available_ram) if observed_available_ram else None,
                "minimum_observed_free_vram_mb": min(observed_free_vram) if observed_free_vram else None,
                "end_snapshot": asdict(end_snapshot),
            },
        )

    def is_critical(self, snapshot: ResourceSnapshot | None = None) -> bool:
        current = snapshot or self.monitor.snapshot(force=True)
        return (
            current.available_ram_mb < self.critical_ram_mb
            and current.available_ram_percent < self.critical_ram_percent
        )

    def status(self, *, user_idle: bool = False) -> dict[str, Any]:
        snapshot = self.monitor.snapshot(user_idle=user_idle)
        with self._state_lock:
            active = self._active_request
            decision = self._last_decision
            deferred = self._deferred_count
        payload = {
            "preset": self.preset,
            "thresholds": {
                "critical_ram_mb": self.critical_ram_mb,
                "critical_ram_percent": self.critical_ram_percent,
                "critical_vram_mb": self.critical_vram_mb,
            },
            "snapshot": asdict(snapshot),
            "critical_pressure": self.is_critical(snapshot),
            "active_workload": active.workload if active else None,
            "last_decision": asdict(decision) if decision else None,
            "deferred_decisions": deferred,
        }
        if self._residency_status_provider is not None:
            payload["residency"] = self._residency_status_provider()
        if self._rocm_profile_provider is not None and snapshot.gpu_backend == "amd_rocm":
            loaded_model = None
            if isinstance(payload.get("residency"), dict):
                loaded_model = payload["residency"].get("loaded_model")
            payload["rocm_tuned_profile"] = self._rocm_profile_provider(str(loaded_model or ""))
        return payload

    def _decision_details(self, request: InferenceRequest, decision: ResourceDecision) -> dict[str, Any]:
        details = {"request": asdict(request), "decision": asdict(decision)}
        if self._rocm_profile_provider is not None:
            details["rocm_tuned_profile"] = self._rocm_profile_provider(request.model_name)
        return details

    def _measured_rocm_vram_requirement_mb(self, model_name: str, reserve_mb: int) -> Optional[int]:
        if self._rocm_profile_provider is None:
            return None
        try:
            profile = self._rocm_profile_provider(model_name)
        except Exception:
            self.logger.exception("Unable to read ROCm tuned profile for %s.", model_name)
            return None
        if not profile:
            return None
        summary = profile.get("summary") or {}
        delta = summary.get("vram_delta_mb_max")
        if delta is None:
            return None
        return max(reserve_mb, int(float(delta)) + reserve_mb)

    def _log_deferral(self, request: InferenceRequest, decision: ResourceDecision) -> None:
        now = time.monotonic()
        if (
            decision.reason == self._last_deferral_reason
            and now - self._last_deferral_log_at < 60.0
        ):
            return
        self._last_deferral_reason = decision.reason
        self._last_deferral_log_at = now
        snapshot = decision.snapshot
        self.logger.warning(
            "Deferred %s: %s (free RAM=%s MB/%.1f%%, free VRAM=%s MB, preset=%s). Capture remains active.",
            request.workload,
            decision.reason,
            snapshot.available_ram_mb,
            snapshot.available_ram_percent,
            snapshot.free_vram_mb if snapshot.free_vram_mb is not None else "unavailable",
            decision.preset,
        )

    def _audit(self, action: str, target: str, details: dict[str, Any]) -> None:
        if self.audit is not None:
            try:
                self.audit("resource_governor", action, target, details)
            except Exception:
                self.logger.exception("Resource audit failed for %s/%s; runtime work will continue.", action, target)


class ModelResidencyManager:
    """Serialize necessary model transitions while preserving useful residency."""

    def __init__(
        self,
        *,
        provider,
        governor: ResourceGovernorService,
        lightweight_chat_model: str = "",
        keep_active_model_resident: bool = True,
        recovery_stable_seconds: float = 30.0,
        transition_cooldown_seconds: float = 60.0,
        logger: logging.Logger | None = None,
    ):
        self.provider = provider
        self.governor = governor
        self.lightweight_chat_model = str(lightweight_chat_model or "").strip()
        self.keep_active_model_resident = bool(keep_active_model_resident)
        self.recovery_stable_seconds = max(0.0, float(recovery_stable_seconds))
        self.transition_cooldown_seconds = max(0.0, float(transition_cooldown_seconds))
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._transition_lock = asyncio.Lock()
        self._last_transition_at = 0.0
        self._healthy_since: Optional[float] = None
        self._saved_model_stack: list[str] = []

    async def load_model(
        self,
        model_name: str,
        *,
        role: str = "interactive",
        background: bool = False,
        user_active: bool = True,
    ) -> ResourceDecision:
        request = InferenceRequest(
            workload=role,
            model_name=model_name,
            background=background,
            user_active=user_active,
            priority=100 if not background else 50,
        )
        decision = self.governor.evaluate(request, force_snapshot=True)
        if not decision.allowed:
            self.governor._log_deferral(request, decision)
            self.governor._audit("resource.model_load_denied", model_name, self.governor._decision_details(request, decision))
            return decision
        async with self._transition_lock:
            before = self.provider.get_current_model()
            if before == model_name:
                resident_decision = self.governor.verify_after_load(request)
                if not resident_decision.allowed:
                    self.governor._audit(
                        "resource.resident_model_headroom_low",
                        model_name,
                        {
                            "role": role,
                            "reason": resident_decision.reason,
                            "post_load_decision": asdict(resident_decision),
                        },
                    )
                    return resident_decision
                self.governor._audit(
                    "resource.model_reused",
                    model_name,
                    {
                        "role": role,
                        "post_load_decision": asdict(resident_decision),
                    },
                )
                return resident_decision
            started_at = time.monotonic()
            await self.provider.load_model(model_name)
            after = self.provider.get_current_model()
            if after != model_name:
                return ResourceDecision(
                    allowed=False,
                    reason=f"model provider did not confirm that {model_name} loaded",
                    preset=decision.preset,
                    snapshot=decision.snapshot,
                    available_ram_mb=decision.available_ram_mb,
                    free_vram_mb=decision.free_vram_mb,
                )
            post_load_decision = self.governor.verify_after_load(request)
            if not post_load_decision.allowed:
                await self.provider.unload_model()
                self._last_transition_at = time.monotonic()
                self.governor._log_deferral(request, post_load_decision)
                self.governor._audit(
                    "resource.model_unloaded_after_unsafe_load",
                    model_name,
                    self.governor._decision_details(request, post_load_decision),
                )
                return post_load_decision
            self._last_transition_at = time.monotonic()
            self.governor._audit(
                "resource.model_loaded",
                model_name,
                {
                    "previous_model": before,
                    "role": role,
                    "duration_seconds": time.monotonic() - started_at,
                    "pre_load_decision": asdict(decision),
                    "post_load_decision": asdict(post_load_decision),
                },
            )
        return post_load_decision

    async def unload_model(self, *, reason: str = "resource release") -> None:
        async with self._transition_lock:
            loaded = self.provider.get_current_model()
            if loaded is None:
                return
            started_at = time.monotonic()
            await self.provider.unload_model()
            self._last_transition_at = time.monotonic()
            self.governor._audit(
                "resource.model_unloaded",
                loaded,
                {"reason": reason, "duration_seconds": time.monotonic() - started_at},
            )

    async def save_and_unload(self, messages):
        async with self._transition_lock:
            loaded = self.provider.get_current_model()
            result = await self.provider.save_and_unload(messages)
            self._last_transition_at = time.monotonic()
            if loaded and result is not None:
                self._saved_model_stack.append(loaded)
                self.governor._audit("resource.model_saved_and_unloaded", loaded, {})
            return result

    async def load_and_restore(
        self,
        *,
        role: str = "state_restore",
        background: bool = False,
        user_active: bool = True,
    ):
        target_model = self._saved_model_stack[-1] if self._saved_model_stack else ""
        if not target_model:
            raise RuntimeError("No saved model residency is available to restore")
        request = InferenceRequest(
            workload=role,
            model_name=target_model,
            background=background,
            user_active=user_active,
            priority=100 if not background else 50,
        )
        decision = self.governor.evaluate(request, force_snapshot=True)
        if not decision.allowed:
            self.governor._audit(
                "resource.model_restore_denied",
                target_model,
                self.governor._decision_details(request, decision),
            )
            raise ResourceUnavailableError(decision)
        async with self._transition_lock:
            started_at = time.monotonic()
            result = await self.provider.load_and_restore()
            self._last_transition_at = time.monotonic()
            loaded = self.provider.get_current_model()
            if loaded != target_model:
                raise RuntimeError(f"provider restored {loaded or 'no model'} instead of {target_model}")
            post_load_decision = self.governor.verify_after_load(request)
            if not post_load_decision.allowed:
                await self.provider.unload_model()
                raise ResourceUnavailableError(post_load_decision)
            self._saved_model_stack.pop()
            self.governor._audit(
                "resource.model_restored",
                loaded,
                {
                    "duration_seconds": time.monotonic() - started_at,
                    "pre_load_decision": asdict(decision),
                    "post_load_decision": asdict(post_load_decision),
                },
            )
            return result

    async def evict_if_critical(self) -> bool:
        snapshot = self.governor.monitor.snapshot()
        if not self.governor.is_critical(snapshot):
            if self._healthy_since is None:
                self._healthy_since = time.monotonic()
            return False
        self._healthy_since = None
        if self.provider.get_current_model() is not None:
            await self.unload_model(reason="critical memory pressure")
            return True
        return False

    async def ensure_lightweight_resident(self, *, user_active: bool, startup: bool = False) -> bool:
        if not self.lightweight_chat_model:
            return False
        if await self.evict_if_critical():
            return False
        if self.provider.get_current_model() == self.lightweight_chat_model:
            return True
        now = time.monotonic()
        if self._healthy_since is None:
            self._healthy_since = now
        if not startup and now - self._healthy_since < self.recovery_stable_seconds:
            return False
        if not startup and now - self._last_transition_at < self.transition_cooldown_seconds:
            return False
        decision = await self.load_model(
            self.lightweight_chat_model,
            role="lightweight_chat",
            background=False,
            user_active=user_active,
        )
        return decision.allowed

    async def settle_to_lightweight(self, *, user_active: bool) -> bool:
        if await self.evict_if_critical():
            return False
        loaded = self.provider.get_current_model()
        if not self.lightweight_chat_model:
            # With no dedicated lightweight model configured, retaining the
            # current healthy model avoids unloading it at every brief gap
            # between screenshots, autonomy events, and chat turns.
            if self.keep_active_model_resident:
                return loaded is not None
            if loaded:
                await self.unload_model(reason="interactive response window ended")
            return False
        if loaded and loaded != self.lightweight_chat_model:
            await self.unload_model(reason="interactive response window ended")
        return await self.ensure_lightweight_resident(user_active=user_active, startup=False)

    def status(self) -> dict[str, Any]:
        return {
            "loaded_model": self.provider.get_current_model(),
            "lightweight_chat_model": self.lightweight_chat_model or None,
            "keep_active_model_resident": self.keep_active_model_resident,
        }
