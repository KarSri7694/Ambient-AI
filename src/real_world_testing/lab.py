from __future__ import annotations

import asyncio
import configparser
import csv
import hashlib
import io
import json
import logging
import re
import shutil
import threading
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from real_world_testing.case_loader import (
    AUDIO_SUFFIXES, IMAGE_SUFFIXES, RealWorldScenario, RealWorldSuite,
    ScheduledMediaInput, load_suites, suite_to_dict,
)
from real_world_testing.store import SQLiteRealWorldTestStore, TERMINAL_STATUSES

try:
    from infrastructure.accelerator import detect_accelerator
except Exception:
    detect_accelerator = None


LOGGER = logging.getLogger(__name__)
MODEL_ROLE_NAMES = (
    "passive_observer_model", "full_passive_observer_model", "passive_followup_model",
    "followup_execution_model", "transcript_processing_model", "reporter_model", "browser_agent_model",
)


class RunCancelled(RuntimeError):
    pass


class RealWorldLab:
    def __init__(self, *, project_root: str | Path, config_path: str | Path,
                 data_root: str | Path, suites_root: str | Path,
                 executor_factory: Callable[..., Any] | None = None):
        self.project_root = Path(project_root).resolve()
        self.config_path = Path(config_path).resolve()
        self.data_root = Path(data_root).resolve()
        self.suites_root = Path(suites_root).resolve()
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.media_root = self.data_root / "media"
        self.runs_root = self.data_root / "runs"
        self.media_root.mkdir(parents=True, exist_ok=True)
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.store = SQLiteRealWorldTestStore(self.data_root / "real_world_tests.db")
        self.executor_factory = executor_factory or ProductionScenarioExecutor
        self._thread: threading.Thread | None = None
        self._active_run_id: str | None = None
        self._lock = threading.RLock()

    def suites(self) -> list[dict[str, Any]]:
        return [suite_to_dict(item) for item in load_suites(self.suites_root)]

    def models(self) -> dict[str, Any]:
        parser = self._config()
        default_model = parser.get("runtime", "default_model", fallback="").strip()
        roles = {
            role: parser.get("models", role, fallback=default_model).strip() or default_model
            for role in MODEL_ROLE_NAMES
        }
        preset_path = self.project_root / "models_preset.ini"
        presets: list[str] = []
        if preset_path.exists():
            # llama.cpp preset files allow global key/value lines before their
            # model sections, which ConfigParser intentionally rejects.
            presets = [
                match.group(1).strip()
                for line in preset_path.read_text(encoding="utf-8").splitlines()
                if (match := re.match(r"^\s*\[([^]]+)]\s*$", line))
            ]
        return {"roles": roles, "presets": presets, "config_path": str(self.config_path)}

    def upload_media(self, *, filename: str, kind: str, data: bytes) -> dict[str, Any]:
        original = Path(filename).name
        suffix = Path(original).suffix.lower()
        allowed = IMAGE_SUFFIXES if kind == "image" else AUDIO_SUFFIXES if kind == "audio" else set()
        if suffix not in allowed:
            raise ValueError(f"Unsupported {kind} media type: {suffix or 'none'}")
        limit = 25 * 1024 * 1024 if kind == "image" else 250 * 1024 * 1024
        if not data or len(data) > limit:
            raise ValueError(f"{kind} upload must be between 1 byte and {limit // (1024 * 1024)} MB")
        digest = hashlib.sha256(data).hexdigest()
        destination = self.media_root / f"{digest}{suffix}"
        if not destination.exists():
            destination.write_bytes(data)
        return self.store.register_media(kind=kind, original_name=original, path=destination)

    def start_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("live_tools_confirmation") != "RUN LIVE TOOLS":
            raise ValueError("Type RUN LIVE TOOLS to confirm real MCP tool execution")
        speed = float(payload.get("playback_speed", 1.0))
        if speed not in {1.0, 2.0, 5.0, 10.0}:
            raise ValueError("playback_speed must be one of 1, 2, 5, or 10")
        suite, scenarios = self._resolve_scenarios(payload)
        model_info = self.models()
        roles = dict(model_info["roles"])
        overrides = dict(payload.get("model_overrides") or {})
        unknown_roles = set(overrides) - set(MODEL_ROLE_NAMES)
        if unknown_roles:
            raise ValueError(f"Unknown model role(s): {', '.join(sorted(unknown_roles))}")
        valid_models = set(model_info["presets"]) | set(roles.values())
        for role, model in overrides.items():
            if str(model) not in valid_models:
                raise ValueError(f"Unknown model preset for {role}: {model}")
            roles[role] = str(model)
        parser = self._config()
        if parser.get("autonomy", "mode", fallback="shadow").strip() != "active":
            raise ValueError("Real live-tool runs require [autonomy] mode = active")
        accelerator = self.accelerator_summary(parser)
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError(f"Run {self._active_run_id} is already active")
            run_id = self.store.create_run(
                suite_id=suite.suite_id, scenario_ids=[item.scenario_id for item in scenarios],
                playback_speed=speed, model_roles=roles,
                config={"config_path": str(self.config_path), "live_tools": True, "accelerator": accelerator},
            )
            self._active_run_id = run_id
            self._thread = threading.Thread(
                target=self._run_thread, args=(run_id, scenarios, speed, roles),
                name=f"RealWorldRun-{run_id[:8]}", daemon=True,
            )
            self._thread.start()
        return self.store.get_run(run_id) or {"run_id": run_id}

    def accelerator_summary(self, parser: configparser.ConfigParser | None = None) -> dict[str, Any]:
        parser = parser or self._config()
        backend = parser.get("accelerator", "backend", fallback="auto")
        require_supported = parser.getboolean("accelerator", "require_supported_gpu", fallback=False)
        if detect_accelerator is None:
            return {"available": False, "backend": backend, "reason": "accelerator module unavailable"}
        return detect_accelerator(backend, require_supported_gpu=require_supported).to_dict()

    def export_run(self, run_id: str) -> dict[str, Any]:
        run = self.store.get_run(run_id)
        if run is None:
            raise KeyError(run_id)
        return {"run": run, "events": self.store.list_events(run_id, limit=5000)}

    def export_run_csv(self, run_id: str) -> str:
        payload = self.export_run(run_id)
        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "sequence", "created_at", "stage", "event_type", "status", "model",
                "duration_ms", "result_id", "payload_json",
            ],
        )
        writer.writeheader()
        for event in payload["events"]:
            writer.writerow({
                "sequence": event.get("sequence"),
                "created_at": event.get("created_at"),
                "stage": event.get("stage"),
                "event_type": event.get("event_type"),
                "status": event.get("status"),
                "model": event.get("model"),
                "duration_ms": event.get("duration_ms"),
                "result_id": event.get("result_id"),
                "payload_json": json.dumps(event.get("payload") or {}, ensure_ascii=False),
            })
        return output.getvalue()

    def cancel_run(self, run_id: str) -> bool:
        return self.store.request_cancel(run_id)

    def wait_for_idle(self, timeout: float | None = None) -> bool:
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()

    def shutdown(self, timeout: float = 15.0) -> None:
        active = self._active_run_id
        if active:
            self.store.request_cancel(active)
        if not self.wait_for_idle(timeout):
            LOGGER.warning("Real-world run thread did not stop within %.1fs", timeout)

    def media_path(self, media_id: str) -> Path | None:
        media = self.store.get_media(media_id)
        if not media:
            return None
        candidate = Path(media["path"]).resolve()
        try:
            candidate.relative_to(self.media_root)
        except ValueError:
            return None
        return candidate if candidate.is_file() else None

    def scenario_media_path(self, run_id: str, result_id: str, index: int) -> Path | None:
        result = self.store.get_result(result_id)
        if not result or result["run_id"] != run_id or index < 0 or index >= len(result["media"]):
            return None
        candidate = Path(result["media"][index]["media_path"]).resolve()
        return candidate if candidate.is_file() else None

    def _run_thread(self, run_id: str, scenarios: list[RealWorldScenario], speed: float,
                    roles: dict[str, str]) -> None:
        self.store.update_run(run_id, "running")
        accelerator = self.accelerator_summary()
        self.store.append_event(
            run_id=run_id,
            result_id=None,
            stage="hardware",
            event_type="accelerator_detected",
            payload=accelerator,
            status="completed" if accelerator.get("available") else "failed",
        )
        errors = 0
        try:
            for scenario in scenarios:
                if self.store.is_cancel_requested(run_id):
                    raise RunCancelled("Run cancelled before next scenario")
                result_id = self.store.create_result(run_id, scenario)
                self.store.update_result(result_id, "running")
                workspace = self.runs_root / run_id / scenario.scenario_id
                workspace.mkdir(parents=True, exist_ok=True)
                emit = lambda stage, event_type, payload=None, **kwargs: self.store.append_event(
                    run_id=run_id, result_id=result_id, stage=stage, event_type=event_type,
                    payload=payload or {}, **kwargs,
                )
                executor = self.executor_factory(
                    project_root=self.project_root, config_path=self.config_path,
                    workspace=workspace, model_roles=roles, emit=emit,
                    should_cancel=lambda: self.store.is_cancel_requested(run_id),
                )
                try:
                    outcome = asyncio.run(executor.execute(scenario, playback_speed=speed))
                    self.store.update_result(
                        result_id, "completed", transcript_text=outcome.get("transcript_text"),
                        final_response=outcome.get("final_response"), summary=outcome.get("summary", {}),
                    )
                except RunCancelled:
                    self.store.update_result(result_id, "cancelled")
                    raise
                except Exception as exc:
                    errors += 1
                    LOGGER.exception("Real-world scenario %s failed", scenario.scenario_id)
                    emit("scenario", "scenario_failed", {"error": str(exc)}, status="failed")
                    self.store.update_result(result_id, "failed", error_text=str(exc))
            self.store.update_run(run_id, "completed_with_errors" if errors else "completed")
        except RunCancelled as exc:
            self.store.update_run(run_id, "cancelled", error_text=str(exc))
        except Exception as exc:
            LOGGER.exception("Real-world run %s failed", run_id)
            self.store.update_run(run_id, "failed", error_text=str(exc))
        finally:
            with self._lock:
                self._active_run_id = None

    def _resolve_scenarios(self, payload: dict[str, Any]) -> tuple[RealWorldSuite, list[RealWorldScenario]]:
        inline = payload.get("inline_scenario")
        if isinstance(inline, dict):
            modality = str(inline.get("modality") or "")
            if modality not in {"image_sequence", "audio_sequence"}:
                raise ValueError("inline_scenario modality must be image_sequence or audio_sequence")
            expected_kind = "image" if modality == "image_sequence" else "audio"
            events: list[ScheduledMediaInput] = []
            previous_offset = -1.0
            for item in inline.get("events", []):
                media = self.store.get_media(str(item.get("media_id") or ""))
                if not media or media["kind"] != expected_kind:
                    raise ValueError("Inline scenario contains missing or incompatible uploaded media")
                offset = float(item.get("offset_seconds", 0))
                if offset < 0 or offset < previous_offset:
                    raise ValueError("Inline scenario offsets must be non-negative and monotonic")
                previous_offset = offset
                events.append(ScheduledMediaInput(
                    offset_seconds=offset, media_path=str(media["path"]),
                    screen_context=dict(item.get("screen_context") or {}),
                ))
            if not events:
                raise ValueError("Inline scenario requires uploaded media")
            scenario = RealWorldScenario(
                scenario_id=f"ui_{uuid.uuid4().hex[:12]}",
                title=str(inline.get("title") or "Uploaded media scenario"),
                modality=modality, events=events,
                rubric_notes=str(inline.get("rubric_notes") or ""), source_path="ui-upload",
            )
            suite = RealWorldSuite(
                schema_version=1, suite_id="ui-upload", title="UI upload", description="",
                scenarios=[scenario], source_path="ui-upload",
            )
            return suite, [scenario]
        suite_id = str(payload.get("suite_id") or "").strip()
        suite = next((item for item in load_suites(self.suites_root) if item.suite_id == suite_id), None)
        if suite is None:
            raise ValueError(f"Unknown suite: {suite_id}")
        selected_ids = {str(item) for item in payload.get("scenario_ids", [])}
        scenarios = [item for item in suite.scenarios if not selected_ids or item.scenario_id in selected_ids]
        if not scenarios or (selected_ids - {item.scenario_id for item in scenarios}):
            raise ValueError("No valid scenarios were selected")
        for scenario in scenarios:
            missing = [event.media_path for event in scenario.events if not Path(event.media_path).is_file()]
            if missing:
                raise ValueError(f"Missing scenario media: {missing[0]}")
        return suite, scenarios

    def _config(self) -> configparser.ConfigParser:
        parser = configparser.ConfigParser()
        parser.read(self.config_path, encoding="utf-8")
        return parser


class _StaticScreenCapture:
    def __init__(self):
        self.path = ""
    def capture_screenshot(self, output_path=None) -> str:
        return self.path


class _InMemoryTaskQueue:
    def __init__(self):
        self.items: list[Any] = []
    def get_pending_tasks(self): return list(self.items)
    def get_all_pending_tasks(self): return list(self.items)
    def get_due_tasks(self, now_utc): return []
    def add_task(self, description, priority="medium", metadata=None):
        item = type("Task", (), {"description": description, "priority": priority,
                                  "metadata_json": json.dumps(metadata or {})})()
        self.items.append(item)
        return str(len(self.items))
    def mark_task_complete(self, task_id, status="completed"): return None


class ProductionScenarioExecutor:
    """Compose production services around prerecorded input adapters."""
    def __init__(self, *, project_root: Path, config_path: Path, workspace: Path,
                 model_roles: dict[str, str], emit, should_cancel):
        self.project_root = project_root
        self.config_path = config_path
        self.workspace = workspace
        self.model_roles = model_roles
        self.emit = emit
        self.should_cancel = should_cancel

    async def execute(self, scenario: RealWorldScenario, *, playback_speed: float) -> dict[str, Any]:
        from application.services.capability_policy_service import AutonomyBudget, CapabilityPolicyService
        from application.services.interaction_trace import interaction_trace
        from application.services.llm_interaction_service import LLMInteractionService
        from application.services.opportunity_judgment_service import OpportunityJudgmentService
        from application.services.autonomy_coordinator_service import AutonomyCoordinatorService
        from infrastructure.adapter.LlamaCppSemanticAdapter import LlamaCppSemanticAdapter
        from infrastructure.adapter.MCPToolAdapter import MCPToolAdapter
        from infrastructure.adapter.SQLiteAutonomyAdapter import SQLiteAutonomyAdapter
        from infrastructure.adapter.SQLiteMemoryAdapter import SQLiteMemoryAdapter
        from infrastructure.adapter.llamaCppAdapter import LlamaCppAdapter
        from infrastructure.plain_capture_store import PlainCaptureStore
        from real_world_testing.tracing_provider import RealWorldTracingLLMProvider

        parser = configparser.ConfigParser()
        parser.read(self.config_path, encoding="utf-8")
        api_url = parser.get("runtime", "api_base_url", fallback="http://127.0.0.1:8080")
        api_key = parser.get("runtime", "api_key", fallback="")
        raw = LlamaCppAdapter(base_url=api_url, api_key=api_key)
        llm = RealWorldTracingLLMProvider(raw, self.emit)
        autonomy_store = SQLiteAutonomyAdapter(str(self.workspace / "autonomy.db"))
        memory = SQLiteMemoryAdapter(str(self.workspace / "memory.db"), str(self.workspace / "memory"))
        capture_store = PlainCaptureStore(str(self.workspace / "captures"))
        policy = CapabilityPolicyService(
            store=autonomy_store,
            budget=AutonomyBudget(
                max_tool_calls_per_hour=parser.getint("autonomy", "max_tool_calls_per_hour", fallback=120),
                max_web_queries_per_day=parser.getint("autonomy", "max_web_queries_per_day", fallback=60),
            ),
        )
        tools = MCPToolAdapter()
        mcp_path = Path(parser.get("runtime", "mcp_config_path", fallback="mcp.json"))
        if not mcp_path.is_absolute():
            mcp_path = self.project_root / mcp_path
        await tools.start_servers(str(mcp_path))
        service = LLMInteractionService(
            llm_provider=llm, tool_bridge=tools,
            browser_agent_model=self.model_roles.get("browser_agent_model"),
            reporter_model=self.model_roles.get("reporter_model"),
            artifact_root=str(self.workspace / "artifacts"), capability_policy=policy,
        )
        await service.initialize_tools()
        coordinator = AutonomyCoordinatorService(
            store=autonomy_store, judgment=OpportunityJudgmentService(llm_provider=llm),
            policy=policy, mode="active", capture_store=capture_store,
            max_inbox_items_per_day=parser.getint("autonomy", "max_inbox_items_per_day", fallback=30),
        )
        self._apply_seed(memory, scenario.seed)
        started = time.monotonic()
        final_response = ""
        transcript_parts: list[str] = []
        try:
            if scenario.modality == "image_sequence":
                from application.services.passive_observer_service import PassiveObserverService
                from application.services.screenshot_queue_service import ScreenshotQueueService
                capture = _StaticScreenCapture()
                observer = PassiveObserverService(
                    memory=memory, llm_provider=llm, screen_capture=capture,
                    screenshot_root=str(self.workspace / "screens"),
                    # The production coordinator deliberately uses one vision-capable
                    # model for screen enrichment, judgment, and investigation.
                    fast_model=self.model_roles["followup_execution_model"],
                    full_model=self.model_roles["followup_execution_model"],
                    persist_observations=True, capture_store=capture_store, persist_payloads=True,
                )
                coordinator.visual_observer = observer
                coordinator.visual_model = self.model_roles["followup_execution_model"]
                queue = ScreenshotQueueService(
                    maxlen=180,
                    ssim_threshold=parser.getfloat("passive_observer", "ssim_threshold", fallback=0.92),
                    ssim_compare_count=parser.getint("passive_observer", "ssim_compare_count", fallback=4),
                )
                for index, event in enumerate(scenario.events):
                    await self._wait_for_event(started, event.offset_seconds / playback_speed)
                    capture.path = event.media_path
                    self.emit("input", "image_released", {"index": index, **asdict(event)})
                    job = queue.enqueue(event.media_path, captured_at=event.captured_at, retain=False)
                    if job is None:
                        self.emit("vision", "image_deduplicated", {"index": index})
                        continue
                    await llm.load_model(self.model_roles["followup_execution_model"])
                    observation = await observer.process_screenshot(
                        screenshot_path=event.media_path,
                        model=self.model_roles["followup_execution_model"],
                        recent_context=memory.get_recent_context(), captured_at=event.captured_at,
                        similarity_score=job.similarity_score,
                        uiat_context_override=event.screen_context or None,
                        archive_source=False,
                    )
                    if observation is None:
                        self.emit("vision", "observation_skipped", {"index": index})
                        continue
                    self.emit("vision", "visual_observation", {
                        "index": index, "observation": dict(observation.__dict__),
                    })
                    coordinator.enqueue_visual_observation(observation)
                    await llm.load_model(self.model_roles["followup_execution_model"])
                    result = await coordinator.process_next(
                        model=self.model_roles["followup_execution_model"], llm_service=service,
                        personalization_context=memory.get_recent_context(),
                        event_callback=self._interaction_event,
                    )
                    final_response = json.dumps(result, ensure_ascii=False)
                    self.emit("autonomy", "autonomy_result", result)
            else:
                for index, event in enumerate(scenario.events):
                    await self._wait_for_event(started, event.offset_seconds / playback_speed)
                    self.emit("input", "audio_released", {"index": index, **asdict(event)})
                    transcript_path = await asyncio.to_thread(self._run_audio_chain, event.media_path, scenario.seed)
                    transcript = Path(transcript_path).read_text(encoding="utf-8").strip()
                    transcript_parts.append(transcript)
                    self.emit("audio", "transcript_merged", {"index": index, "transcript_path": transcript_path,
                                                              "transcript_text": transcript})
                    coordinator.enqueue_transcript(transcript_path=transcript_path, transcript_text=transcript)
                    await llm.load_model(self.model_roles["transcript_processing_model"])
                    result = await coordinator.process_next(
                        model=self.model_roles["transcript_processing_model"], llm_service=service,
                        personalization_context=memory.get_recent_context(),
                        event_callback=self._interaction_event,
                    )
                    final_response = json.dumps(result, ensure_ascii=False)
                    self.emit("autonomy", "autonomy_result", result)
        finally:
            try:
                await llm.unload_model()
            finally:
                await tools.cleanup()
        return {"transcript_text": "\n\n".join(transcript_parts) or None,
                "final_response": final_response or None,
                "summary": {"event_count": len(scenario.events), "modality": scenario.modality}}

    async def _wait_for_event(self, started: float, due_seconds: float) -> None:
        while True:
            if self.should_cancel():
                raise RunCancelled("Run cancelled by operator")
            remaining = started + due_seconds - time.monotonic()
            if remaining <= 0:
                return
            await asyncio.sleep(min(0.25, remaining))

    def _interaction_event(self, event: dict[str, Any]) -> None:
        # Complete text/reasoning is persisted by RealWorldTracingLLMProvider;
        # token deltas are intentionally not duplicated as thousands of rows.
        if event.get("type") == "delta":
            return
        self.emit("agent", str(event.get("type") or "agent_event"), event,
                  status="failed" if event.get("ok") is False else "completed")

    def _run_audio_chain(self, media_path: str, seed: dict[str, Any]) -> str:
        from audio_agent import AudioAgent
        import queue
        voice_db = self.workspace / "voice.db"
        seed_voice = seed.get("voice_db_path")
        if seed_voice and Path(seed_voice).is_file():
            shutil.copy2(seed_voice, voice_db)
        agent = AudioAgent(
            transcription_queue=queue.Queue(), voice_db=str(voice_db),
            transcriptions_dir=str(self.workspace / "transcriptions"),
            cleaned_audio_dir=str(self.workspace / "cleaned_audio"),
            temp_audio_dir=str(self.workspace / "temp_audio"),
            asr_model=self._audio_model_path(),
            stage_callback=lambda stage, payload: self.emit("audio", stage, payload),
        )
        result = agent.run(media_path)
        if not result or not result.get("transcript_path"):
            raise RuntimeError("Audio pipeline did not produce a transcript")
        return str(result["transcript_path"])

    def _audio_model_path(self) -> str:
        parser = configparser.ConfigParser()
        parser.read(self.config_path, encoding="utf-8")
        return parser.get("audio", "hin2hinglish_model", fallback="Hin2Hinglish-ct2/")

    def _apply_seed(self, memory, seed: dict[str, Any]) -> None:
        if seed.get("user_info"):
            memory.save_user_info(str(seed["user_info"]))
        if seed.get("working_memory"):
            memory.save_working_memory(str(seed["working_memory"]))
