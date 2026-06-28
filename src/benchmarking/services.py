from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional

from application.services.passive_observer_service import PassiveObserverService
from application.services.reflection_service import ReflectionService
from benchmarking.case_loader import BenchmarkCase
from benchmarking.metrics import strip_json_text
from infrastructure.adapter.SQLiteMemoryAdapter import SQLiteMemoryAdapter


class _StaticScreenCapture:
    def __init__(self, screenshot_path: str):
        self.screenshot_path = screenshot_path

    def capture_screenshot(self, output_path: Optional[str] = None) -> str:
        return self.screenshot_path


class _InMemoryTaskQueue:
    def __init__(self, pending: Optional[List[Dict[str, Any]]] = None):
        self._pending = []
        self._added = []
        for item in pending or []:
            self._pending.append(
                type(
                    "PendingTask",
                    (),
                    {
                        "description": str(item.get("description", "")),
                        "priority": str(item.get("priority", "medium")),
                        "metadata_json": json.dumps(item.get("metadata", {}), ensure_ascii=False),
                    },
                )()
            )

    def get_pending_tasks(self):
        return list(self._pending)

    def add_task(self, description: str, priority: str = "medium", metadata=None) -> str:
        self._added.append(
            {
                "description": description,
                "priority": priority,
                "metadata": metadata or {},
            }
        )
        return "queued"

    def mark_task_complete(self, task_id: int, status: str = "completed") -> None:
        return None

    @property
    def added(self) -> List[Dict[str, Any]]:
        return list(self._added)


@dataclass(frozen=True)
class BenchmarkExecution:
    response_text: str
    structured_output_json: str
    screenshot_path: Optional[str] = None
    transcript_path: Optional[str] = None
    metadata: Dict[str, Any] | None = None


def available_services() -> List[str]:
    return ["passive_observer", "reflection_service"]


async def execute_service_case(case: BenchmarkCase, llm_provider, model_name: str) -> BenchmarkExecution:
    if case.service == "passive_observer":
        return await _run_passive_observer_case(case, llm_provider, model_name)
    if case.service == "reflection_service":
        return await _run_reflection_case(case, llm_provider, model_name)
    raise ValueError(f"Unsupported benchmark service: {case.service}")


async def _run_passive_observer_case(case: BenchmarkCase, llm_provider, model_name: str) -> BenchmarkExecution:
    screenshot_path = str(case.inputs["screenshot_path"])
    recent_context = str(case.inputs.get("recent_context", "") or "")
    captured_at = case.inputs.get("captured_at")
    with TemporaryDirectory() as tmpdir:
        memory = SQLiteMemoryAdapter(
            db_path=str(Path(tmpdir) / "memory.db"),
            memory_root=str(Path(tmpdir) / "memory"),
        )
        if recent_context:
            memory.save_recent_context(recent_context)
        service = PassiveObserverService(
            memory=memory,
            llm_provider=llm_provider,
            screen_capture=_StaticScreenCapture(screenshot_path),
            screenshot_root=str(Path(tmpdir) / "screens"),
        )
        observation = await service.process_screenshot(
            screenshot_path=screenshot_path,
            model=model_name,
            recent_context=recent_context,
            captured_at=captured_at,
        )
        if observation is None:
            raise RuntimeError("Passive observer returned no observation.")
        payload = json.loads(observation.raw_payload_json or "{}")
        return BenchmarkExecution(
            response_text=payload.get("summary") or observation.summary,
            structured_output_json=json.dumps(payload, ensure_ascii=False, indent=2),
            screenshot_path=screenshot_path,
            metadata={
                "observation_id": observation.observation_id,
                "app_name": observation.app_name,
                "page_hint": observation.page_hint,
            },
        )


async def _run_reflection_case(case: BenchmarkCase, llm_provider, model_name: str) -> BenchmarkExecution:
    with TemporaryDirectory() as tmpdir:
        history_path = Path(tmpdir) / "reflection" / "history.json"
        memory = SQLiteMemoryAdapter(
            db_path=str(Path(tmpdir) / "memory.db"),
            memory_root=str(Path(tmpdir) / "memory"),
        )
        user_info = str(case.inputs.get("user_info", "") or "")
        working_memory = str(case.inputs.get("working_memory", "") or "")
        transcript_path = case.inputs.get("transcript_path")
        transcript_text = ""
        if transcript_path:
            transcript_text = Path(transcript_path).read_text(encoding="utf-8")
        if transcript_text and not working_memory:
            working_memory = transcript_text
        if user_info:
            memory.save_user_info(user_info)
        if working_memory:
            memory.save_working_memory(working_memory)
        task_queue = _InMemoryTaskQueue(case.inputs.get("pending_tasks"))
        if case.inputs.get("history_payload"):
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history_path.write_text(
                json.dumps(case.inputs["history_payload"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        service = ReflectionService(
            memory=memory,
            task_queue=task_queue,
            llm_provider=llm_provider,
            history_path=str(history_path),
            cadence_mode="every_idle_cycle",
            max_generated_tasks=int(case.inputs.get("max_generated_tasks", 8) or 8),
        )
        result = await service.run(model=model_name)
        return BenchmarkExecution(
            response_text=strip_json_text(json.dumps(result, ensure_ascii=False)),
            structured_output_json=json.dumps(result, ensure_ascii=False, indent=2),
            transcript_path=str(transcript_path) if transcript_path else None,
            metadata={"queued_tasks": task_queue.added},
        )


def score_case(case: BenchmarkCase, execution: BenchmarkExecution) -> tuple[float, Dict[str, Any]]:
    expected = dict(case.expected or {})
    checks: List[Dict[str, Any]] = []
    passed = 0
    total = 0
    normalized_output = (execution.response_text or "").lower()
    structured = {}
    if execution.structured_output_json:
        try:
            structured = json.loads(execution.structured_output_json)
        except json.JSONDecodeError:
            structured = {}

    for phrase in expected.get("contains", []) or []:
        total += 1
        ok = str(phrase).lower() in normalized_output
        passed += int(ok)
        checks.append({"kind": "contains", "expected": phrase, "passed": ok})

    for field_name in expected.get("json_fields", []) or []:
        total += 1
        ok = field_name in structured
        passed += int(ok)
        checks.append({"kind": "json_field", "expected": field_name, "passed": ok})

    for key, expected_value in (expected.get("equals") or {}).items():
        total += 1
        ok = structured.get(key) == expected_value
        passed += int(ok)
        checks.append({"kind": "equals", "expected": {key: expected_value}, "passed": ok})

    score = 1.0 if total == 0 else round(passed / total, 4)
    return score, {"passed_checks": passed, "total_checks": total, "checks": checks}


def run_service_case(case: BenchmarkCase, llm_provider, model_name: str) -> BenchmarkExecution:
    return asyncio.run(execute_service_case(case, llm_provider, model_name))
