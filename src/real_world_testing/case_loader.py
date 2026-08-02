from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".opus", ".flac"}


@dataclass(frozen=True)
class ScheduledMediaInput:
    offset_seconds: float
    media_path: str
    captured_at: str | None = None
    screen_context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentTaskInput:
    task_id: str
    agent_kind: str
    title: str
    instruction: str
    start_url: str = ""
    read_only: bool = True
    max_steps: int | None = None
    timeout_seconds: float | None = None
    success_criteria: str = ""
    rubric_notes: str = ""


@dataclass(frozen=True)
class RealWorldScenario:
    scenario_id: str
    title: str
    modality: str
    events: list[ScheduledMediaInput]
    tasks: list[AgentTaskInput] = field(default_factory=list)
    rubric_notes: str = ""
    seed: dict[str, Any] = field(default_factory=dict)
    source_path: str = ""


@dataclass(frozen=True)
class RealWorldSuite:
    schema_version: int
    suite_id: str
    title: str
    description: str
    scenarios: list[RealWorldScenario]
    source_path: str


def load_suites(root: str | Path) -> list[RealWorldSuite]:
    suites_root = Path(root)
    if not suites_root.exists():
        return []
    return [load_suite(path) for path in sorted(suites_root.rglob("*.json"))]


def load_suite(path: str | Path) -> RealWorldSuite:
    suite_path = Path(path).resolve()
    payload = json.loads(suite_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"{suite_path}: schema_version must be 1")
    suite_id = _identifier(payload.get("suite_id"), "suite_id", suite_path)
    scenarios = [_parse_scenario(item, suite_path) for item in payload.get("scenarios", [])]
    if not scenarios:
        raise ValueError(f"{suite_path}: at least one scenario is required")
    ids = [item.scenario_id for item in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{suite_path}: scenario_id values must be unique")
    return RealWorldSuite(
        schema_version=1,
        suite_id=suite_id,
        title=str(payload.get("title") or suite_id).strip(),
        description=str(payload.get("description") or "").strip(),
        scenarios=scenarios,
        source_path=str(suite_path),
    )


def suite_to_dict(suite: RealWorldSuite) -> dict[str, Any]:
    return asdict(suite)


def _parse_scenario(payload: dict[str, Any], suite_path: Path) -> RealWorldScenario:
    scenario_id = _identifier(payload.get("scenario_id"), "scenario_id", suite_path)
    modality = str(payload.get("modality") or "").strip()
    if modality not in {"image_sequence", "audio_sequence", "agent_task_sequence"}:
        raise ValueError(f"{suite_path}: {scenario_id} has unsupported modality {modality!r}")
    events: list[ScheduledMediaInput] = []
    tasks: list[AgentTaskInput] = []
    if modality in {"image_sequence", "audio_sequence"}:
        previous_offset = -1.0
        for index, item in enumerate(payload.get("events", []), start=1):
            offset = float(item.get("offset_seconds", 0))
            if offset < 0 or offset < previous_offset:
                raise ValueError(f"{suite_path}: {scenario_id} event offsets must be non-negative and monotonic")
            previous_offset = offset
            raw_path = str(item.get("media_path") or "").strip()
            if not raw_path:
                raise ValueError(f"{suite_path}: {scenario_id} event {index} is missing media_path")
            resolved = Path(raw_path).expanduser()
            if not resolved.is_absolute():
                resolved = (suite_path.parent / resolved).resolve()
            suffixes = IMAGE_SUFFIXES if modality == "image_sequence" else AUDIO_SUFFIXES
            if resolved.suffix.lower() not in suffixes:
                raise ValueError(f"{suite_path}: unsupported {modality} media type: {resolved.suffix}")
            events.append(
                ScheduledMediaInput(
                    offset_seconds=offset,
                    media_path=str(resolved),
                    captured_at=str(item.get("captured_at") or "").strip() or None,
                    screen_context=dict(item.get("screen_context") or {}),
                )
            )
        if not events:
            raise ValueError(f"{suite_path}: {scenario_id} requires at least one event")
    else:
        for index, item in enumerate(payload.get("tasks", []), start=1):
            task_id = _identifier(item.get("task_id") or f"task_{index}", "task_id", suite_path)
            agent_kind = str(item.get("agent_kind") or "").strip().lower()
            if agent_kind not in {"browser", "computer"}:
                raise ValueError(f"{suite_path}: {scenario_id} task {task_id} has unsupported agent_kind {agent_kind!r}")
            instruction = str(item.get("instruction") or "").strip()
            if not instruction:
                raise ValueError(f"{suite_path}: {scenario_id} task {task_id} is missing instruction")
            tasks.append(
                AgentTaskInput(
                    task_id=task_id,
                    agent_kind=agent_kind,
                    title=str(item.get("title") or task_id).strip(),
                    instruction=instruction,
                    start_url=str(item.get("start_url") or "").strip(),
                    read_only=bool(item.get("read_only", True)),
                    max_steps=int(item["max_steps"]) if item.get("max_steps") is not None else None,
                    timeout_seconds=float(item["timeout_seconds"]) if item.get("timeout_seconds") is not None else None,
                    success_criteria=str(item.get("success_criteria") or "").strip(),
                    rubric_notes=str(item.get("rubric_notes") or "").strip(),
                )
            )
        if not tasks:
            raise ValueError(f"{suite_path}: {scenario_id} requires at least one agent task")
    if modality in {"image_sequence", "audio_sequence"} and not events:
        raise ValueError(f"{suite_path}: {scenario_id} requires at least one event")
    seed = dict(payload.get("seed") or {})
    for key, value in list(seed.items()):
        if key.endswith("_path") and value:
            candidate = Path(str(value)).expanduser()
            seed[key] = str(candidate if candidate.is_absolute() else (suite_path.parent / candidate).resolve())
    return RealWorldScenario(
        scenario_id=scenario_id,
        title=str(payload.get("title") or scenario_id).strip(),
        modality=modality,
        events=events,
        tasks=tasks,
        rubric_notes=str(payload.get("rubric_notes") or "").strip(),
        seed=seed,
        source_path=str(suite_path),
    )


def _identifier(value: Any, field_name: str, path: Path) -> str:
    normalized = str(value or "").strip()
    if not normalized or not all(ch.isalnum() or ch in "-_" for ch in normalized):
        raise ValueError(f"{path}: {field_name} must contain only letters, numbers, '-' or '_'")
    return normalized
