import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    service: str
    title: str
    inputs: Dict[str, Any]
    expected: Dict[str, Any]
    rubric_notes: str = ""
    source_path: str = ""


@dataclass(frozen=True)
class BenchmarkSuite:
    service: str
    models: List[str]
    cases: List[BenchmarkCase]
    source_path: str = ""


def load_cases(cases_root: str, *, service: Optional[str] = None) -> List[BenchmarkCase]:
    root = Path(cases_root)
    if not root.exists():
        return []
    cases: List[BenchmarkCase] = []
    for path in sorted(root.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            case = _parse_case(item, path)
            if service and case.service != service:
                continue
            cases.append(case)
    return cases


def build_inline_case(
    *,
    service: str,
    screenshot_path: Optional[str] = None,
    transcript_path: Optional[str] = None,
    title: Optional[str] = None,
) -> BenchmarkCase:
    resolved_inputs: Dict[str, Any] = {}
    if screenshot_path:
        resolved_inputs["screenshot_path"] = str(Path(screenshot_path).resolve())
    if transcript_path:
        resolved_inputs["transcript_path"] = str(Path(transcript_path).resolve())
    default_title = title or _default_title(service=service, inputs=resolved_inputs)
    return BenchmarkCase(
        case_id=f"{service}_inline",
        service=service,
        title=default_title,
        inputs=resolved_inputs,
        expected={},
        rubric_notes="",
        source_path="inline-cli",
    )


def load_suite(path: str) -> BenchmarkSuite:
    suite_path = Path(path).resolve()
    payload = json.loads(suite_path.read_text(encoding="utf-8"))
    service = str(payload["service"]).strip()
    models = [str(item).strip() for item in payload.get("models", []) if str(item).strip()]
    if not models:
        raise ValueError("Benchmark suite must define a non-empty 'models' list.")
    cases: List[BenchmarkCase] = []
    if service == "passive_observer":
        screenshots = payload.get("screenshots", []) or payload.get("screenshot_paths", [])
        for index, item in enumerate(screenshots, start=1):
            resolved = str((suite_path.parent / item).resolve()) if not Path(str(item)).is_absolute() else str(item)
            cases.append(
                BenchmarkCase(
                    case_id=f"passive_observer_{index}",
                    service=service,
                    title=f"Passive observer on {Path(resolved).name}",
                    inputs={"screenshot_path": resolved},
                    expected={},
                    rubric_notes=str(payload.get("rubric_notes") or "").strip(),
                    source_path=str(suite_path),
                )
            )
    elif service == "reflection_service":
        transcripts = payload.get("transcripts", []) or payload.get("transcript_paths", [])
        for index, item in enumerate(transcripts, start=1):
            resolved = str((suite_path.parent / item).resolve()) if not Path(str(item)).is_absolute() else str(item)
            cases.append(
                BenchmarkCase(
                    case_id=f"reflection_service_{index}",
                    service=service,
                    title=f"Reflection on {Path(resolved).name}",
                    inputs={"transcript_path": resolved},
                    expected={},
                    rubric_notes=str(payload.get("rubric_notes") or "").strip(),
                    source_path=str(suite_path),
                )
            )
    else:
        raw_cases = payload.get("cases", [])
        for item in raw_cases:
            cases.append(_parse_case(item, suite_path))
    if not cases:
        raise ValueError("Benchmark suite did not produce any cases.")
    return BenchmarkSuite(
        service=service,
        models=models,
        cases=cases,
        source_path=str(suite_path),
    )


def _parse_case(payload: Dict[str, Any], path: Path) -> BenchmarkCase:
    inputs = dict(payload.get("inputs") or {})
    if "screenshot_path" in payload and "screenshot_path" not in inputs:
        inputs["screenshot_path"] = payload["screenshot_path"]
    if "transcript_path" in payload and "transcript_path" not in inputs:
        inputs["transcript_path"] = payload["transcript_path"]
    resolved_inputs = {}
    for key, value in inputs.items():
        if key.endswith("_path") and isinstance(value, str):
            resolved_inputs[key] = str((path.parent / value).resolve()) if not Path(value).is_absolute() else value
        else:
            resolved_inputs[key] = value
    service = str(payload["service"]).strip()
    return BenchmarkCase(
        case_id=str(payload["case_id"]).strip(),
        service=service,
        title=str(payload.get("title") or _default_title(service=service, inputs=resolved_inputs) or payload["case_id"]).strip(),
        inputs=resolved_inputs,
        expected=dict(payload.get("expected") or {}),
        rubric_notes=str(payload.get("rubric_notes") or "").strip(),
        source_path=str(path),
    )


def _default_title(*, service: str, inputs: Dict[str, Any]) -> str:
    if service == "passive_observer" and inputs.get("screenshot_path"):
        return f"Passive observer on {Path(str(inputs['screenshot_path'])).name}"
    if service == "reflection_service" and inputs.get("transcript_path"):
        return f"Reflection on {Path(str(inputs['transcript_path'])).name}"
    return service.replace("_", " ").strip().title()
