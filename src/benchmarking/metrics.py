import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class BenchmarkCallRecord:
    started_at: str
    completed_at: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prefill_seconds: float
    generation_seconds: float
    prefill_tokens_per_second: float
    generation_tokens_per_second: float
    token_count_method: str
    response_text: str
    error_text: Optional[str] = None


@dataclass(frozen=True)
class AggregatedMetrics:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prefill_seconds: float
    generation_seconds: float
    prefill_tokens_per_second: float
    generation_tokens_per_second: float
    token_count_method: str
    call_count: int
    calls_json: str = field(default="")


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_message_tokens(messages: List[Dict[str, Any]], image: str = "") -> int:
    normalized: List[Dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            flattened = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    flattened.append(item.get("text", ""))
                elif isinstance(item, dict) and item.get("type") == "image_url":
                    flattened.append("[image]")
                else:
                    flattened.append(str(item))
            content = "\n".join(flattened)
        normalized.append({"role": message.get("role", ""), "content": content})
    serialized = json.dumps(normalized, ensure_ascii=False)
    if image:
        serialized += "\n[image-attached]"
    return estimate_text_tokens(serialized)


def safe_rate(tokens: int, seconds: float) -> float:
    if tokens <= 0 or seconds <= 0:
        return 0.0
    return round(tokens / seconds, 3)


def aggregate_metrics(calls: List[BenchmarkCallRecord]) -> AggregatedMetrics:
    prompt_tokens = sum(item.prompt_tokens for item in calls)
    completion_tokens = sum(item.completion_tokens for item in calls)
    total_tokens = sum(item.total_tokens for item in calls)
    prefill_seconds = round(sum(item.prefill_seconds for item in calls), 6)
    generation_seconds = round(sum(item.generation_seconds for item in calls), 6)
    methods = {item.token_count_method for item in calls if item.token_count_method}
    method = methods.pop() if len(methods) == 1 else ("mixed" if methods else "estimated_chars_div4")
    return AggregatedMetrics(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        prefill_seconds=prefill_seconds,
        generation_seconds=generation_seconds,
        prefill_tokens_per_second=safe_rate(prompt_tokens, prefill_seconds),
        generation_tokens_per_second=safe_rate(completion_tokens, generation_seconds),
        token_count_method=method,
        call_count=len(calls),
        calls_json=json.dumps([item.__dict__ for item in calls], ensure_ascii=False, indent=2),
    )


def now_iso() -> str:
    return datetime.now().isoformat()


def strip_json_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).strip()


def elapsed_seconds(started: float, ended: float) -> float:
    return round(max(0.0, ended - started), 6)
