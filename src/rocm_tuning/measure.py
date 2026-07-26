from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any

import requests

from infrastructure.accelerator import detect_accelerator


@dataclass(frozen=True)
class VramSnapshot:
    total_mb: int | None
    free_mb: int | None


@dataclass(frozen=True)
class ChatMeasurement:
    warmup: bool
    index: int
    ttft_seconds: float | None
    total_seconds: float
    generation_seconds: float
    completion_chars: int
    chars_per_second: float | None
    free_vram_before_mb: int | None
    free_vram_after_mb: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def vram_snapshot() -> VramSnapshot:
    try:
        import torch

        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            divisor = 1024 * 1024
            return VramSnapshot(total_mb=int(total_bytes / divisor), free_mb=int(free_bytes / divisor))
    except Exception:
        pass
    return VramSnapshot(total_mb=None, free_mb=None)


def load_model(api_base_url: str, model: str, timeout: float = 60.0) -> float:
    started = time.perf_counter()
    response = requests.post(f"{api_base_url.rstrip('/')}/models/load", json={"model": model}, timeout=timeout)
    if response.status_code not in {200, 400}:
        raise RuntimeError(f"model load failed: {response.status_code} {response.text}")
    if response.status_code == 400 and "already" not in response.text.lower():
        raise RuntimeError(f"model load failed: {response.status_code} {response.text}")
    return (time.perf_counter() - started) * 1000.0


def unload_model(api_base_url: str, model: str, timeout: float = 30.0) -> None:
    try:
        requests.post(f"{api_base_url.rstrip('/')}/models/unload", json={"model": model}, timeout=timeout)
    except requests.RequestException:
        return


def chat_once(api_base_url: str, api_key: str, model: str, prompt: str, *, index: int, warmup: bool) -> ChatMeasurement:
    before = vram_snapshot()
    started = time.perf_counter()
    first_token_at = None
    text_parts: list[str] = []
    response = requests.post(
        f"{api_base_url.rstrip('/')}/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "stream": True, "messages": [{"role": "user", "content": prompt}]},
        stream=True,
        timeout=180,
    )
    response.raise_for_status()
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
        if delta:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            text_parts.append(delta)
    ended = time.perf_counter()
    after = vram_snapshot()
    generation_seconds = max(0.0, ended - (first_token_at or ended))
    completion_chars = len("".join(text_parts))
    return ChatMeasurement(
        warmup=warmup,
        index=index,
        ttft_seconds=(first_token_at - started) if first_token_at else None,
        total_seconds=ended - started,
        generation_seconds=generation_seconds,
        completion_chars=completion_chars,
        chars_per_second=completion_chars / generation_seconds if generation_seconds > 0 else None,
        free_vram_before_mb=before.free_mb,
        free_vram_after_mb=after.free_mb,
    )


def summarize_measurements(rows: list[ChatMeasurement]) -> dict[str, Any]:
    measured = [row for row in rows if not row.warmup]
    summary: dict[str, Any] = {"measured_runs": len(measured), "accelerator": detect_accelerator().to_dict()}
    for field in ("ttft_seconds", "total_seconds", "chars_per_second"):
        values = [float(getattr(row, field)) for row in measured if getattr(row, field) is not None]
        summary[f"{field}_median"] = statistics.median(values) if values else None
        summary[f"{field}_mean"] = statistics.fmean(values) if values else None
    free_before = [row.free_vram_before_mb for row in measured if row.free_vram_before_mb is not None]
    free_after = [row.free_vram_after_mb for row in measured if row.free_vram_after_mb is not None]
    summary["free_vram_before_mb_min"] = min(free_before) if free_before else None
    summary["free_vram_after_mb_min"] = min(free_after) if free_after else None
    if free_before and free_after:
        summary["vram_delta_mb_max"] = max(free_before) - min(free_after)
    else:
        summary["vram_delta_mb_max"] = None
    return summary
