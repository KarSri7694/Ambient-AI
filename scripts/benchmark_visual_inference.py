"""Benchmark passive visual inference against a running llama-server.

Examples:
    python scripts/benchmark_visual_inference.py
    python scripts/benchmark_visual_inference.py --counts 1 2 3 6 --repeat 2
    python scripts/benchmark_visual_inference.py --capture-root D:/USERS_DATA/captures

The script is intentionally standalone. It does not start app.py, mutate the
capture directory, or change runtime configuration. It reports wall-clock
latency, llama.cpp usage timing when returned by the server, and JSON validity.
"""

from __future__ import annotations

import argparse
import asyncio
import configparser
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from application.services.passive_observer_service import PassiveObserverService
from infrastructure.adapter.llamaCppAdapter import LlamaCppAdapter


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LOG = logging.getLogger("visual_inference_benchmark")


@dataclass(frozen=True)
class Capture:
    path: Path
    captured_at: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.ini")
    parser.add_argument("--capture-root", type=Path, default=None)
    parser.add_argument("--endpoint", default=None, help="llama-server base URL, e.g. http://127.0.0.1:8080")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--images", type=int, default=6, help="Images per benchmark case (default: 6)")
    parser.add_argument("--counts", type=int, nargs="+", default=None, help="Override image counts; default is --images")
    parser.add_argument("--slots", type=int, default=3, help="Maximum concurrent llama-server requests (default: 3)")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--load-model", action="store_true", help="Ask the endpoint to load the model; default assumes it is already loaded")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path")
    return parser.parse_args()


def read_config(path: Path) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read(path, encoding="utf-8")
    return config


def config_value(config: configparser.ConfigParser, section: str, option: str, fallback: str) -> str:
    return config.get(section, option, fallback=fallback).strip()


def capture_timestamp(path: Path) -> str:
    """Prefer the capture sidecar timestamp, then filename, then mtime."""
    candidates = [path.with_suffix(path.suffix + ".json"), path.parent / "metadata" / f"{path.stem}.json"]
    for sidecar in candidates:
        if sidecar.exists():
            try:
                data = json.loads(sidecar.read_text(encoding="utf-8"))
                value = data.get("captured_at") or data.get("created_at")
                if value:
                    return str(value)
            except (OSError, json.JSONDecodeError):
                pass
    # Common capture names include an ISO-like timestamp or epoch milliseconds.
    match = re.search(r"(20\d{2}[-_\dT:.Z]+)", path.stem)
    if match:
        return match.group(1).replace("_", "T")
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def find_captures(root: Path, count: int) -> list[Capture]:
    files = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.stat().st_mtime,
    )
    if len(files) < count:
        raise FileNotFoundError(f"Found {len(files)} images under {root}, but {count} are required")
    selected = files[-count:]
    return [Capture(path, capture_timestamp(path)) for path in selected]


def payload(captures: Iterable[Capture]) -> dict[str, Any]:
    return {
        "frames": [
            {"frame_index": index, "screenshot_captured_at": item.captured_at, "similarity_score": None}
            for index, item in enumerate(captures)
        ],
        "recent_context": "Standalone visual inference benchmark.",
        "temporal_work_context": "Compare ordered screen states using their capture timestamps.",
        "previous_observation": None,
    }


def batch_system_prompt(image_count: int) -> str:
    now = datetime.now()
    preamble = (
        f"Current day of week: {now.strftime('%A')}\n"
        f"Current date: {now.strftime('%Y-%m-%d')}\n"
        f"Current time: {now.strftime('%H:%M:%S')}\n\n"
    )
    contract = f"""
IMPORTANT OUTPUT CONTRACT:
- You received exactly {image_count} images and exactly {image_count} frame records.
- Return exactly one observation JSON object for each image: exactly {image_count} items in
  the `observations` array.
- Each item must contain the matching zero-based `frame_index` from 0 through {image_count - 1}.
- Never merge, summarize, or omit images. Do not return a single overall observation.
- Process the images independently, then return the observations in frame_index order.
"""
    return preamble + PassiveObserverService.BATCH_ROUTER_PROMPT + contract


def parse_json(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None


async def consume(completion: Any) -> tuple[str, dict[str, Any]]:
    text_parts: list[str] = []
    usage: dict[str, Any] = {}
    if hasattr(completion, "__aiter__"):
        async for chunk in completion:
            for choice in getattr(chunk, "choices", []) or []:
                delta = getattr(choice, "delta", None)
                text = getattr(delta, "content", None) if delta is not None else None
                if text:
                    text_parts.append(str(text))
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = {key: getattr(chunk_usage, key) for key in ("prompt_tokens", "completion_tokens", "total_tokens") if getattr(chunk_usage, key, None) is not None}
    else:
        choice = (getattr(completion, "choices", []) or [None])[0]
        message = getattr(choice, "message", None)
        text_parts.append(str(getattr(message, "content", "") or ""))
        raw_usage = getattr(completion, "usage", None)
        if raw_usage is not None:
            usage = {key: getattr(raw_usage, key) for key in ("prompt_tokens", "completion_tokens", "total_tokens") if getattr(raw_usage, key, None) is not None}
    return "".join(text_parts), usage


async def request(
    adapter: LlamaCppAdapter,
    captures: list[Capture],
    model: str,
    timeout: float,
    max_output_tokens: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    image_paths = [str(item.path) for item in captures]
    messages = [
        {"role": "system", "content": batch_system_prompt(len(captures))},
        {"role": "user", "content": json.dumps(payload(captures), ensure_ascii=False)},
    ]
    completion = await asyncio.wait_for(
        adapter.chat_completion_stream(
            model=model,
            messages=messages,
            tools=None,
            image=image_paths,
            temperature=0.1,
            max_tokens=max(1, int(max_output_tokens)),
            response_format=PassiveObserverService.BATCH_RESPONSE_SCHEMA,
            chat_template_kwargs={"enable_thinking": False},
            request_timeout_seconds=timeout,
        ),
        timeout=timeout,
    )
    text, usage = await asyncio.wait_for(consume(completion), timeout=timeout)
    elapsed_ms = (time.perf_counter() - started) * 1000
    parsed = parse_json(text)
    observations = parsed.get("observations", []) if parsed else []
    prompt_processing_ms = usage.get("prompt_processing_ms") or usage.get("prompt_eval_duration_ms")
    return {
        "elapsed_ms": round(elapsed_ms, 2),
        "prompt_processing_ms": prompt_processing_ms,
        "usage": usage,
        "json_valid": parsed is not None,
        "observation_count": len(observations) if isinstance(observations, list) else 0,
        "raw_output": parsed,
    }


async def run_mode(
    adapter: LlamaCppAdapter,
    captures: list[Capture],
    mode: str,
    slots: int,
    model: str,
    timeout: float,
    max_output_tokens: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    semaphore = asyncio.Semaphore(slots)

    async def limited(group: list[Capture]) -> dict[str, Any]:
        async with semaphore:
            return await request(adapter, group, model, timeout, max_output_tokens)

    if mode == "parallel":
        results = await asyncio.gather(*(limited([item]) for item in captures))
    elif mode == "batch":
        results = [await limited(captures)]
    else:
        groups = [captures[index:index + max(1, (len(captures) + slots - 1) // slots)] for index in range(0, len(captures), max(1, (len(captures) + slots - 1) // slots))]
        results = await asyncio.gather(*(limited(group) for group in groups))
    total_ms = (time.perf_counter() - started) * 1000
    return {"mode": mode, "image_count": len(captures), "request_count": len(results), "total_ms": round(total_ms, 2), "sum_request_ms": round(sum(item["elapsed_ms"] for item in results), 2), "results": results}


async def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = read_config(args.config)
    endpoint = args.endpoint or config_value(config, "vision_runtime", "api_base_url", "") or config_value(config, "runtime", "api_base_url", "http://127.0.0.1:8080")
    api_key = args.api_key or config_value(config, "vision_runtime", "api_key", "") or config_value(config, "runtime", "api_key", "testkey")
    model = args.model or config_value(config, "models", "passive_observer_model", config_value(config, "runtime", "default_model", "Qwen3.5-4B"))
    root = args.capture_root or Path(config_value(config, "privacy", "capture_root", "D:/USERS_DATA/captures"))
    counts = args.counts or [args.images]
    adapter = LlamaCppAdapter(endpoint, api_key=api_key, default_max_tokens=args.max_output_tokens)
    LOG.info("Using endpoint=%s model=%s capture_root=%s slots=%s", endpoint, model, root, args.slots)
    if args.load_model:
        await adapter.load_model(model)
    else:
        # Standalone llama-server commonly exposes only /v1/chat/completions.
        # Mark the configured model ready so the adapter does not require the
        # optional project router management endpoints.
        adapter.currently_loaded_model = model
        adapter._ready_model = model
    report: dict[str, Any] = {"endpoint": endpoint, "model": model, "capture_root": str(root), "slots": args.slots, "generated_at": datetime.now(timezone.utc).isoformat(), "cases": []}
    for count in counts:
        captures = find_captures(root, count)
        case = {"image_count": count, "captures": [{"path": str(item.path), "captured_at": item.captured_at} for item in captures], "runs": []}
        for repeat in range(args.repeat):
            LOG.info("Benchmarking %s images, repeat %s/%s", count, repeat + 1, args.repeat)
            for mode in ("parallel", "batch", "combined"):
                result = await run_mode(
                    adapter,
                    captures,
                    mode,
                    max(1, args.slots),
                    model,
                    args.timeout,
                    args.max_output_tokens,
                )
                result["repeat"] = repeat + 1
                case["runs"].append(result)
                LOG.info("%s: total=%.0fms request_sum=%.0fms valid=%s", mode, result["total_ms"], result["sum_request_ms"], all(item["json_valid"] for item in result["results"]))
        report["cases"].append(case)
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        LOG.info("Wrote report to %s", args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
