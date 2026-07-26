from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from infrastructure.accelerator import detect_accelerator


def _chat_once(api_base_url: str, api_key: str, model: str, prompt: str) -> dict[str, object]:
    started = time.perf_counter()
    first_token_at = None
    text_parts: list[str] = []
    response = requests.post(
        f"{api_base_url.rstrip('/')}/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "stream": True,
            "messages": [{"role": "user", "content": prompt}],
        },
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
    completion_chars = len("".join(text_parts))
    generation_seconds = max(0.0, ended - (first_token_at or ended))
    return {
        "ttft_seconds": (first_token_at - started) if first_token_at else None,
        "total_seconds": ended - started,
        "generation_seconds": generation_seconds,
        "completion_chars": completion_chars,
        "chars_per_second": completion_chars / generation_seconds if generation_seconds > 0 else None,
    }


def _summarize(rows: list[dict[str, object]]) -> dict[str, float | int | None]:
    measured = [row for row in rows if not row.get("warmup")]
    summary: dict[str, float | int | None] = {"measured_runs": len(measured)}
    for field in ("ttft_seconds", "total_seconds", "chars_per_second"):
        values = [float(row[field]) for row in measured if row.get(field) is not None]
        summary[f"{field}_median"] = statistics.median(values) if values else None
        summary[f"{field}_mean"] = statistics.fmean(values) if values else None
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark a running llama.cpp router for ROCm evidence.")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--api-key", default="testkey")
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", default="rocm")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--prompt", default="Summarize the benefits and risks of ambient AI in five concise bullets.")
    parser.add_argument("--output", default=".ambient_data/benchmarks/rocm-benchmark.json")
    args = parser.parse_args()

    rows = []
    for index in range(args.warmups + args.runs):
        row = _chat_once(args.api_base_url, args.api_key, args.model, args.prompt)
        row.update({"label": args.label, "model": args.model, "index": index, "warmup": index < args.warmups})
        rows.append(row)
        print(json.dumps(row))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": args.label,
        "model": args.model,
        "api_base_url": args.api_base_url,
        "accelerator": detect_accelerator().to_dict(),
        "summary": _summarize(rows),
        "runs": rows,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"json": str(output), "csv": str(csv_path), "summary": payload["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
