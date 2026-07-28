from __future__ import annotations

import argparse
import csv
import json
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


DEFAULT_PROMPT = """
You are Ambient AI running a long real-world reasoning test.

Analyze the following scenario in depth: a user has a busy workday with overlapping
meetings, a pending document review, an urgent email thread, a calendar conflict,
and several privacy-sensitive browser tabs open. Produce a detailed operational
plan for how a local ambient agent should observe context, decide whether to act,
request permission, call tools, summarize outcomes, and avoid unsafe behavior.

Include:
1. a detailed timeline of decisions,
2. the exact tool calls the agent should consider,
3. what information should remain private,
4. how memory should be updated,
5. how the agent should recover from failed tools,
6. how to explain the final result to the user.

Write a long, structured answer with concrete examples and no filler.
""".strip()


TIMING_PATTERNS = {
    "prompt_eval": re.compile(
        r"prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*tokens\s*\(([0-9.]+)\s*tokens per second\)",
        re.IGNORECASE,
    ),
    "eval": re.compile(
        r"\beval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*runs?\s*\(([0-9.]+)\s*tokens per second\)",
        re.IGNORECASE,
    ),
    "total": re.compile(r"total time\s*=\s*([0-9.]+)\s*ms", re.IGNORECASE),
}


@dataclass(frozen=True)
class Variant:
    name: str
    args: list[str]


def default_variants() -> list[Variant]:
    return [
        Variant("baseline-fa-q8-b4096-ub1024", ["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-b", "4096", "-ub", "1024"]),
        Variant("kv-q4-fa-b4096-ub1024", ["-fa", "on", "-ctk", "q4_0", "-ctv", "q4_0", "-b", "4096", "-ub", "1024"]),
        Variant("big-batch-fa-q8-b8192-ub2048", ["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-b", "8192", "-ub", "2048"]),
        Variant("mtp-fa-q8-b4096-ub1024", [
            "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-b", "4096", "-ub", "1024",
            "--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "--spec-draft-n-min", "1",
        ]),
        Variant("mtp-kv-q4-b4096-ub1024", [
            "-fa", "on", "-ctk", "q4_0", "-ctv", "q4_0", "-b", "4096", "-ub", "1024",
            "--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "--spec-draft-n-min", "1",
        ]),
    ]


def parse_variant(raw: str) -> Variant:
    if "::" not in raw:
        raise argparse.ArgumentTypeError("variant must be NAME::ARG ARG ARG")
    name, arg_text = raw.split("::", 1)
    args = [part for part in arg_text.split(" ") if part]
    return Variant(name=name.strip(), args=args)


def wait_for_server(base_url: str, process: subprocess.Popen, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited early with code {process.returncode}")
        try:
            response = requests.get(f"{base_url}/v1/models", timeout=2)
            if response.status_code == 200:
                return
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"llama-server did not become ready: {last_error}")


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if sys.platform.startswith("win"):
        process.terminate()
    else:
        process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def run_chat(base_url: str, api_key: str, model: str, prompt: str, max_tokens: int, timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    first_token_at = None
    completion = []
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        stream=True,
        timeout=timeout,
    )
    response.raise_for_status()
    usage = None
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        if chunk.get("usage"):
            usage = chunk["usage"]
        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
        if delta:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            completion.append(delta)
    ended = time.perf_counter()
    output = "".join(completion)
    generation_seconds = max(0.0, ended - (first_token_at or ended))
    return {
        "ttft_seconds": (first_token_at - started) if first_token_at else None,
        "total_seconds": ended - started,
        "generation_seconds": generation_seconds,
        "completion_chars": len(output),
        "completion_estimated_tokens": max(1, len(output) // 4) if output else 0,
        "client_chars_per_second": len(output) / generation_seconds if generation_seconds > 0 else None,
        "usage": usage,
    }


def parse_server_timings(log_text: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    prompt_matches = list(TIMING_PATTERNS["prompt_eval"].finditer(log_text))
    eval_matches = list(TIMING_PATTERNS["eval"].finditer(log_text))
    total_matches = list(TIMING_PATTERNS["total"].finditer(log_text))
    if prompt_matches:
        match = prompt_matches[-1]
        payload.update({
            "prompt_eval_ms": float(match.group(1)),
            "prompt_eval_tokens": int(match.group(2)),
            "prompt_eval_tokens_per_second": float(match.group(3)),
        })
    if eval_matches:
        match = eval_matches[-1]
        payload.update({
            "server_eval_ms": float(match.group(1)),
            "server_eval_tokens": int(match.group(2)),
            "server_eval_tokens_per_second": float(match.group(3)),
        })
    if total_matches:
        payload["server_total_ms"] = float(total_matches[-1].group(1))
    return payload


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"runs": len(rows)}
    fields = [
        "ttft_seconds",
        "total_seconds",
        "client_chars_per_second",
        "prompt_eval_tokens_per_second",
        "server_eval_tokens_per_second",
    ]
    for field in fields:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        summary[f"{field}_median"] = statistics.median(values) if values else None
        summary[f"{field}_mean"] = statistics.fmean(values) if values else None
    return summary


def benchmark_variant(args: argparse.Namespace, variant: Variant, prompt: str) -> dict[str, Any]:
    base_url = f"http://{args.host}:{args.port}"
    with tempfile.TemporaryDirectory(prefix=f"llama-bench-{variant.name}-") as tmp:
        log_path = Path(tmp) / "llama-server.log"
        with log_path.open("w", encoding="utf-8") as log:
            command = [
                args.llama_server,
                "--host", args.host,
                "--port", str(args.port),
                "--api-key", args.api_key,
                "--models-preset", args.models_preset,
                "--model", args.model,
                "-c", str(args.context),
                "--perf",
                *variant.args,
            ]
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            try:
                wait_for_server(base_url, process, args.startup_timeout)
                all_rows: list[dict[str, Any]] = []
                for index in range(args.warmups + args.runs):
                    before_size = log_path.stat().st_size
                    row = run_chat(base_url, args.api_key, args.model, prompt, args.max_tokens, args.request_timeout)
                    time.sleep(args.log_settle_seconds)
                    with log_path.open("r", encoding="utf-8", errors="replace") as reader:
                        reader.seek(before_size)
                        timings = parse_server_timings(reader.read())
                    row.update(timings)
                    row.update({"index": index, "warmup": index < args.warmups})
                    all_rows.append(row)
                measured = [row for row in all_rows if not row["warmup"]]
                return {
                    "variant": variant.name,
                    "status": "completed",
                    "server_command": command,
                    "server_args": variant.args,
                    "model": args.model,
                    "models_preset": args.models_preset,
                    "context": args.context,
                    "prompt_chars": len(prompt),
                    "max_tokens": args.max_tokens,
                    "summary": summarize(measured),
                    "runs": all_rows,
                }
            except Exception as exc:
                return {
                    "variant": variant.name,
                    "status": "failed",
                    "server_args": variant.args,
                    "model": args.model,
                    "models_preset": args.models_preset,
                    "context": args.context,
                    "error": str(exc),
                    "log_tail": log_path.read_text(encoding="utf-8", errors="replace")[-4000:] if log_path.exists() else "",
                }
            finally:
                stop_process(process)


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = output_dir / f"llama-server-matrix-{stamp}.json"
    csv_path = output_dir / f"llama-server-matrix-{stamp}.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = []
    for result in payload["results"]:
        summary = result.get("summary") or {}
        rows.append({
            "variant": result["variant"],
            "status": result["status"],
            "context": result.get("context"),
            "ttft_seconds_median": summary.get("ttft_seconds_median"),
            "total_seconds_median": summary.get("total_seconds_median"),
            "prompt_eval_tokens_per_second_median": summary.get("prompt_eval_tokens_per_second_median"),
            "server_eval_tokens_per_second_median": summary.get("server_eval_tokens_per_second_median"),
            "client_chars_per_second_median": summary.get("client_chars_per_second_median"),
            "server_args": " ".join(result.get("server_args") or []),
            "error": result.get("error"),
        })
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["variant"])
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Start llama-server with multiple configs and benchmark a long prompt.")
    parser.add_argument("--llama-server", required=True, help="Path to llama-server.")
    parser.add_argument("--models-preset", required=True, help="models_preset.ini containing the model details.")
    parser.add_argument("--model", required=True, help="Preset/model id to pass to llama-server and chat completions.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--api-key", default="testkey")
    parser.add_argument("--context", type=int, default=131072)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--variant", action="append", type=parse_variant, help="Custom variant as NAME::ARG ARG ARG. Can be repeated.")
    parser.add_argument("--output-dir", default=".ambient_data/benchmarks")
    parser.add_argument("--startup-timeout", type=float, default=90.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--log-settle-seconds", type=float, default=0.2)
    args = parser.parse_args()

    prompt = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else args.prompt
    variants = args.variant or default_variants()
    results = []
    for variant in variants:
        print(f"Running {variant.name}: {' '.join(variant.args)}", flush=True)
        result = benchmark_variant(args, variant, prompt)
        results.append(result)
        print(json.dumps({"variant": result["variant"], "status": result["status"], "summary": result.get("summary"), "error": result.get("error")}, indent=2), flush=True)
    payload = {
        "created_at": datetime.now().isoformat(),
        "llama_server": args.llama_server,
        "models_preset": args.models_preset,
        "model": args.model,
        "prompt_chars": len(prompt),
        "results": results,
    }
    json_path, csv_path = write_outputs(Path(args.output_dir), payload)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path)}, indent=2))
    return 0 if any(result["status"] == "completed" for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
