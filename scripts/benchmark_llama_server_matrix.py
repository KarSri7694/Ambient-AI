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


DEFAULT_PROMPTS = [
    """
    Explain the history, engineering tradeoffs, and practical applications of suspension bridges.
    Include load paths, materials, failure modes, and how modern monitoring systems improve safety.
    """,
    """
    Now design a fictional suspension bridge for a windy coastal city. Give constraints,
    dimensions, materials, inspection routines, and the reasoning behind each decision.
    """,
    """
    A city council says the bridge must also support emergency evacuation, cycling lanes,
    maintenance robots, and severe corrosion risk. Revise the design while preserving
    the earlier assumptions where possible.
    """,
    """
    Create a risk register for the revised bridge. Include likelihood, impact, detection
    strategy, mitigation, and the owner responsible for each risk.
    """,
    """
    Summarize the entire bridge proposal as an executive briefing, explicitly referencing
    earlier design decisions and explaining what changed across the conversation.
    """,
]


TIMING_PATTERNS = {
    "prompt_eval": re.compile(
        r"prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*tokens\s*"
        r"\(\s*([0-9.]+)\s*ms per token,\s*([0-9.]+)\s*tokens per second\s*\)",
        re.IGNORECASE,
    ),
    "eval": re.compile(
        r"(?<!prompt )\beval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*tokens\s*"
        r"\(\s*([0-9.]+)\s*ms per token,\s*([0-9.]+)\s*tokens per second\s*\)",
        re.IGNORECASE,
    ),
    "total": re.compile(r"total time\s*=\s*([0-9.]+)\s*ms", re.IGNORECASE),
    "draft": re.compile(
        r"draft acceptance\s*=\s*([0-9.]+)\s*"
        r"\(\s*([0-9]+)\s*accepted\s*/\s*([0-9]+)\s*generated\s*\),\s*mean len\s*=\s*([0-9.]+)",
        re.IGNORECASE,
    ),
}


@dataclass(frozen=True)
class Variant:
    name: str
    args: list[str]

    @property
    def uses_mtp(self) -> bool:
        return "--spec-type" in self.args and "draft-mtp" in self.args


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


def run_chat_turn(base_url: str, api_key: str, model: str, messages: list[dict[str, str]], max_tokens: int | None, timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=body,
        timeout=timeout,
    )
    response.raise_for_status()
    ended = time.perf_counter()
    payload = response.json()
    output = payload.get("choices", [{}])[0].get("message", {}).get("content") or ""
    timings = {
        key: payload.get(key)
        for key in (
            "tokens_predicted",
            "tokens_evaluated",
            "generation_settings",
            "prompt_ms",
            "prompt_n",
            "prompt_per_second",
            "predicted_ms",
            "predicted_n",
            "predicted_per_second",
            "timings",
        )
        if key in payload
    }
    return {
        "total_seconds": ended - started,
        "completion_seconds": ended - started,
        "completion_chars": len(output),
        "completion_estimated_tokens": max(1, len(output) // 4) if output else 0,
        "client_chars_per_second": len(output) / (ended - started) if ended > started else None,
        "usage": payload.get("usage"),
        "server_timings": timings,
        "response_preview": output[:500],
        "response_text": output,
    }


def model_for_variant(args: argparse.Namespace, variant: Variant) -> str:
    if variant.uses_mtp and args.mtp_model:
        return args.mtp_model
    return args.model


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
            "prompt_eval_ms_per_token": float(match.group(3)),
            "prompt_eval_tokens_per_second": float(match.group(4)),
        })
    if eval_matches:
        match = eval_matches[-1]
        payload.update({
            "server_eval_ms": float(match.group(1)),
            "server_eval_tokens": int(match.group(2)),
            "server_eval_ms_per_token": float(match.group(3)),
            "server_eval_tokens_per_second": float(match.group(4)),
        })
    if total_matches:
        payload["server_total_ms"] = float(total_matches[-1].group(1))
    draft_matches = list(TIMING_PATTERNS["draft"].finditer(log_text))
    if draft_matches:
        match = draft_matches[-1]
        payload.update({
            "draft_acceptance": float(match.group(1)),
            "draft_tokens_accepted": int(match.group(2)),
            "draft_tokens_generated": int(match.group(3)),
            "draft_mean_length": float(match.group(4)),
        })
    return payload


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"runs": len(rows)}
    fields = [
        "total_seconds",
        "client_chars_per_second",
        "prompt_eval_tokens_per_second",
        "server_eval_tokens_per_second",
        "response_prompt_per_second",
        "response_predicted_per_second",
    ]
    for field in fields:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        summary[f"{field}_median"] = statistics.median(values) if values else None
        summary[f"{field}_mean"] = statistics.fmean(values) if values else None
    return summary


def _response_speed_fields(row: dict[str, Any]) -> dict[str, Any]:
    timings = row.get("server_timings") or {}
    nested = timings.get("timings") if isinstance(timings.get("timings"), dict) else {}
    prompt_speed = timings.get("prompt_per_second") or nested.get("prompt_per_second")
    predicted_speed = timings.get("predicted_per_second") or nested.get("predicted_per_second")
    prompt_ms = timings.get("prompt_ms") or nested.get("prompt_ms")
    predicted_ms = timings.get("predicted_ms") or nested.get("predicted_ms")
    return {
        "response_prompt_per_second": prompt_speed,
        "response_predicted_per_second": predicted_speed,
        "response_prompt_ms": prompt_ms,
        "response_predicted_ms": predicted_ms,
    }


def benchmark_variant(args: argparse.Namespace, variant: Variant, prompts: list[str]) -> dict[str, Any]:
    base_url = f"http://{args.host}:{args.port}"
    selected_model = model_for_variant(args, variant)
    with tempfile.TemporaryDirectory(prefix=f"llama-bench-{variant.name}-") as tmp:
        log_path = Path(tmp) / "llama-server.log"
        with log_path.open("w", encoding="utf-8") as log:
            command = [
                args.llama_server,
                "--host", args.host,
                "--port", str(args.port),
                "--api-key", args.api_key,
                "--models-preset", args.models_preset,
                "--model", selected_model,
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
                    messages: list[dict[str, str]] = []
                    conversation_rows: list[dict[str, Any]] = []
                    for turn_index, prompt in enumerate(prompts):
                        messages.append({"role": "user", "content": prompt})
                        before_size = log_path.stat().st_size
                        row = run_chat_turn(base_url, args.api_key, selected_model, messages, args.max_tokens, args.request_timeout)
                        messages.append({"role": "assistant", "content": row["response_text"]})
                        time.sleep(args.log_settle_seconds)
                        with log_path.open("r", encoding="utf-8", errors="replace") as reader:
                            reader.seek(before_size)
                            timings = parse_server_timings(reader.read())
                        row.update(timings)
                        row.update(_response_speed_fields(row))
                        row.update({
                            "index": index,
                            "turn_index": turn_index,
                            "warmup": index < args.warmups,
                            "prompt_chars": len(prompt),
                            "context_messages_after_turn": len(messages),
                        })
                        if not args.keep_full_responses:
                            row.pop("response_text", None)
                        conversation_rows.append(row)
                    all_rows.extend(conversation_rows)
                measured = [row for row in all_rows if not row["warmup"]]
                return {
                    "variant": variant.name,
                    "status": "completed",
                    "server_command": command,
                    "server_args": variant.args,
                    "model": selected_model,
                    "base_model": args.model,
                    "mtp_model": args.mtp_model,
                    "uses_mtp": variant.uses_mtp,
                    "models_preset": args.models_preset,
                    "context": "from-preset",
                    "prompt_count": len(prompts),
                    "prompt_chars": sum(len(item) for item in prompts),
                    "max_tokens": args.max_tokens,
                    "summary": summarize(measured),
                    "runs": all_rows,
                }
            except Exception as exc:
                return {
                    "variant": variant.name,
                    "status": "failed",
                    "server_args": variant.args,
                    "model": selected_model,
                    "base_model": args.model,
                    "mtp_model": args.mtp_model,
                    "uses_mtp": variant.uses_mtp,
                    "models_preset": args.models_preset,
                    "context": "from-preset",
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
                    "prompt_eval_tokens_per_second_median": summary.get("prompt_eval_tokens_per_second_median") or summary.get("response_prompt_per_second_median"),
                    "server_eval_tokens_per_second_median": summary.get("server_eval_tokens_per_second_median"),
                    "response_predicted_per_second_median": summary.get("response_predicted_per_second_median"),
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
    parser.add_argument("--model", required=True, help="Normal preset/model id to pass to llama-server and chat completions.")
    parser.add_argument("--mtp-model", default=None, help="Optional preset/model id from the same models_preset.ini for MTP variants.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--api-key", default="testkey")
    parser.add_argument("--max-tokens", type=int, default=None, help="Optional completion limit. Omit to use server/model defaults.")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--prompt", action="append", help="Prompt turn. Repeat for multi-turn benchmark. Defaults to five bridge-engineering turns.")
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--keep-full-responses", action="store_true", help="Store complete model responses in JSON instead of only previews.")
    parser.add_argument("--variant", action="append", type=parse_variant, help="Custom variant as NAME::ARG ARG ARG. Can be repeated.")
    parser.add_argument("--output-dir", default=".ambient_data/benchmarks")
    parser.add_argument("--startup-timeout", type=float, default=90.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--log-settle-seconds", type=float, default=0.2)
    args = parser.parse_args()

    if args.prompt_file:
        prompt_text = Path(args.prompt_file).read_text(encoding="utf-8")
        prompts = [part.strip() for part in re.split(r"\n-{3,}\n", prompt_text) if part.strip()]
    else:
        prompts = args.prompt or DEFAULT_PROMPTS
    variants = args.variant or default_variants()
    results = []
    for variant in variants:
        print(f"Running {variant.name}: {' '.join(variant.args)}", flush=True)
        result = benchmark_variant(args, variant, prompts)
        results.append(result)
        print(json.dumps({"variant": result["variant"], "status": result["status"], "summary": result.get("summary"), "error": result.get("error")}, indent=2), flush=True)
    payload = {
        "created_at": datetime.now().isoformat(),
        "llama_server": args.llama_server,
        "models_preset": args.models_preset,
        "model": args.model,
        "mtp_model": args.mtp_model,
        "prompt_count": len(prompts),
        "prompt_chars": sum(len(item) for item in prompts),
        "results": results,
    }
    json_path, csv_path = write_outputs(Path(args.output_dir), payload)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path)}, indent=2))
    return 0 if any(result["status"] == "completed" for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
