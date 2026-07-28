#!/usr/bin/env bash
set -euo pipefail

# Standalone llama-server ROCm/HIP benchmark matrix.
#
# Required:
#   LLAMA_SERVER=/path/to/llama-server
#   MODEL=/path/to/main-model.gguf
#
# Optional:
#   MTP_MODEL=/path/to/mtp-or-draft-model.gguf
#   MMPROJ=/path/to/mmproj.gguf
#   HOST=127.0.0.1
#   PORT=8092
#   API_KEY=testkey
#   OUT_DIR=.ambient_data/benchmarks
#   RUNS=3
#   WARMUPS=1
#   MAX_TOKENS=1024
#
# Custom configs:
#   CONFIGS=("name::llama-server args" "other::llama-server args")

LLAMA_SERVER="${LLAMA_SERVER:-$(command -v llama-server || true)}"
MODEL="${MODEL:-/persistent/models/Gemma-4-26B/gemma-4-26B-A4B-it-Q8_0.gguf}"
MMPROJ="${MMPROJ:-/persistent/models/Gemma-4-26B/mmproj-F16.gguf}"
MTP_MODEL="${MTP_MODEL:-/persistent/models/Gemma-4-26B/mtp-gemma-4-26B-A4B-it.gguf}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8092}"
API_KEY="${API_KEY:-testkey}"
OUT_DIR="${OUT_DIR:-.ambient_data/benchmarks}"
RUNS="${RUNS:-3}"
WARMUPS="${WARMUPS:-1}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-120}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-900}"
LOG_SETTLE_SECONDS="${LOG_SETTLE_SECONDS:-0.25}"
MAX_TOKENS="${MAX_TOKENS:-1024}"

if [[ -z "$LLAMA_SERVER" || ! -x "$LLAMA_SERVER" ]]; then
  echo "Set LLAMA_SERVER to an executable llama-server path." >&2
  exit 2
fi
if [[ -z "$MODEL" || ! -f "$MODEL" ]]; then
  echo "Set MODEL to the main GGUF model path." >&2
  exit 2
fi
if [[ -n "$MMPROJ" && ! -f "$MMPROJ" ]]; then
  echo "Set MMPROJ to an existing mmproj GGUF path, or set MMPROJ= to skip it." >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
JSONL="$OUT_DIR/llama-server-matrix-$STAMP.jsonl"
CSV="$OUT_DIR/llama-server-matrix-$STAMP.csv"
SUMMARY="$OUT_DIR/llama-server-matrix-$STAMP-summary.json"
TMP_ROOT="$(mktemp -d)"
SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 1
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

if ! declare -p CONFIGS >/dev/null 2>&1 || [[ ${#CONFIGS[@]} -eq 0 ]]; then
  CONFIGS=(
    "q8-b4096-ub1024::-c 200000 -ngl 999 -fa on -ctk q8_0 -ctv q8_0 -b 4096 -ub 1024"
    "q4-b4096-ub1024::-c 200000 -ngl 999 -fa on -ctk q4_0 -ctv q4_0 -b 4096 -ub 1024"
    "q8-b8192-ub2048::-c 200000 -ngl 999 -fa on -ctk q8_0 -ctv q8_0 -b 8192 -ub 2048"
  )
  if [[ -n "$MTP_MODEL" && -f "$MTP_MODEL" ]]; then
    CONFIGS+=(
      "mtp-q8-b4096-ub1024::-c 200000 -ngl 999 -fa on -ctk q8_0 -ctv q8_0 -b 4096 -ub 1024 --model-draft $MTP_MODEL --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-min 1"
      "mtp-q4-b4096-ub1024::-c 200000 -ngl 999 -fa on -ctk q4_0 -ctv q4_0 -b 4096 -ub 1024 --model-draft $MTP_MODEL --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-min 1"
    )
  fi
fi

cat > "$TMP_ROOT/prompts.json" <<'JSON'
[
  "Explain the history, engineering tradeoffs, and practical applications of suspension bridges. Include load paths, materials, failure modes, and how modern monitoring systems improve safety.",
  "Now design a fictional suspension bridge for a windy coastal city. Give constraints, dimensions, materials, inspection routines, and the reasoning behind each decision.",
  "A city council says the bridge must also support emergency evacuation, cycling lanes, maintenance robots, and severe corrosion risk. Revise the design while preserving earlier assumptions where possible.",
  "Create a risk register for the revised bridge. Include likelihood, impact, detection strategy, mitigation, and the owner responsible for each risk.",
  "Summarize the entire bridge proposal as an executive briefing, explicitly referencing earlier design decisions and explaining what changed across the conversation."
]
JSON

cat > "$CSV" <<'CSV'
config,status,run_index,turn_index,warmup,total_seconds,ttft_seconds,prompt_eval_tokens_per_second,server_eval_tokens_per_second,response_prompt_per_second,response_predicted_per_second,completion_chars,error
CSV

wait_for_server() {
  local deadline=$((SECONDS + STARTUP_TIMEOUT))
  until curl -fsS "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; do
    if [[ $SECONDS -ge $deadline ]]; then
      return 1
    fi
    if [[ -n "${SERVER_PID:-}" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
      return 1
    fi
    sleep 1
  done
}

stop_server() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}

parse_perf_from_log() {
  local log_file="$1"
  local offset="$2"
  python3 - "$log_file" "$offset" <<'PY'
import json, re, sys
path, offset = sys.argv[1], int(sys.argv[2])
with open(path, "r", encoding="utf-8", errors="replace") as handle:
    handle.seek(offset)
    text = handle.read()
prompt = list(re.finditer(r"prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*tokens\s*\(([0-9.]+)\s*tokens per second\)", text, re.I))
gen = list(re.finditer(r"\beval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*runs?\s*\(([0-9.]+)\s*tokens per second\)", text, re.I))
payload = {}
if prompt:
    m = prompt[-1]
    payload.update(prompt_eval_ms=float(m.group(1)), prompt_eval_tokens=int(m.group(2)), prompt_eval_tokens_per_second=float(m.group(3)))
if gen:
    m = gen[-1]
    payload.update(server_eval_ms=float(m.group(1)), server_eval_tokens=int(m.group(2)), server_eval_tokens_per_second=float(m.group(3)))
print(json.dumps(payload))
PY
}

run_turn() {
  local messages_file="$1"
  local prompt="$2"
  local response_file="$3"
  local max_tokens="$4"
  python3 - "$HOST" "$PORT" "$API_KEY" "$REQUEST_TIMEOUT" "$messages_file" "$prompt" "$response_file" "$max_tokens" <<'PY'
import json, sys, time, urllib.request

host, port, api_key, timeout, messages_path, prompt, response_path, max_tokens = sys.argv[1:9]
timeout = float(timeout)
with open(messages_path, "r", encoding="utf-8") as handle:
    messages = json.load(handle)
messages.append({"role": "user", "content": prompt})
body = json.dumps({
    "model": "local-model",
    "messages": messages,
    "max_tokens": int(max_tokens),
    "stream": True,
    "stream_options": {"include_usage": True},
}).encode("utf-8")
headers = {"Content-Type": "application/json"}
if api_key:
    headers["Authorization"] = f"Bearer {api_key}"
request = urllib.request.Request(f"http://{host}:{port}/v1/chat/completions", data=body, headers=headers, method="POST")
started = time.perf_counter()
first = None
chunks = []
usage = None
server_timings = {}
with urllib.request.urlopen(request, timeout=timeout) as response:
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        if chunk.get("usage"):
            usage = chunk["usage"]
        for key in ("prompt_ms", "prompt_n", "prompt_per_second", "predicted_ms", "predicted_n", "predicted_per_second", "timings"):
            if key in chunk:
                server_timings[key] = chunk[key]
        choices = chunk.get("choices") or []
        delta = None
        if choices:
            delta = choices[0].get("delta", {}).get("content")
        if delta:
            if first is None:
                first = time.perf_counter()
            chunks.append(delta)
ended = time.perf_counter()
answer = "".join(chunks)
messages.append({"role": "assistant", "content": answer})
with open(messages_path, "w", encoding="utf-8") as handle:
    json.dump(messages, handle, ensure_ascii=False)
nested = server_timings.get("timings") if isinstance(server_timings.get("timings"), dict) else {}
payload = {
    "total_seconds": ended - started,
    "ttft_seconds": (first - started) if first else None,
    "completion_chars": len(answer),
    "usage": usage,
    "server_timings": server_timings,
    "response_prompt_per_second": server_timings.get("prompt_per_second") or nested.get("prompt_per_second"),
    "response_predicted_per_second": server_timings.get("predicted_per_second") or nested.get("predicted_per_second"),
    "response_preview": answer[:500],
}
with open(response_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False)
PY
}

json_get() {
  python3 - "$1" "$2" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    data = json.load(handle)
value = data
for part in sys.argv[2].split("."):
    if not part:
        continue
    value = value.get(part) if isinstance(value, dict) else None
print("" if value is None else value)
PY
}

csv_escape() {
  python3 - "$1" <<'PY'
import csv, io, sys
buf = io.StringIO()
csv.writer(buf).writerow([sys.argv[1]])
print(buf.getvalue().strip())
PY
}

for entry in "${CONFIGS[@]}"; do
  name="${entry%%::*}"
  cfg_args="${entry#*::}"
  log_file="$OUT_DIR/llama-server-$STAMP-$name.log"
  echo "Starting config: $name"
  # shellcheck disable=SC2086
  server_command=(
    "$LLAMA_SERVER"
    --host "$HOST" \
    --port "$PORT" \
    --api-key "$API_KEY" \
    -m "$MODEL" \
    --perf
  )
  if [[ -n "$MMPROJ" ]]; then
    server_command+=(--mmproj "$MMPROJ")
  fi
  # shellcheck disable=SC2206
  cfg_array=($cfg_args)
  server_command+=("${cfg_array[@]}")
  "${server_command[@]}" > "$log_file" 2>&1 &
  SERVER_PID="$!"

  if ! wait_for_server; then
    error="server did not become ready"
    echo "{\"config\":\"$name\",\"status\":\"failed\",\"error\":\"$error\",\"log_file\":\"$log_file\"}" >> "$JSONL"
    echo "$(csv_escape "$name"),failed,,,,,,,,,,,,$(csv_escape "$error")" >> "$CSV"
    stop_server
    continue
  fi

  total_iterations=$((WARMUPS + RUNS))
  for ((run_index=0; run_index<total_iterations; run_index++)); do
    messages_file="$TMP_ROOT/messages-$name-$run_index.json"
    echo "[]" > "$messages_file"
    warmup=false
    if [[ "$run_index" -lt "$WARMUPS" ]]; then
      warmup=true
    fi
    prompt_count="$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$TMP_ROOT/prompts.json")"
    for ((turn_index=0; turn_index<prompt_count; turn_index++)); do
      prompt="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[int(sys.argv[2])])' "$TMP_ROOT/prompts.json" "$turn_index")"
      before_size="$(wc -c < "$log_file" | tr -d ' ')"
      response_file="$TMP_ROOT/response-$name-$run_index-$turn_index.json"
      status="completed"
      error=""
      if ! run_turn "$messages_file" "$prompt" "$response_file" "$MAX_TOKENS"; then
        status="failed"
        error="request failed"
        echo "{}" > "$response_file"
      fi
      sleep "$LOG_SETTLE_SECONDS"
      perf_file="$TMP_ROOT/perf-$name-$run_index-$turn_index.json"
      parse_perf_from_log "$log_file" "$before_size" > "$perf_file"
      merged_file="$TMP_ROOT/merged-$name-$run_index-$turn_index.json"
      python3 - "$response_file" "$perf_file" "$merged_file" "$name" "$status" "$run_index" "$turn_index" "$warmup" "$error" "$cfg_args" "$log_file" <<'PY'
import json, sys
response_path, perf_path, merged_path, name, status, run_index, turn_index, warmup, error, cfg_args, log_file = sys.argv[1:12]
with open(response_path, "r", encoding="utf-8") as handle:
    response = json.load(handle)
with open(perf_path, "r", encoding="utf-8") as handle:
    perf = json.load(handle)
payload = {
    "config": name,
    "status": status,
    "run_index": int(run_index),
    "turn_index": int(turn_index),
    "warmup": warmup == "true",
    "server_args": cfg_args,
    "log_file": log_file,
    "error": error or None,
    **response,
    **perf,
}
with open(merged_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False)
print(json.dumps(payload, ensure_ascii=False))
PY
      cat "$merged_file" >> "$JSONL"
      echo >> "$JSONL"
      echo "$(csv_escape "$name"),$status,$run_index,$turn_index,$warmup,$(json_get "$merged_file" total_seconds),$(json_get "$merged_file" ttft_seconds),$(json_get "$merged_file" prompt_eval_tokens_per_second),$(json_get "$merged_file" server_eval_tokens_per_second),$(json_get "$merged_file" response_prompt_per_second),$(json_get "$merged_file" response_predicted_per_second),$(json_get "$merged_file" completion_chars),$(csv_escape "$error")" >> "$CSV"
    done
  done
  stop_server
done

python3 - "$JSONL" "$SUMMARY" <<'PY'
import json, statistics, sys
jsonl, summary_path = sys.argv[1:3]
rows = []
with open(jsonl, "r", encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if line:
            rows.append(json.loads(line))
summary = {}
for config in sorted({row.get("config") for row in rows}):
    measured = [row for row in rows if row.get("config") == config and not row.get("warmup") and row.get("status") == "completed"]
    item = {"turns": len(measured)}
    for field in ("ttft_seconds", "total_seconds", "prompt_eval_tokens_per_second", "server_eval_tokens_per_second", "response_prompt_per_second", "response_predicted_per_second"):
        values = [float(row[field]) for row in measured if row.get(field) not in (None, "")]
        item[f"{field}_median"] = statistics.median(values) if values else None
        item[f"{field}_mean"] = statistics.fmean(values) if values else None
    summary[config] = item
with open(summary_path, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)
print(json.dumps(summary, indent=2))
PY

echo "JSONL: $JSONL"
echo "CSV: $CSV"
echo "Summary: $SUMMARY"
