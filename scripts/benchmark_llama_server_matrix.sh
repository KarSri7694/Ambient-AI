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
#   BATCH_START=2048
#   BATCH_MAX=4096
#   UBATCH_START=512
#   UBATCH_MAX=2048
#   BATCH_STEP=1024
#   MTP_DRAFT_MIN=1
#   MTP_DRAFT_MAX=4
#   MD_SUMMARY_FILE=/path/to/already-loaded.md
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
TURN_IDLE_TIMEOUT="${TURN_IDLE_TIMEOUT:-120}"
BATCH_START="${BATCH_START:-2048}"
BATCH_MAX="${BATCH_MAX:-4096}"
UBATCH_START="${UBATCH_START:-512}"
UBATCH_MAX="${UBATCH_MAX:-2048}"
BATCH_STEP="${BATCH_STEP:-1024}"
MTP_DRAFT_MIN="${MTP_DRAFT_MIN:-1}"
MTP_DRAFT_MAX="${MTP_DRAFT_MAX:-4}"
MD_SUMMARY_FILE="${MD_SUMMARY_FILE:-}"

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
  CONFIGS=()
  for ((batch_size=BATCH_START; batch_size<=BATCH_MAX; batch_size+=BATCH_STEP)); do
    for ((ubatch_size=UBATCH_START; ubatch_size<=UBATCH_MAX; ubatch_size+=BATCH_STEP)); do
      if (( ubatch_size > batch_size )); then
        continue
      fi
      CONFIGS+=(
        "base-b${batch_size}-ub${ubatch_size}::-c 200000 -ngl 999 -fa on -ctk q8_0 -ctv q8_0 -b ${batch_size} -ub ${ubatch_size}"
      )
    done
  done
  if [[ -n "$MTP_MODEL" && -f "$MTP_MODEL" ]]; then
    for ((batch_size=BATCH_START; batch_size<=BATCH_MAX; batch_size+=BATCH_STEP)); do
      for ((ubatch_size=UBATCH_START; ubatch_size<=UBATCH_MAX; ubatch_size+=BATCH_STEP)); do
        if (( ubatch_size > batch_size )); then
          continue
        fi
        for ((draft_tokens=MTP_DRAFT_MIN; draft_tokens<=MTP_DRAFT_MAX; draft_tokens++)); do
          CONFIGS+=(
            "mtp-d${draft_tokens}-b${batch_size}-ub${ubatch_size}::-c 200000 -ngl 999 -fa on -ctk q8_0 -ctv q8_0 -b ${batch_size} -ub ${ubatch_size} --model-draft $MTP_MODEL --spec-type draft-mtp --spec-draft-n-max ${draft_tokens} --spec-draft-n-min 1"
          )
        done
      done
    done
  fi
fi

MD_SUMMARY_FILE="$MD_SUMMARY_FILE" python3 - "$TMP_ROOT/prompts.json" <<'PY'
import json
import os
import sys
from pathlib import Path

md_path = os.environ.get("MD_SUMMARY_FILE", "").strip()
if md_path:
    markdown_text = Path(md_path).read_text(encoding="utf-8", errors="replace")
else:
    markdown_text = """# Radeon ROCm Benchmark Notes

This fallback Markdown document is used when MD_SUMMARY_FILE is not set. It describes an AI agent benchmark plan that compares local inference performance across llama-server batch sizes, micro-batch sizes, and speculative MTP draft token counts. The benchmark records time to first token, prompt preprocessing throughput, generation throughput, server command lines, and per-turn responses. The goal is to identify a stable, high-throughput configuration for AMD Radeon GPU inference while preserving multi-turn context behavior."""

prompts = [
    """Act as an expert game engineer specializing in retro rogue-like mechanics and 2D graphics math.

Your task is to build a complete, self-contained "Procedural Dungeon Generator with Dynamic Fog of War" inside a single, beautifully styled HTML file. Use vanilla HTML, CSS, and JavaScript with NO external libraries or dependencies.

### Visual & UI Requirements:
1. UI Layout: A clean, dark cyber-grid theme. Center the canvas. Include a control panel with:
   - A button to "Regenerate Map"
   - Text display showing current player coordinates
   - A button to toggle "God Mode" (revealing the entire map layout completely for debugging/demonstration).
2. Canvas Dimensions: Fixed at 600x600 pixels. Use `image-rendering: pixelated;` in CSS. The internal simulation grid should map to exactly 60x60 cells (meaning each map tile is a 10x10 pixel square on the canvas).
3. Theme Colors:
   - Unexplored/Hidden (Fog of War): Absolute pitch black (#000000)
   - Visible Walls: Deep stone gray (#4a4e69)
   - Visible Floor: Soft slate gray (#9a8c98)
   - Player Character: A bright, distinct green square or sprite (#55ff33)

### Core Mechanics & Algorithmic Rules:
You must implement a 2D grid array representing the map states (0 = Floor, 1 = Wall) and a secondary 2D array tracking visibility (0 = Hidden, 1 = Revealed/Visible).

1. Map Generation (Choose One Engine):
   - Option A: Binary Space Partitioning (BSP) to split the grid recursively into rectangular rooms, then carving straight hallway connections between them.
   - Option B: Cellular Automata (Fill grid randomly with ~45% walls, then run 4-5 simulation steps of Conway's-like rules to create organic cave layouts). Ensure there is an algorithm to connect isolated cavern pockets.
2. Player Movement: Spawn the player on a valid Floor tile. Implement standard keyboard controls (Arrow keys or WASD) to move the player exactly 1 grid tile per keypress. Prevent walking through Wall tiles.
3. Raycasting Line of Sight (The Core Math Challenge):
   - You must calculate the player's real-time visibility radius (limited to 8-10 tiles away).
   - To do this, cast virtual rays from the player's tile out to every single grid tile along the perimeter of their visibility radius.
   - For EACH ray, you MUST implement a generalized Bresenham's Line Algorithm to trace the line of grid cells step-by-step from the player to the destination tile.
   - Trace rules: Mark cells along the line as "Revealed". If the line hits a cell marked as a Wall, immediately stop tracing that ray further (blocking vision behind the wall).

### Deliverable:
Provide the complete, working code inside a single code block containing the HTML, CSS, and JavaScript. Do not leave any functions empty, do not use comments like "// implement logic here", and do not omit edge cases. It must run immediately when opened in a web browser.

Note: You may not have access to a browser, dev server, or vision tools — do not rely on running or visually inspecting the game for verification. Validate by reading and reasoning about the code instead.

Before finishing, review the code to confirm:
- Map generation (BSP or cellular automata) produces valid floor tiles with connectivity between areas
- Player moves one tile per keypress and cannot walk through walls
- Bresenham raycasting marks revealed cells along each ray and stops at walls
- God mode toggle reveals the full map; regenerate resets the map
- Complete single HTML file with no empty functions or placeholder logic""",
    f"""Read the following Markdown document and write a detailed, structured summary of it. Preserve important headings, decisions, technical constraints, commands, results, open questions, and action items. If the document contains implementation details, explain what was implemented and what remains unresolved.

--- MARKDOWN DOCUMENT START ---
{markdown_text}
--- MARKDOWN DOCUMENT END ---""",
    """Build a toy desktop environment as a single-page web app using vanilla HTML, CSS, and JavaScript only (no React, Vue, Svelte, etc.).

Do not reuse a tutorial layout or repo you already know. Implement from the rules below.

Goal:
A playful fake OS desktop in the browser — draggable windows, a taskbar, and two working applets. Prioritize correct window stacking, focus, and interaction first — that is what this test measures. Also give it a cohesive, presentable look using CSS only (dark theme, gradient wallpaper, styled title bars/taskbar, subtle borders or shadows on windows). Do not add extra features or assets just for visuals. It should feel interactive and good on screen, not a wireframe or a feature-heavy OS clone.

Technical requirements:
- Vanilla HTML, CSS, and JavaScript only.
- Runs locally without a backend: open index.html directly or use a minimal dev server (e.g. npx serve . or Vite).
- No external UI frameworks or window-manager libraries.
- Persist Notepad document text and desktop icon positions in localStorage so refresh does not lose them.
- Include a README with: how to run, feature list, brief file/structure notes.

Desktop shell:

1. Wallpaper & icons
   - Full-viewport desktop area with an attractive gradient wallpaper (CSS only — no image files required).
   - Two desktop icons: "Notepad" and "Calculator" (labels + clickable icons — emoji or simple SVG/CSS shapes are fine).
   - Double-click an icon to open its window (single-click may select/highlight if you want; opening must work reliably).
   - Icons are draggable on the desktop; positions persist after reload.

2. Windows (shared behavior)
   - Each app opens in its own window with: title bar, minimize button, close button.
   - Windows are draggable by the title bar only (not by clicking content inside).
   - Clicking a window brings it to the front (highest z-index). The focused window has a visibly distinct title bar style.
   - Only the focused window receives keyboard input for its app.
   - Minimize hides the window but keeps it in the taskbar; clicking the taskbar button restores it.
   - Close removes the window; reopening from the desktop icon creates a fresh instance (Notepad restores saved text from localStorage; Calculator starts cleared).
   - New windows open slightly offset so they do not perfectly stack on first launch.
   - Title text longer than 18 characters must ellipsize in the title bar (e.g. "Untitled — Notep…").

3. Taskbar
   - Fixed bar at the bottom of the screen.
   - Shows a button for each open (including minimized) window; label matches the window title (ellipsized if needed).
   - Clicking a taskbar button: restores if minimized; otherwise brings that window to the front.
   - Optional clock display is fine; not required.

Notepad app:

1. Multi-line text area filling the client area below the title bar.
2. Auto-save content to localStorage on input (debounced up to 500 ms is fine).
3. On open, load the last saved document from localStorage.
4. Window title reflects document state: default "Untitled — Notepad"; append a " •" or "*" when there are unsaved changes since last save, if you track that — otherwise static title is acceptable if auto-save runs on every input.

Calculator app:

1. Basic four-function calculator: +, −, ×, ÷.
2. Number buttons 0–9, decimal point, equals, and clear (C).
3. Display shows the current input or result; divide-by-zero shows "Error" and does not crash the app.
4. Keyboard support when the Calculator window is focused: digits, operators, Enter (=), Escape (clear).
5. Chain calculations are not required; each equals press evaluates the current expression is enough.

UX expectations:
- Visual style should be consistent across desktop, windows, taskbar, and apps (readable fonts, clear button states).
- Cursor changes on draggable regions (title bar, desktop icons).
- Minimized windows cannot be interacted with until restored.
- Clicking the desktop (not on a window) does not break window state.
- Layout works on a desktop browser; mobile-perfect design is not required.

Code quality expectations:
- Split logic across multiple JS files or clear modules (e.g. window manager, taskbar, notepad, calculator, storage) — not one unmaintainable script.
- Central window manager owns: open windows list, z-index, focus, minimize/restore/close.
- No dead UI: every button and icon must do something.

Scope guidance:
- Do NOT build: file system, multiple desktops, networking, themes, or more than the two apps above.
- Do NOT add npm runtime dependencies.
- Do NOT use iframes for app windows.

Deliverables:
- All source files to run the app
- README with run instructions
- App should be complete without manual code edits

Note: You may not have access to a browser, dev server, or vision tools — do not rely on running or visually inspecting the app for verification. Validate by reading and reasoning about the code instead.

Before finishing, review the code to confirm:
- Window manager assigns incrementing z-index on focus; focused window style is applied
- Dragging uses title-bar hit target only; content area drag does not move the window
- Minimize hides window and taskbar restore works; minimized windows are not focusable
- Close removes window and taskbar entry; reopen from icon works
- Only the focused window's app handles keyboard events
- Notepad loads/saves via localStorage; Calculator handles divide-by-zero safely
- Desktop icon positions persist via localStorage
- Title ellipsizes after 18 characters
- No empty functions, stubbed handlers, or placeholder "// TODO" logic remain""",
]

Path(sys.argv[1]).write_text(json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8")
PY

cat > "$CSV" <<'CSV'
config,status,run_index,turn_index,warmup,total_seconds,ttft_seconds,prompt_eval_tokens_per_second,server_eval_tokens_per_second,response_prompt_per_second,response_predicted_per_second,completion_chars,error
CSV

wait_for_server() {
  local deadline=$((SECONDS + STARTUP_TIMEOUT))
  while [[ $SECONDS -lt $deadline ]]; do
    if [[ -n "${SERVER_PID:-}" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
      return 1
    fi
    if curl -fsS "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

assert_port_free() {
  if curl -fsS "http://$HOST:$PORT/v1/models" >/dev/null 2>&1 || curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    echo "Port $HOST:$PORT already has a responding llama-server/API process." >&2
    echo "Stop it first, or run this benchmark with a different PORT." >&2
    echo "Example: pkill -f llama-server" >&2
    exit 2
  fi
}

wait_for_idle() {
  local deadline=$((SECONDS + TURN_IDLE_TIMEOUT))
  while [[ $SECONDS -lt $deadline ]]; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      return 1
    fi
    if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
      return 0
    fi
    if curl -fsS "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
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
    "n_predict": int(max_tokens),
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
try:
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
            if chunk.get("error"):
                raise RuntimeError(json.dumps(chunk["error"], ensure_ascii=False))
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
except Exception as exc:
    error_text = str(exc)
    if hasattr(exc, "read"):
        try:
            error_text = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
    with open(response_path, "w", encoding="utf-8") as handle:
        json.dump({"error": error_text, "total_seconds": time.perf_counter() - started}, handle, ensure_ascii=False)
    raise SystemExit(1)
ended = time.perf_counter()
answer = "".join(chunks)
if not answer and usage is None and not server_timings:
    with open(response_path, "w", encoding="utf-8") as handle:
        json.dump({"error": "empty streamed response with no usage or timings", "total_seconds": ended - started}, handle, ensure_ascii=False)
    raise SystemExit(1)
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

assert_port_free

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
  {
    printf 'benchmark config: %s\n' "$name"
    printf 'server command:'
    printf ' %q' "${server_command[@]}"
    printf '\n'
  } > "$log_file"
  "${server_command[@]}" >> "$log_file" 2>&1 &
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
    config_failed=false
    messages_file="$TMP_ROOT/messages-$name-$run_index.json"
    echo "[]" > "$messages_file"
    warmup=false
    if [[ "$run_index" -lt "$WARMUPS" ]]; then
      warmup=true
    fi
    prompt_count="$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$TMP_ROOT/prompts.json")"
    for ((turn_index=0; turn_index<prompt_count; turn_index++)); do
      prompt="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[int(sys.argv[2])])' "$TMP_ROOT/prompts.json" "$turn_index")"
      if ! wait_for_idle; then
        status="failed"
        error="server not reachable before turn"
        response_file="$TMP_ROOT/response-$name-$run_index-$turn_index.json"
        echo "{\"error\":\"$error\",\"total_seconds\":0}" > "$response_file"
        config_failed=true
      fi
      before_size="$(wc -c < "$log_file" | tr -d ' ')"
      if [[ "$config_failed" != "true" ]]; then
        response_file="$TMP_ROOT/response-$name-$run_index-$turn_index.json"
        status="completed"
        error=""
        if ! run_turn "$messages_file" "$prompt" "$response_file" "$MAX_TOKENS"; then
          status="failed"
          error="$(json_get "$response_file" error)"
          if [[ -z "$error" ]]; then
            error="request failed"
          fi
          config_failed=true
        fi
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
      if [[ "$config_failed" == "true" ]]; then
        echo "Stopping config $name after failed turn $turn_index: $error"
        break
      fi
    done
    if [[ "$config_failed" == "true" ]]; then
      break
    fi
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
