# Ambient AI

**A local-first ambient agent that continuously judges when context warrants useful action.**

## Overview
Ambient AI is a local-first agent that listens to ongoing audio, optionally observes the screen, builds context over time, and executes bounded autonomous work when appropriate.

The current runtime is optimized around:

- Transcript ingestion and classification
- Passive screen observation and durable context events
- Proactive research packaging
- Policy-bounded actions and approvals
- A coherent, capability-filtered tool surface

It is not a chat app with manual modes anymore. `src/app.py` runs the ambient runtime manager directly.

## Current Runtime
The main runtime in [src/app.py](src/app.py) manages:

- A primary llama.cpp server for the main chat/tool model via `API_BASE_URL` (default `http://localhost:8080`)
- A separate llama.cpp server for embeddings and reranking via `SEMANTIC_API_BASE_URL` (default `http://localhost:8081`)
- Transcript processing from the audio pipeline
- Durable context-event ingestion and opportunity judgment during active and idle use
- Optional passive screen observation when `PASSIVE_OBSERVER_ENABLED=true`
- Exact-time work represented as policy-checked events rather than execution bypasses
- Continuous lightweight capture separated from resource-gated, bounded inference

Key subsystems:

- `LLMInteractionService`: tool-calling interaction loop, sub-agent loading, and direct-handled tools such as `capture_screen`
- `PassiveObserverService`: screenshot capture and visual observation persistence
- `AutonomyCoordinatorService`: turns context events into deduplicated opportunities, research, and inbox results
- `CapabilityPolicyService`: applies user policy tiers, approvals, budgets, and high-risk action boundaries to every tool call
- `PlainCaptureStore`: directly readable raw-context retention under the configured capture directory
- `ResourceGovernorService` and `ModelResidencyManager`: RAM/VRAM admission control and single-model residency
- `AmbientReflectionService`: bounded agenda/task/notification shaping

## Architecture
The project follows a ports-and-adapters structure:

```text
src/
|- core/
|  |- models.py
|- application/
|  |- ports/
|  |- services/
|- infrastructure/
|  |- adapter/
|- app.py
|- realtime_audio_input.py
|- MCP_tools.py
|- finance_tools.py
|- mcp.json
```

High-level storage/runtime components:

- SQLite-backed memory, agenda, activity ledger, interaction logs, task queue, proactive topic queue, finance DB
- MCP servers configured through `mcp.json`
- Local screenshot capture via `MssScreenCaptureAdapter`

## Key Capabilities
### Audio
- Real-time audio capture via `realtime_audio_input.py`
- Transcript normalization, classification, evidence extraction, session tracking, and open-loop extraction

### Vision
- Optional passive screenshot capture
- Visual observation persistence with follow-up send-state
- Cross-modal context fusion between transcript evidence and visual observations
- On-demand `capture_screen` tool for the active model turn

### Autonomy
- Continuous screen/transcript/scheduled-task event evaluation; idle state only affects resource priority
- Shadow and active judgment modes with durable leasing, retry, dead-letter, and semantic deduplication
- General proactive research and personalized planning delivered to the Proactive Inbox
- Typed capability packs, per-capability policy tiers, scoped approvals, and idempotent external writes

### Tools
Configured MCP tools currently include:

- Custom local MCP tools from [src/MCP_tools.py](src/MCP_tools.py)
- Finance tools from [src/finance_tools.py](src/finance_tools.py)
- Tavily MCP remote search
- A delegated Playwright MCP browser agent through `use_browser`

The exact active tool surface depends on the servers listed in [mcp.json](mcp.json).

## Prerequisites
- Python 3.11+
- A local llama.cpp server for the main model on `http://localhost:8080`
- A second llama.cpp server for embedding/reranking on `http://localhost:8081`
- Node.js / npm for MCP servers that use `npx`
- A Windows environment for the current idle detection and MSS-based screen capture path

## Installation
```powershell
git clone <your-repo>
cd ambient_ai
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configuration
Important environment variables:

```powershell
$env:API_BASE_URL="http://localhost:8080"
$env:SEMANTIC_API_BASE_URL="http://localhost:8081"
$env:PASSIVE_OBSERVER_ENABLED="true"   # optional
$env:USER_DATA_DIR="D:\USER_DATA"      # optional
```

Important runtime defaults in `src/app.py` and `config.example.ini`:

- `DEFAULT_MODEL = "Qwen-3.5-9B-Mythos-Distilled-Q4_K_M-Vision"`
- `EMBEDDING_MODEL_PATH = "EmbeddingGemma"`
- `RERANKER_MODEL_PATH = "JinaReranker"`
- autonomy starts in `shadow` mode
- 120 weighted tool calls/hour, 60 web queries/day, and 30 inbox cards/day

### MCP Configuration
The runtime reads MCP server definitions from [mcp.json](mcp.json).

At minimum, verify:

- the path to `src/MCP_tools.py`
- the path to `src/finance_tools.py`
- any required environment variables such as `GEMINI_API_KEY`, `SERPAPI_API_KEY`, or `TODOIST_API_TOKEN`

The `playwright` server in `mcp.example.json` is marked with
`"exposure": "browser_agent"`. Its raw `browser_*` tools are intentionally hidden
from the main model. The main model sees `use_browser(task)`; browser visibility is
selected by the user's `[browser] headless` setting. The delegated model runs with
the configured `browser_agent_model` and must end with
`finish_browser_task(exit_browser, status, summary)`. Setting `exit_browser=false`
returns control to the main model while retaining the browser session; retained
sessions close during application shutdown.

Do not commit real secrets in `mcp.json` or related config.

### Direct Chat

The runtime dashboard at `http://127.0.0.1:8765/` includes a persistent Chat tab.
Conversations are stored locally in SQLite, can be reopened later, and stream
responses while Ambient AI uses its normal MCP tools. Natural-language requests
for an exact future time are scheduled through `schedule_task_at`; overdue tasks
run the next time Ambient AI is available and publish their result to both the
originating conversation and Reports.

The dashboard has no login or security database and is deliberately restricted to
loopback (`127.0.0.1`, `localhost`, or `::1`). Non-loopback binding is refused.

The React dashboard also includes an Interaction Logs tab backed by
`interaction_logs.db`. Each model request is paired with its response, can be sorted
or filtered by date, and displays its visual input when one was recorded. Protected
ambient prompts remain hidden until explicitly revealed, and that reveal is audited.
The light/dark preference is saved locally in the browser.

The production frontend is committed under `src/infrastructure/runtime_ui/dist`, so
Node.js is not required to run Ambient AI. To work on the frontend, use Node.js 20.19+
or 22.12+ and run `npm install`, `npm run dev`, `npm test`, or `npm run build` from
`src/infrastructure/runtime_ui`. The development server proxies API calls to the local
runtime on port 8765.

Raw screenshots, processed audio, transcripts, and ambient model inputs are stored as
ordinary files under `USER_DATA_DIR/captures`. Raw data is retained indefinitely and is
never pruned automatically; the loopback-only capture APIs provide audited viewing,
export, and manual deletion. The dashboard also exposes a global capture
pause switch, live application/domain exclusions, and storage-pressure warnings.

### Resource-aware inference

Ambient awareness does not require heavy models to remain loaded. Window metadata,
UI Automation text, screenshots, and audio segments are persisted as plain files first.
Screenshots are taken during normal lightweight capture, but vision analysis waits for
a granted resource lease. Audio uses the same durable queue and defers ASR under memory
pressure. Deferred work is not counted as a failure and is never dropped.

No per-model memory estimates are configured or hardcoded. Before a model transition,
the governor checks current measured RAM and VRAM against the selected preset. After the
real model loads, it measures again and immediately unloads the model if the remaining
headroom is unsafe. In `balanced`, active work reserves 768 MB of VRAM and 2 GB of host
RAM; idle work reserves 512 MB of VRAM and 1.5 GB of host RAM. The dashboard exposes live
telemetry, the loaded model, deferred counts, and `capture_only`, `balanced`, and
`aggressive` presets.

## Running
Ambient AI currently expects multiple local processes.

### 1. Start the main llama.cpp router

Ambient AI manages model residency through llama.cpp's `/models/load` and
`/models/unload` endpoints, so start the server in router mode before `src/app.py`:

```powershell
llama-server --models-preset .\models_preset.ini --models-max 1 --no-models-autoload --host 127.0.0.1 --port 8080 --api-key testkey
```

Ambient AI does not start this process automatically. If it is unavailable, capture
continues and the runtime logs a clear error, but judgment and research remain deferred.

### 2. Start the semantic llama.cpp server
Example:

```powershell
llama-server -m <embedding-or-rerank-model> --host 0.0.0.0 --port 8081
```

This server is used for embeddings and reranking, not the main tool-using model.

### 3. Start the ambient runtime
```powershell
python src/app.py
```

This launches the ambient runtime manager plus the audio agent thread orchestration from `app.py`.

### 4. Start real-time audio input
```powershell
python src/realtime_audio_input.py
```

## Notes on Current Behavior
- Passive observation is optional and controlled by `PASSIVE_OBSERVER_ENABLED`.
- Startup never loads a full chat, vision, judgment, research, or ASR model.
- Opportunity judgment does not wait for user idle; idle detection is only a resource and interruption hint.
- Active-use inference runs only when the stricter resource headroom test passes; otherwise capture continues and work queues durably.
- Screen capture for the active model turn is available through the `capture_screen` tool.
- Scheduled work enters the same policy gateway as inferred and interactive work.
- Repeated context is coalesced into evolving opportunities, and negative inbox feedback suppresses recurrence.

## Development Notes
Useful test entry points:

```powershell
pytest tests/test_night_mode.py -q
pytest tests/test_passive_observer.py -q
pytest tests/test_llm_interaction_service.py -q
pytest tests/test_autonomy_control_plane.py -q
```

## License
This project is licensed under the MIT License.
