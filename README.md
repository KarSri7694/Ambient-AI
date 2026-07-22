# Ambient AI

**A fully local ambient agent for audio, screen context, and idle-time autonomy.**

## Overview
Ambient AI is a local-first agent that listens to ongoing audio, optionally observes the screen, builds context over time, and executes bounded autonomous work when appropriate.

The current runtime is optimized around:

- Transcript ingestion and classification
- Passive screen observation and follow-up queueing
- Proactive research packaging
- Idle-time and night-mode task execution
- MCP-backed tool use for local automation

It is not a chat app with manual modes anymore. `src/app.py` runs the ambient runtime manager directly.

## Current Runtime
The main runtime in [src/app.py](src/app.py) manages:

- A primary llama.cpp server for the main chat/tool model via `API_BASE_URL` (default `http://localhost:8080`)
- A separate llama.cpp server for embeddings and reranking via `SEMANTIC_API_BASE_URL` (default `http://localhost:8081`)
- Transcript processing from the audio pipeline
- Idle detection and ambient runtime loading/unloading
- Optional passive screen observation when `PASSIVE_OBSERVER_ENABLED=true`
- Night-mode execution during the configured window (`20:00-23:00` in the current code)

Key subsystems:

- `LLMInteractionService`: tool-calling interaction loop, sub-agent loading, and direct-handled tools such as `capture_screen`
- `PassiveObserverService`: screenshot capture and visual observation persistence
- `PassiveObserverFollowupService`: queues durable follow-up tasks from unsent observations
- `NightModeService`: executes external tasks, queued night tasks, proactive research, queue dedupe, and reflection
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
- Simple task execution for low-risk tasks
- Proactive research queue and research vault
- Night queue execution with semantic deduplication
- Ambient agenda / reflection layer

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

Current code-level defaults in `src/app.py`:

- `DEFAULT_MODEL = "Qwen-3.5-9B-Mythos-Distilled-Q4_K_M-Vision"`
- `EMBEDDING_MODEL_PATH = "EmbeddingGemma"`
- `RERANKER_MODEL_PATH = "JinaReranker"`
- `USER_IDLE_THRESHOLD_SECONDS = 20`
- `NIGHT_MODE_START_HOUR = 20`
- `NIGHT_MODE_END_HOUR = 23`

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

For access from another device on the LAN, set `[log_api] host = 0.0.0.0` and
configure a strong `[chat] auth_token`. The runtime refuses non-loopback startup
without that token. Keep the dashboard on a trusted network.

## Running
Ambient AI currently expects multiple local processes.

### 1. Start the main llama.cpp server
Example:

```powershell
llama-server -m <main-model> --host 0.0.0.0 --port 8080 --slots --slot-save-path .\model_kv_states
```

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
- Ambient idle/night execution does not depend on passive observer being enabled.
- Screen capture for the active model turn is available through the `capture_screen` tool.
- Night-mode queue dedupe now keeps only semantically unique tasks before execution.
- Passive follow-up only evaluates unsent observations and marks reviewed observations as sent.

## Development Notes
Useful test entry points:

```powershell
pytest tests/test_night_mode.py -q
pytest tests/test_passive_observer.py -q
pytest tests/test_llm_interaction_service.py -q
```

## Roadmap
Near-term architecture gaps still visible in the codebase:

- Continuous active-use ambient behavior is still weaker than idle-time behavior
- Cross-modal context exists, but execution is still queue-heavy
- Tool/action autonomy is still conservative for many real workflows

## License
This project is licensed under the MIT License.
