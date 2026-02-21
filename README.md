# 🌌 Ambient AI

**The Fully Local, Zero-Interaction Autonomous Agent.**

---

## 📖 Overview

**Ambient AI** is a fully local autonomous agent designed to *ambient* around you. Unlike traditional assistants that wait for commands, Ambient AI actively listens to your conversations and watches your screen to understand your intent and context in real-time.

Its core philosophy is **Zero-Interaction**, but it goes beyond simple automation. Ambient AI acts as a proactive partner:

- **Contextual Research:** Catches topics you discuss in passing, autonomously performs deep web research, and presents answers before you even ask.
- **Intelligent Planning:** Generates daily to-do lists based on your previous days' context and historical conversations.
- **Seamless Delegation:** Assign complex workflows by adding them to a **"Ambient AI Tasks"** project in Todoist — the agent picks these up and executes them during idle time.

---

## 🏗️ Architecture

The project follows the **Hexagonal Architecture** (Ports & Adapters) pattern, cleanly separating core logic from external dependencies.

```
src/
├── core/                        # Domain models & pure logic (NO outward deps)
│   ├── models.py                # Domain objects (ChatMessage, NightTask, etc.)
│   └── services/
│       └── merge_transcript.py          # Pure transcription merging logic
├── application/                 # Depends on core only
│   ├── ports/                   # Abstract interfaces (contracts)
│   │   ├── LLMProvider.py       # LLM generation & streaming
│   │   ├── modelManager.py      # Model loading/unloading
│   │   ├── tool_bridge_port.py  # MCP tool system
│   │   ├── notification_port.py # System notifications
│   │   ├── task_queue_port.py   # Night task queue
│   │   ├── task_provider_port.py# External task providers
│   │   ├── asr_port.py          # Speech recognition
│   │   ├── identity_port.py     # Speaker identification
│   │   └── voice_repository_port.py # Voice embedding store
│   └── services/                # Orchestration (depends on ports + core)
│       ├── llm_interaction_service.py   # Streaming chat loop + tool execution
│       └── night_mode_service.py        # Autonomous night-time processing
├── infrastructure/
│   └── adapter/                 # Concrete implementations
│       ├── llamaCppAdapter.py   # llama.cpp OpenAI-compatible server
│       ├── openVinoAdapter.py   # OpenVINO GenAI local inference
│       ├── MCPToolAdapter.py    # MCP bridge wrapper
│       ├── SQLiteNotificationAdapter.py
│       ├── SQLiteTaskQueueAdapter.py
│       ├── TodoistTaskAdapter.py
│       ├── ASR_Adapter.py       # Whisper transcription
│       ├── pyannoteAdapter.py   # Speaker diarization
│       ├── ecapaVoxcelebAdapter.py # Voice identification
│       └── SQLiteVoiceAdapter.py   # Voice embedding storage
├── app.py                       # Composition root (entry point)
├── server.py                    # FastAPI audio streaming server
└── mcp.json                     # MCP server configuration
```

---

## 🚀 Key Capabilities

### 👂 The "Ear" (Audio Intelligence)

- **Hinglish Transcription:** Fine-tuned Whisper model optimized for code-mixed audio — [Hindi2Hinglish (Oriserve)](https://huggingface.co/Oriserve/Whisper-Hindi2Hinglish-Apex)
- **Speaker Diarization:** Real-time VAD and speaker separation using **Pyannote**
- **Voice Identity:** Voice embedding creation using **SpeechBrain VoxCeleb** to distinguish the user from guests

### 🧠 The "Brain" (Reasoning & Control)

- **Local Inference:** Powered by **Qwen 3 VL 4B** (default) via llama.cpp
- **Notification System:** State-aware feedback loop that updates the LLM on outcomes of previous tasks
- **Chat Mode:** Direct interactive interface with the local LLM
- **Night Mode:** Fully autonomous task processing during idle hours

### 🛠️ The "Hands" (Tool Ecosystem)

Custom **MCP Bridge** enabling unlimited extensibility:

| Category | Tool | Description |
|---|---|---|
| **Productivity** | ✅ Todoist | Extracts and adds tasks from audio context |
| | 📅 Google Meet | Creates meetings from conversation details |
| | 📓 Obsidian | Manages your personal knowledge base |
| **Research** | 🌐 Tavily | Autonomous deep web search |
| | 🕸️ Web Browsing | Full page navigation and extraction |
| **System** | 🔌 Custom MCP | Bridge any MCP server (Filesystem, GitHub, etc.) |
| | 📊 Live Dashboard | Visualizes agent activity in real-time |
| | 🎙️ FastAPI Server | Stream audio notes for transcription |

---

## ⚡ Getting Started

### Prerequisites

- **Python 3.11+**
- **llama.cpp server** — running locally with OpenAI-compatible API (default: `http://localhost:8080`)
- **Node.js / npm** — required for MCP server tooling (`npx`)
- **CUDA-capable GPU** — recommended for model inference

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ambient-ai.git
cd ambient-ai

# Create and activate virtual environment
python -m venv venv

# Windows
./venv/Scripts/activate.ps1

# Linux / macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. **LLM Server** — Start llama.cpp server with the model preset manager or manually:
   ```bash
   # The app expects the server at http://localhost:8080
   # It will auto-load the model "Qwen3-VL-4b-Instruct-Q4_K_M" on startup
   ```

2. **MCP Tools** — Configure your MCP servers in `mcp.json`:
   ```json
   {
     "mcpServers": {
       "My MCP Server": {
         "command": "fastmcp",
         "args": ["run", "path/to/MCP_tools.py"],
         "env": { "TODOIST_API_TOKEN": "your-token" }
       }
     }
   }
   ```

3. **Todoist** *(optional)* — Set your API token in the `TODOIST_API_TOKEN` environment variable and configure `todoist.json` with your project ID.

4. **Night Queue Database** — Auto-initializes on first run. To manually initialize:
   ```bash
   python src/night_mode.py
   ```

### Backend Selection

Ambient AI supports two inference backends. Set the `LLM_BACKEND` environment variable to choose:

| Backend | Value | Description |
|---|---|---|
| **llama.cpp** *(default)* | `llamacpp` | Uses an external llama.cpp server with OpenAI-compatible API. Best for CUDA GPUs. |
| **OpenVINO** | `openvino` | Uses Intel's OpenVINO GenAI runtime. Best for Intel GPUs, CPUs, and NPUs. No separate server needed. |

#### llama.cpp (default — no extra config needed)

The default backend. Just start the llama.cpp server and run the app:

```bash
python src/app.py
```

#### OpenVINO

1. **Install the runtime:**
   ```bash
   pip install openvino-genai
   ```

2. **Prepare a model** — You need an OpenVINO-optimized model (IR format). You can convert GGUF/HuggingFace models using the [OpenVINO Model Conversion Guide](https://docs.openvino.ai/latest/openvino_docs_MO_DG_prepare_model_convert_model.html) or download pre-converted models from HuggingFace (look for repos with `-ov` or `-openvino` suffix).

3. **Set environment variables and run:**
   ```bash
   # Windows (PowerShell)
   $env:LLM_BACKEND = "openvino"
   $env:OPENVINO_MODEL_PATH = "path/to/your/openvino-model-dir"
   $env:OPENVINO_DEVICE = "GPU"   # Options: CPU, GPU, NPU
   python src/app.py

   # Linux / macOS
   LLM_BACKEND=openvino OPENVINO_MODEL_PATH=path/to/model OPENVINO_DEVICE=GPU python src/app.py
   ```

| Variable | Default | Description |
|---|---|---|
| `LLM_BACKEND` | `llamacpp` | `llamacpp` or `openvino` |
| `OPENVINO_MODEL_PATH` | `Qwen3-4B-int4-ov` | Path to the OpenVINO model directory |
| `OPENVINO_DEVICE` | `GPU` | Target device: `CPU`, `GPU`, or `NPU` |

### Running

```bash
# Main application (all modes)
python src/app.py

# FastAPI audio streaming server
python src/main.py
```

On launch, you'll be presented with three modes:

| Mode | Description |
|---|---|
| **1 — User Interaction** | Interactive chat with the LLM, with full tool access |
| **2 — Transcription Automation** | Processes `.txt` files in `transcriptions/` through the LLM |
| **3 — Night Mode** | Autonomous processing of queued tasks, Todoist tasks, and notifications |

---

### 🎥Demo Video

https://github.com/user-attachments/assets/bde47b83-526b-4ff2-8b0f-3119021149a1

---

## 🗺️ Future Roadmap

- [x] **Hexagonal Architecture Refactor:** Codebase refactored using Ports and Adapters pattern
- [ ] **GUI Agent:** Vision-based agent for direct screen control *(in progress)*
- [ ] **Context Fusion:** Merging audio and screenshot context streams
- [ ] **Model Fine-tuning:** Fine-tuning Qwen 3 for Ambient AI's autonomous workflows

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
