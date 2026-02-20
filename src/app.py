"""
Composition Root — wires all ports to concrete adapters and launches the application.

This replaces the original monolithic LLM.py entry point.
"""

import asyncio
import logging
import os
from pathlib import Path

from infrastructure.adapter.llamaCppAdapter import LlamaCppAdapter
from infrastructure.adapter.MCPToolAdapter import MCPToolAdapter
from infrastructure.adapter.SQLiteNotificationAdapter import SQLiteNotificationAdapter
from infrastructure.adapter.SQLiteTaskQueueAdapter import SQLiteTaskQueueAdapter
from infrastructure.adapter.TodoistTaskAdapter import TodoistTaskAdapter
from application.services.llm_interaction_service import LLMInteractionService
from application.services.night_mode_service import NightModeService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────
API_BASE_URL = "http://localhost:8080"
DEFAULT_MODEL = "Qwen3-VL-4b-Instruct-Q4_K_M"
MCP_CONFIG_PATH = "mcp.json"

# Backend selection: "llamacpp" (default) or "openvino"
LLM_BACKEND = os.environ.get("LLM_BACKEND", "llamacpp").lower()
OPENVINO_DEVICE = os.environ.get("OPENVINO_DEVICE", "GPU")
OPENVINO_MODEL_PATH = os.environ.get("OPENVINO_MODEL_PATH", "Qwen3-4B-int4-ov")
USERNAME = ""

USER_SYSTEM_PROMPT = (
    "You are a helpful assistant. You can use tools to assist the user."
)

TRANSCRIPTION_SYSTEM_PROMPT = f"""
You are assistant of {USERNAME}, you have to help him in his daily tasks.
You have access to various tools to help you accomplish tasks. The user will pass a conversation to you, you have to interpret it and decide which tools to use to best assist the user.
When you need to perform a task, use the appropriate tool from the available tools list 
if possible convert text to english before passing to tools
Any task that requires extensive time or resources should be queued for night-time execution using the `queue_night_task` tool.
If a task is SLOW (Deep Research, Downloading huge files), use the `queue_night_task` tool. Dont use tavily for such tasks.
For all else, use standard tools.
You will also be provided with notifications from the system about important events that were started. Take these into account when assisting the user.
If you have no new notifications, ignore the notification section.
"""


def _read_transcriptions(transcriptions_dir: str) -> dict[str, str]:
    """Read all .txt files from the transcriptions directory."""
    path = Path(transcriptions_dir)
    contents = {}
    for f in path.iterdir():
        if f.is_file() and f.suffix == ".txt":
            contents[f.name] = f.read_text()
    return contents


async def run_app() -> None:
    # ── Wire adapters ─────────────────────────────────────────
    if LLM_BACKEND == "openvino":
        from infrastructure.adapter.openVinoAdapter import OpenVinoAdapter
        llm_adapter = OpenVinoAdapter(
            model_path=OPENVINO_MODEL_PATH,
            device=OPENVINO_DEVICE,
        )
        default_model = OPENVINO_MODEL_PATH
    else:
        llm_adapter = LlamaCppAdapter(base_url=API_BASE_URL)
        default_model = DEFAULT_MODEL

    tool_bridge = MCPToolAdapter()
    notification_adapter = SQLiteNotificationAdapter()
    task_queue_adapter = SQLiteTaskQueueAdapter()
    todoist_adapter = TodoistTaskAdapter()

    # ── Build services ────────────────────────────────────────
    llm_service = LLMInteractionService(
        llm_provider=llm_adapter,
        tool_bridge=tool_bridge,
    )
    night_service = NightModeService(
        task_queue=task_queue_adapter,
        task_provider=todoist_adapter,
        notification_port=notification_adapter,
        llm_service=llm_service,
        model=DEFAULT_MODEL,
        username=USERNAME,
    )

    try:
        # ── Startup ───────────────────────────────────────────
        await llm_adapter.load_model(default_model)
        await tool_bridge.start_servers(MCP_CONFIG_PATH)
        await llm_service.initialize_tools()

        # ── Mode selection loop ───────────────────────────────
        while True:
            print("\n1. Enter User interaction mode")
            print("2. Enter Transcription Automation mode")
            print("3. Enter late night execution mode")
            print("Type 'exit' to quit.\n")
            mode = input("Select mode (1, 2, or 3): ")

            notifications_str = _format_notifications(notification_adapter)

            if mode == "1":
                await _user_mode(llm_service, llm_adapter, notifications_str)

            elif mode == "2":
                await _transcription_mode(llm_service, llm_adapter, notifications_str)

            elif mode == "3":
                await night_service.run_night_loop()

            elif mode.lower() == "exit":
                print("Exiting program.")
                break

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        await llm_adapter.unload_model()
        await tool_bridge.cleanup()


def _format_notifications(notification_adapter: SQLiteNotificationAdapter) -> str:
    """Format unread notifications into a string."""
    notifications = notification_adapter.get_unread_notifications()
    if not notifications:
        return "No new notifications."
    lines = ["New notifications:"]
    for note in notifications:
        lines.append(f"- {note.message} (Source: {note.source})")
    return "\n".join(lines)


async def _user_mode(
    llm_service: LLMInteractionService,
    llm_adapter: LlamaCppAdapter,
    notifications_str: str,
) -> None:
    """Interactive user chat mode."""
    model = llm_adapter.get_current_model()
    while True:
        user_input = input("User--> ")
        if user_input.lower() == "exit":
            break
        await llm_service.run_interaction(
            user_input=f"{user_input}\n{notifications_str}",
            system_prompt=USER_SYSTEM_PROMPT,
            model=model,
        )


async def _transcription_mode(
    llm_service: LLMInteractionService,
    llm_adapter: LlamaCppAdapter,
    notifications_str: str,
) -> None:
    """Process transcription files through the LLM."""
    model = llm_adapter.get_current_model()
    contents = _read_transcriptions("transcriptions/")
    for filename, content in contents.items():
        print(f"\nProcessing transcription file: {filename}")
        await llm_service.run_interaction(
            user_input=content,
            system_prompt=TRANSCRIPTION_SYSTEM_PROMPT,
            model=model,
        )


if __name__ == "__main__":
    asyncio.run(run_app())
