"""
Composition Root — wires all ports to concrete adapters and launches the application.

This replaces the original monolithic LLM.py entry point.
"""

import asyncio
import logging
import os
from pathlib import Path
import threading

from infrastructure.adapter.llamaCppAdapter import LlamaCppAdapter
from infrastructure.adapter.MCPToolAdapter import MCPToolAdapter
from infrastructure.adapter.SQLiteNotificationAdapter import SQLiteNotificationAdapter
from infrastructure.adapter.SQLiteTaskQueueAdapter import SQLiteTaskQueueAdapter
from infrastructure.adapter.TodoistTaskAdapter import TodoistTaskAdapter
from infrastructure.adapter.MSSScreenCaptureAdapter import MssScreenCaptureAdapter
from application.ports.screen_capture_port import ScreenCapturePort
from application.services.llm_interaction_service import LLMInteractionService
from application.services.night_mode_service import NightModeService
from utils.model_swapper import ModelSwapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────
API_BASE_URL = "http://localhost:8080"
DEFAULT_MODEL = "Qwen-4b-Thinking-2507-Q4_K_M"
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
    vision_tool_bridge = MCPToolAdapter()  # Separate bridge for vision tools if needed
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
            print("4. Enter Agentic Control Mode (Vision)")
            print("Type 'exit' to quit.\n")
            mode = input("Select mode (1, 2, 3, or 4): ")

            notifications_str = _format_notifications(notification_adapter)

            if mode == "1":
                await _user_mode(llm_service, llm_adapter, notifications_str)

            elif mode == "2":
                await _transcription_mode(llm_service, llm_adapter, notifications_str)

            elif mode == "3":
                await night_service.run_night_loop()

            elif mode == "4":
                await vision_tool_bridge.start_servers("GUI_mcp.json")
                try:
                    gui_agent_service = LLMInteractionService(
                        llm_provider=llm_adapter,
                        tool_bridge=vision_tool_bridge,
                    )
                    await gui_agent_service.initialize_tools()

                    screenshot_adapter = MssScreenCaptureAdapter()
                    async with ModelSwapper("Qwen3-VL-4b-Instruct-Q4_K_M", llm_adapter, gui_agent_service) as model:
                        await _agentic_control_mode(gui_agent_service, llm_adapter, screenshot_adapter, notifications_str, model)
                finally:
                    await vision_tool_bridge.cleanup()

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


async def _agentic_control_mode(
    llm_service: LLMInteractionService,
    llm_adapter: LlamaCppAdapter,
    screenshot_adapter: ScreenCapturePort,
    notifications_str: str,
    model: str
) -> None:
    """Agentic Control Mode: Captures screen and controls computer based on vision."""
    system_prompt = (
        "You are an agent that controls the computer using vision and tools. "
        "You will be provided with a screenshot of the current screen for every turn. "
        "Use the screenshot to understand the state of the system and use tools to achieve the user's goal."
        "Explain your actions before making a tool call."
        "Once clicked a text field do not click it again and again, assume it is selected and in next step start typing."
        "ALWAYS USE open_app tool open apps, DO NOT USE START BUTTON OR TASKBAR ICONS TO OPEN APPS. "
        
    )
    i=0
    while True:  # Limit iterations to prevent infinite loop during testing
        print("\n--- Agentic Control Mode ---")
        user_input = input("Target Goal (or 'exit' to quit)--> ")
        if user_input.lower() == "exit":
            break
            
        while i < 10:
            if i==0:
                logging.info("Waiting 10 seconds before first screenshot ...")
                threading.Event().wait(10)  # Wait a moment before first screenshot to allow user to prepare
            print("Capturing screen...")
            screenshot_path = screenshot_adapter.capture_screenshot()
            print(f"Screenshot captured: {screenshot_path}")
            
            # The agent will continue until it decides it's done or reached a limit.
            # Currently LLMInteractionService has its own MAX_ITERATIONS loop for tool calls.
            # But here we want a fresh screenshot for every 'turn' of interaction.
            # However, run_interaction ALREADY has a loop.
            # To get a NEW screenshot for every TOOL CALL iteration, we might need 
            # to modify run_interaction or handle it here.
            
            # The user request said: "after every turn model will get new screenshot"
            # In run_interaction, a 'turn' usually refers to one LLM call + tool calls.
            
            assistant_response = await llm_service.run_interaction(
                user_input=f"Current Objective: {user_input}",
                system_prompt=system_prompt,
                model=model,
                image_path=screenshot_path
            )
            
            print(f"\nAgent: {assistant_response}")
            
            # If the agent says it's done, we break to the next goal
            if "DONE" in assistant_response.upper() or "TASK COMPLETE" in assistant_response.upper():
                break
            
            logger.info("Waiting before next screenshot...")
            threading.Event().wait(5)
            # If we want it to be fully autonomous, we could just loop.
            # But let's ask the user if they want to continue or if it should auto-continue.
            i+=1

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
