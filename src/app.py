import asyncio
import logging
import os
from pathlib import Path
import threading
import time
import queue
from datetime import datetime

from infrastructure.adapter.llamaCppAdapter import LlamaCppAdapter
from infrastructure.adapter.MCPToolAdapter import MCPToolAdapter
from application.services.llm_interaction_service import LLMInteractionService
from audio_agent import AudioAgentService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8080"
DEFAULT_MODEL = "Qwen-4b-Thinking-2507-Q4_K_M"
MCP_CONFIG_PATH = "mcp.json"
USERNAME = ""

TRANSCRIPTION_SYSTEM_PROMPT = f"""
You are assistant of {USERNAME}, you have to help him in his daily tasks.
You have access to various tools to help you accomplish tasks. The user will pass a conversation to you, you have to interpret it and decide which tools to use to best assist the user.
When you need to perform a task, use the appropriate tool from the available tools list.
If possible convert text to english before passing to tools.
Any task that requires extensive time or resources should be queued for night-time execution using the queue_night_task tool.
If a task is SLOW (Deep Research, Downloading huge files), use the queue_night_task tool.
For all else, use standard tools.
You will also be provided with notifications from the system about important events that were started.
If you have no new notifications, ignore the notification section.
"""


class TranscriptionService:

    def __init__(self, transcription_queue: queue.Queue, gpu_lock: threading.Lock = None, audio_active_event: threading.Event = None, llm_active_event: threading.Event = None):
        self.queue = transcription_queue
        self.gpu_lock = gpu_lock
        self.audio_active_event = audio_active_event
        self.llm_active_event = llm_active_event

    def _build_services(self):
        """Create service objects — no GPU work happens here."""
        llm_adapter = LlamaCppAdapter(base_url=API_BASE_URL)
        tool_bridge = MCPToolAdapter()
        llm_service = LLMInteractionService(
            llm_provider=llm_adapter,
            tool_bridge=tool_bridge,
        )
        return llm_adapter, tool_bridge, llm_service

    async def _process_one(
        self,
        llm_service: LLMInteractionService,
        content: str,
    ):
        """
        Load model → run inference → unload model.
        Entire method runs while holding the gpu_lock.
        """
        try:
            await llm_service.run_interaction(
                user_input=f"Current date and time: {datetime.now()}     Transcript Content: {content}",
                system_prompt=TRANSCRIPTION_SYSTEM_PROMPT,
                model=DEFAULT_MODEL,
            )
        finally:
            # always unload even if inference threw an exception
            llm_service.reset_conversation()
            return

    async def run_loop(self):
        """
        Main async loop — polls the transcript queue and processes
        one file at a time, acquiring the gpu_lock around each one.
        """
        llm_adapter, tool_bridge, llm_service = self._build_services()
        services_initialized = False

        IDLE_TIMEOUT = 20
        CHECK_INTERVAL = 5
        idle_elapsed = 0
        
        while True:
            # poll with timeout so we don't block the event loop forever
            if self.audio_active_event.is_set():
                logger.info("Audio Pipeline is active agent waiting")
                while self.audio_active_event.is_set():
                    await asyncio.sleep(1)
                logger.info("Audio pipeline finished. LLM taking over GPU.")    
            self.llm_active_event.set()

            logging.info("LLM pipeline active.")
            idle_elapsed = 0  # ← reset inside outer loop
            while True:
                try:
                    file_path = self.queue.get_nowait()
                    idle_elapsed = 0
                    self.llm_active_event.set()
                    if not services_initialized:
                        logging.info("First transcript received — loading model.")
                        await llm_adapter.load_model(DEFAULT_MODEL)
                        await tool_bridge.start_servers(MCP_CONFIG_PATH)
                        await llm_service.initialize_tools()
                        services_initialized = True
                    logging.info("Ambient Agent Now Active")
                    
                except queue.Empty:
                    idle_elapsed += CHECK_INTERVAL
                    logging.warning(
                        f"No transcripts received. "
                        f"Idle for {idle_elapsed}s / {IDLE_TIMEOUT}s before handback."
                    )
                    if idle_elapsed >= IDLE_TIMEOUT:
                        logger.info(
                        f"LLM pipeline was idle for {IDLE_TIMEOUT}s. Releasing GPU"
                    )
                        if services_initialized:
                            await llm_adapter.unload_model()
                            await tool_bridge.cleanup()
                            services_initialized = False
                        self.llm_active_event.clear()  # signal audio it can proceed
                        self.audio_active_event.set()
                        break
                        
                    await asyncio.sleep(CHECK_INTERVAL)  # yield control, check again soon
                    continue

                logger.info(f"Picked up transcript: {file_path}")

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except FileNotFoundError:
                    logger.error(f"Transcript file not found: {file_path}")
                    self.queue.task_done()
                    continue

                try:
                    # threading.Lock acquire is blocking — this is fine here because
                    # we are in a separate thread from the audio pipeline.
                    # It will block this thread (not the event loop coroutine) until
                    # the audio pipeline releases the lock.
                    with self.gpu_lock:
                        await self._process_one(
                            llm_service=llm_service, 
                            content=content
                            )
                        await llm_adapter.unload_model()
                        await tool_bridge.cleanup()
                except KeyboardInterrupt:
                    logger.info("TranscriptionService shutting down.")
                    break
                except Exception:
                    logger.exception(f"Failed to process transcript: {file_path}")
                    await asyncio.sleep(5)
                finally:
                    self.queue.task_done()
                    await llm_adapter.unload_model()
                    await tool_bridge.cleanup()

    def start_service(self):
        """
        Entry point when running in its own thread.
        Creates a brand new event loop for this thread.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.run_loop())
        finally:
            loop.close()


if __name__ == "__main__":
    # ── Single shared lock — both services receive the same object ──
    gpu_lock = threading.Lock()
    audio_active_event = threading.Event()
    llm_active_event = threading.Event()  
    # ── Audio side ────────────────────────────────────────────────
    audio_agent = AudioAgentService(
        gpu_lock=gpu_lock,
        audio_active_event=audio_active_event, llm_active_event=llm_active_event
        )
    transcription_queue = audio_agent.get_transcription_queue()

    # ── LLM side ──────────────────────────────────────────────────
    t_service = TranscriptionService(
        transcription_queue=transcription_queue,
        gpu_lock=gpu_lock,
        audio_active_event=audio_active_event,
        llm_active_event=llm_active_event
    )

    # ── Start both in separate threads ────────────────────────────
    audio_thread = threading.Thread(
        target=audio_agent.start_service,
        daemon=True,
        name="AudioPipelineThread",
    )
    transcription_thread = threading.Thread(
        target=t_service.start_service,
        daemon=True,
        name="TranscriptionLLMThread",
    )

    audio_thread.start()
    transcription_thread.start()

    # ── Keep main thread alive ────────────────────────────────────
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Main thread shutting down.")