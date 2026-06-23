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
DEFAULT_MODEL = "Qwen-3.5-9B-Q4_K_M"
MCP_CONFIG_PATH = "mcp.json"
USERNAME = ""


def build_available_skills_summary() -> str:
    base_dir = Path(__file__).parent.parent
    skills_dir = base_dir / "skills"
    if not skills_dir.exists():
        skills_dir = base_dir / "Skills"

    skill_entries = []
    for skill_file in sorted(skills_dir.glob("*.md")):
        try:
            content = skill_file.read_text(encoding="utf-8")
        except OSError:
            continue

        if not content.startswith("---"):
            continue

        parts = content.split("---", 2)
        if len(parts) < 3:
            continue

        name = ""
        description = ""
        for line in parts[1].splitlines():
            stripped_line = line.strip()
            if stripped_line.startswith("name:"):
                name = stripped_line.partition(":")[2].strip()
            elif stripped_line.startswith("description:"):
                description = stripped_line.partition(":")[2].strip()

        if name and description:
            skill_entries.append(f"Name: {name}\nDescription: {description}")

    return "Available Skills:\n\n" + "\n\n".join(skill_entries)


def build_prompt():
    #Agent prompt
    path = Path(__file__).parent.parent / "prompts" / "AGENT.md"
    prompt = ""
    with open(path, "r") as f:
        prompt = f.read() + "\n\n"
    
    #details about user
    user_path = Path(__file__).parent.parent / "prompts" / "USER.md"
    with open(user_path, "r") as f:
        prompt += f.read() + "\n\n"
    
    #add available skills
    prompt += build_available_skills_summary()
    return prompt

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
                system_prompt=build_prompt(),
                model=DEFAULT_MODEL,
            )
        finally:
            # always unload even if inference threw an exception
            llm_service.reset_context()
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
