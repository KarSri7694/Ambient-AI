import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
import queue
import threading
import time
from typing import Optional

from application.services.llm_interaction_service import LLMInteractionService
from application.services.memory_consolidation_service import MemoryConsolidationService
from application.services.memory_context_builder import MemoryContextBuilder
from application.services.memory_extraction_service import MemoryExtractionService
from application.services.speaker_resolution_service import SpeakerResolutionService
from audio_agent import AudioAgentService
from infrastructure.adapter.llamaCppAdapter import LlamaCppAdapter
from infrastructure.adapter.MCPToolAdapter import MCPToolAdapter
from infrastructure.adapter.SQLiteMemoryAdapter import SQLiteMemoryAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8080"
DEFAULT_MODEL = "Qwen-3.5-9B-Q4_K_M"
MCP_CONFIG_PATH = "mcp.json"
PROJECT_ROOT = Path(__file__).parent.parent
PROMPTS_ROOT = PROJECT_ROOT / "prompts"
USER_DATA_DIR = Path(os.getenv("USER_DATA_DIR", "D:\\USER_DATA"))
MEMORY_ROOT = USER_DATA_DIR / "memory"
MEMORY_DB_PATH = PROJECT_ROOT / "database" / "memory.db"


def build_available_skills_summary() -> str:
    base_dir = PROJECT_ROOT
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


class TranscriptionService:
    def __init__(
        self,
        transcription_queue: queue.Queue,
        gpu_lock: Optional[threading.Lock] = None,
        audio_active_event: Optional[threading.Event] = None,
        llm_active_event: Optional[threading.Event] = None,
    ):
        self.queue = transcription_queue
        self.gpu_lock = gpu_lock
        self.audio_active_event = audio_active_event
        self.llm_active_event = llm_active_event

    def _build_services(self):
        llm_adapter = LlamaCppAdapter(base_url=API_BASE_URL)
        tool_bridge = MCPToolAdapter()
        llm_service = LLMInteractionService(
            llm_provider=llm_adapter,
            tool_bridge=tool_bridge,
        )
        memory_store = SQLiteMemoryAdapter(
            db_path=str(MEMORY_DB_PATH),
            memory_root=str(MEMORY_ROOT),
        )
        memory_context_builder = MemoryContextBuilder(
            memory=memory_store,
            prompts_root=str(PROMPTS_ROOT),
        )
        speaker_resolution = SpeakerResolutionService(memory=memory_store)
        memory_extraction = MemoryExtractionService(llm_provider=llm_adapter)
        memory_consolidation = MemoryConsolidationService(memory=memory_store)
        return (
            llm_adapter,
            tool_bridge,
            llm_service,
            memory_store,
            memory_context_builder,
            speaker_resolution,
            memory_extraction,
            memory_consolidation,
        )

    async def _process_one(
        self,
        llm_service: LLMInteractionService,
        memory_store: SQLiteMemoryAdapter,
        memory_context_builder: MemoryContextBuilder,
        speaker_resolution: SpeakerResolutionService,
        memory_extraction: MemoryExtractionService,
        content: str,
        transcript_path: str,
    ):
        turns = memory_extraction.parse_transcript(content)
        participants = speaker_resolution.resolve_labels(
            [turn.speaker_label for turn in turns],
            source_ref=Path(transcript_path).stem,
        )
        system_prompt = memory_context_builder.build_prompt(
            skills_summary=build_available_skills_summary(),
            participants=participants.values(),
        )

        try:
            await llm_service.run_interaction(
                user_input=f"Current date and time: {datetime.now()}     Transcript Content: {content}",
                system_prompt=system_prompt,
                model=DEFAULT_MODEL,
            )
            candidate_events = await memory_extraction.extract_events(
                transcript_text=content,
                participants=participants,
                model=DEFAULT_MODEL,
                source_ref=transcript_path,
            )
            for event in candidate_events:
                memory_store.append_event(event)
            logger.info(
                "Captured %s candidate memory events from %s",
                len(candidate_events),
                transcript_path,
            )
        finally:
            llm_service.reset_context()

    async def run_loop(self):
        (
            llm_adapter,
            tool_bridge,
            llm_service,
            memory_store,
            memory_context_builder,
            speaker_resolution,
            memory_extraction,
            memory_consolidation,
        ) = self._build_services()
        services_initialized = False

        idle_timeout = 20
        check_interval = 5
        idle_elapsed = 0

        while True:
            if self.audio_active_event.is_set():
                logger.info("Audio pipeline is active, agent waiting.")
                while self.audio_active_event.is_set():
                    await asyncio.sleep(1)
                logger.info("Audio pipeline finished. LLM taking over GPU.")
            self.llm_active_event.set()

            logger.info("LLM pipeline active.")
            idle_elapsed = 0
            while True:
                try:
                    file_path = self.queue.get_nowait()
                    idle_elapsed = 0
                    self.llm_active_event.set()
                    if not services_initialized:
                        logger.info("First transcript received, loading model.")
                        await llm_adapter.load_model(DEFAULT_MODEL)
                        await tool_bridge.start_servers(MCP_CONFIG_PATH)
                        await llm_service.initialize_tools()
                        services_initialized = True
                    logger.info("Ambient agent active.")
                except queue.Empty:
                    idle_elapsed += check_interval
                    logger.warning(
                        "No transcripts received. Idle for %ss / %ss before handback.",
                        idle_elapsed,
                        idle_timeout,
                    )
                    if idle_elapsed >= idle_timeout:
                        logger.info("LLM pipeline idle for %ss. Releasing GPU.", idle_timeout)
                        if services_initialized:
                            consolidated = memory_consolidation.consolidate()
                            logger.info("Memory consolidation processed %s events.", consolidated)
                            await llm_adapter.unload_model()
                            await tool_bridge.cleanup()
                            services_initialized = False
                        self.llm_active_event.clear()
                        self.audio_active_event.set()
                        break

                    await asyncio.sleep(check_interval)
                    continue

                logger.info("Picked up transcript: %s", file_path)

                try:
                    with open(file_path, "r", encoding="utf-8") as handle:
                        content = handle.read()
                except FileNotFoundError:
                    logger.error("Transcript file not found: %s", file_path)
                    self.queue.task_done()
                    continue

                try:
                    with self.gpu_lock:
                        await self._process_one(
                            llm_service=llm_service,
                            memory_store=memory_store,
                            memory_context_builder=memory_context_builder,
                            speaker_resolution=speaker_resolution,
                            memory_extraction=memory_extraction,
                            content=content,
                            transcript_path=file_path,
                        )
                except KeyboardInterrupt:
                    logger.info("TranscriptionService shutting down.")
                    break
                except Exception:
                    logger.exception("Failed to process transcript: %s", file_path)
                    await asyncio.sleep(5)
                finally:
                    self.queue.task_done()

        if services_initialized:
            consolidated = memory_consolidation.consolidate()
            logger.info("Memory consolidation processed %s events during shutdown.", consolidated)
            await llm_adapter.unload_model()
            await tool_bridge.cleanup()

    def start_service(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.run_loop())
        finally:
            loop.close()


if __name__ == "__main__":
    gpu_lock = threading.Lock()
    audio_active_event = threading.Event()
    llm_active_event = threading.Event()

    audio_agent = AudioAgentService(
        gpu_lock=gpu_lock,
        audio_active_event=audio_active_event,
        llm_active_event=llm_active_event,
    )
    transcription_queue = audio_agent.get_transcription_queue()

    t_service = TranscriptionService(
        transcription_queue=transcription_queue,
        gpu_lock=gpu_lock,
        audio_active_event=audio_active_event,
        llm_active_event=llm_active_event,
    )

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

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Main thread shutting down.")
