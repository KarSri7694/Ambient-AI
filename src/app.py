import asyncio
from dataclasses import replace
import logging
import os
from datetime import datetime
from pathlib import Path
import queue
import threading
import time
from typing import Optional
import uuid

from application.services.llm_interaction_service import LLMInteractionService
from application.services.ambient_reflection_service import AmbientReflectionService
from application.services.agenda_scoring_service import AgendaScoringService
from application.services.memory_consolidation_service import MemoryConsolidationService
from application.services.memory_context_builder import MemoryContextBuilder
from application.services.proactive_research_service import ProactiveResearchService
from application.services.proactive_topic_detection_service import ProactiveTopicDetectionService
from application.services.research_vault_service import ResearchVaultService
from application.services.session_tracker_service import SessionTrackerService
from application.services.simple_task_execution_service import SimpleTaskExecutionService
from application.services.speaker_resolution_service import SpeakerResolutionService
from application.services.transcript_evidence_service import TranscriptEvidenceService
from application.services.transcript_classification_service import TranscriptClassificationService
from application.services.open_loop_service import OpenLoopService
from application.services.user_profile_service import UserProfileService
from audio_agent import AudioAgentService
from infrastructure.adapter.llamaCppAdapter import LlamaCppAdapter
from infrastructure.adapter.LoggingLLMProvider import LoggingLLMProvider
from infrastructure.adapter.MCPToolAdapter import MCPToolAdapter
from infrastructure.adapter.SQLiteMemoryAdapter import SQLiteMemoryAdapter
from infrastructure.adapter.SQLiteAmbientAgendaAdapter import SQLiteAmbientAgendaAdapter
from infrastructure.adapter.SQLiteInteractionLogAdapter import SQLiteInteractionLogAdapter
from infrastructure.adapter.SQLiteNotificationAdapter import SQLiteNotificationAdapter
from infrastructure.adapter.SQLiteProactiveTopicQueueAdapter import SQLiteProactiveTopicQueueAdapter
from infrastructure.adapter.SQLiteTaskQueueAdapter import SQLiteTaskQueueAdapter
from core.models import MemoryEvent, TranscriptClassificationResult

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
PROACTIVE_TOPICS_DB_PATH = PROJECT_ROOT / "database" / "proactive_topics.db"
AMBIENT_AGENDA_DB_PATH = PROJECT_ROOT / "database" / "ambient_agenda.db"
INTERACTION_LOG_DB_PATH = PROJECT_ROOT / "database" / "interaction_logs.db"
CURRENT_RESPONSE_PATH = PROJECT_ROOT / "database" / "current_llm_response.md"
RESEARCH_VAULT_ROOT = USER_DATA_DIR / "research_vault"
CLASSIFIER_PROMPT = "TRANSCRIPT_CLASSIFIER.md"
SIMPLE_EXECUTOR_PROMPT = "SIMPLE_EXECUTOR.md"
PROACTIVE_RESEARCH_PROMPT = "PROACTIVE_RESEARCH.md"


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

    async def _ensure_llm_ready(self, llm_adapter, tool_bridge, llm_service) -> None:
        await llm_adapter.load_model(DEFAULT_MODEL)
        await tool_bridge.start_servers(MCP_CONFIG_PATH)
        await llm_service.initialize_tools()
        self.llm_active_event.set()

    async def _release_llm(self, llm_adapter, tool_bridge) -> None:
        await llm_adapter.unload_model()
        await tool_bridge.cleanup()
        self.llm_active_event.clear()

    async def _run_idle_cycle(
        self,
        *,
        memory_consolidation: MemoryConsolidationService,
        proactive_research_service: ProactiveResearchService,
        ambient_reflection: AmbientReflectionService,
        memory_store: SQLiteMemoryAdapter,
        proactive_research_prompt: str,
    ) -> None:
        consolidated = memory_consolidation.consolidate()
        if consolidated:
            logger.info("Memory consolidation processed %s events.", consolidated)
        processed = await proactive_research_service.process_topics(
            system_prompt=proactive_research_prompt,
            model=DEFAULT_MODEL,
            max_topics=1,
        )
        if processed:
            logger.info("Processed %s proactive research topics during idle.", processed)
        reflection_actions = await ambient_reflection.reflect(
            model=DEFAULT_MODEL,
            recent_context=memory_store.get_recent_context(),
        )
        if reflection_actions:
            logger.info(
                "Ambient reflection produced %s action(s): %s",
                len(reflection_actions),
                ", ".join(action.action_type for action in reflection_actions),
            )

    def _build_services(self):
        llm_adapter = LlamaCppAdapter(base_url=API_BASE_URL)
        interaction_log_store = SQLiteInteractionLogAdapter(
            db_path=str(INTERACTION_LOG_DB_PATH),
        )
        logged_llm = LoggingLLMProvider(
            provider=llm_adapter,
            log_store=interaction_log_store,
            current_response_path=str(CURRENT_RESPONSE_PATH),
        )
        tool_bridge = MCPToolAdapter()
        llm_service = LLMInteractionService(
            llm_provider=logged_llm,
            tool_bridge=tool_bridge,
        )
        task_queue = SQLiteTaskQueueAdapter()
        notification_store = SQLiteNotificationAdapter()
        agenda_store = SQLiteAmbientAgendaAdapter(
            db_path=str(AMBIENT_AGENDA_DB_PATH),
        )
        proactive_topic_queue = SQLiteProactiveTopicQueueAdapter(
            db_path=str(PROACTIVE_TOPICS_DB_PATH),
        )
        memory_store = SQLiteMemoryAdapter(
            db_path=str(MEMORY_DB_PATH),
            memory_root=str(MEMORY_ROOT),
        )
        research_vault = ResearchVaultService(str(RESEARCH_VAULT_ROOT))
        memory_context_builder = MemoryContextBuilder(
            memory=memory_store,
            prompts_root=str(PROMPTS_ROOT),
        )
        speaker_resolution = SpeakerResolutionService(memory=memory_store)
        classifier = TranscriptClassificationService(llm_provider=logged_llm)
        evidence_service = TranscriptEvidenceService(llm_provider=logged_llm)
        session_tracker = SessionTrackerService(memory=memory_store, llm_provider=logged_llm)
        open_loop_service = OpenLoopService(memory=memory_store, llm_provider=logged_llm)
        user_profile_service = UserProfileService(memory=memory_store, llm_provider=logged_llm)
        simple_executor = SimpleTaskExecutionService(llm_service=llm_service)
        proactive_topic_detector = ProactiveTopicDetectionService(
            queue=proactive_topic_queue,
            vault=research_vault,
        )
        proactive_research_service = ProactiveResearchService(
            llm_service=llm_service,
            topic_queue=proactive_topic_queue,
            vault=research_vault,
            notifications=notification_store,
        )
        memory_consolidation = MemoryConsolidationService(memory=memory_store)
        agenda_scorer = AgendaScoringService(agenda=agenda_store)
        ambient_reflection = AmbientReflectionService(
            llm_service=llm_service,
            memory=memory_store,
            agenda=agenda_store,
            notifications=notification_store,
            task_queue=task_queue,
            topic_queue=proactive_topic_queue,
            scorer=agenda_scorer,
        )
        return (
            llm_adapter,
            tool_bridge,
            llm_service,
            task_queue,
            notification_store,
            agenda_store,
            proactive_topic_queue,
            memory_store,
            research_vault,
            memory_context_builder,
            speaker_resolution,
            classifier,
            evidence_service,
            session_tracker,
            open_loop_service,
            user_profile_service,
            simple_executor,
            proactive_topic_detector,
            proactive_research_service,
            memory_consolidation,
            ambient_reflection,
        )

    async def _process_one(
        self,
        llm_service: LLMInteractionService,
        task_queue: SQLiteTaskQueueAdapter,
        notification_store: SQLiteNotificationAdapter,
        proactive_topic_detector: ProactiveTopicDetectionService,
        memory_store: SQLiteMemoryAdapter,
        memory_context_builder: MemoryContextBuilder,
        speaker_resolution: SpeakerResolutionService,
        classifier: TranscriptClassificationService,
        evidence_service: TranscriptEvidenceService,
        session_tracker: SessionTrackerService,
        open_loop_service: OpenLoopService,
        user_profile_service: UserProfileService,
        simple_executor: SimpleTaskExecutionService,
        content: str,
        transcript_path: str,
    ):
        turns = classifier.parse_transcript(content)
        if not turns:
            logger.info("No parsed turns found in transcript %s.", transcript_path)
            return
        participants = speaker_resolution.resolve_labels(
            [turn.speaker_label for turn in turns],
            source_ref=Path(transcript_path).stem,
        )
        evidence_items = await evidence_service.extract(
            turns=turns,
            participants=participants,
            source_ref=transcript_path,
            model=DEFAULT_MODEL,
        )
        session = await session_tracker.attach_to_session(evidence_items, model=DEFAULT_MODEL)
        evidence_items = [replace(item, session_id=session.session_id) for item in evidence_items]
        for item in evidence_items:
            memory_store.append_evidence(item)
        session_tracker.refresh_digest()
        open_loops = await open_loop_service.process(
            session=session,
            evidence_items=evidence_items,
            model=DEFAULT_MODEL,
        )
        if open_loops:
            logger.info("Updated %s open loop(s) from transcript state.", len(open_loops))
        user_speaker = user_profile_service.infer_user_speaker()
        facets = await user_profile_service.update_from_evidence(
            user_speaker=user_speaker,
            evidence_items=evidence_items,
            model=DEFAULT_MODEL,
        )
        if facets:
            logger.info("Updated %s user profile facet(s).", len(facets))
        classifier_prompt = memory_context_builder.build_prompt(
            base_prompt_filename=CLASSIFIER_PROMPT,
            skills_summary=build_available_skills_summary(),
            participants=participants.values(),
            include_skills=False,
        )
        executor_prompt = memory_context_builder.build_prompt(
            base_prompt_filename=SIMPLE_EXECUTOR_PROMPT,
            skills_summary=build_available_skills_summary(),
            participants=participants.values(),
            include_skills=False,
        )
        try:
            classification = await classifier.classify(
                transcript_text=content,
                participants=participants,
                system_prompt=classifier_prompt,
                model=DEFAULT_MODEL,
            )
            logger.info(
                "Transcript classified as %s for %s: %s",
                classification.label,
                classification.speaker_label,
                classification.summary,
            )
            await self._dispatch_classification(
                classification=classification,
                content=content,
                transcript_path=transcript_path,
                task_queue=task_queue,
                notification_store=notification_store,
                proactive_topic_detector=proactive_topic_detector,
                memory_store=memory_store,
                participants=participants,
                simple_executor=simple_executor,
                executor_prompt=executor_prompt,
            )
        finally:
            llm_service.reset_context()

    async def _dispatch_classification(
        self,
        classification: TranscriptClassificationResult,
        content: str,
        transcript_path: str,
        task_queue: SQLiteTaskQueueAdapter,
        notification_store: SQLiteNotificationAdapter,
        proactive_topic_detector: ProactiveTopicDetectionService,
        memory_store: SQLiteMemoryAdapter,
        participants,
        simple_executor: SimpleTaskExecutionService,
        executor_prompt: str,
    ) -> None:
        if classification.label == "NOTHING":
            return

        if classification.label in {"FACT", "PREFERENCE"}:
            participant = participants.get(classification.speaker_label)
            if participant is None:
                return
            memory_text = classification.memory_content or classification.summary
            memory_store.append_event(
                MemoryEvent(
                    event_id=uuid.uuid4().hex,
                    speaker_id=participant.speaker_id,
                    source_type="transcript",
                    source_ref=transcript_path,
                    event_kind=classification.label.lower(),
                    content=memory_text,
                    confidence=classification.confidence,
                    status="candidate",
                    created_at=datetime.now().isoformat(),
                )
            )
            topic = proactive_topic_detector.maybe_queue_topic(
                classification=classification,
                transcript_text=content,
                source_ref=transcript_path,
            )
            if topic is not None:
                notification_store.add_notification(
                    f"Queued proactive research topic: {topic.display_title}",
                    source="topic_detector",
                )
            return

        if classification.label == "TASK_COMPLEX":
            topic = proactive_topic_detector.maybe_queue_topic(
                classification=classification,
                transcript_text=content,
                source_ref=transcript_path,
            )
            queue_payload = (
                f"Transcript source: {transcript_path}\n"
                f"Speaker: {classification.speaker_label}\n"
                f"Summary: {classification.summary}\n"
                f"Suggested action: {classification.suggested_action or classification.summary}\n"
                f"Transcript content:\n{content}"
            )
            task_queue.add_task(queue_payload, priority="medium")
            notification_store.add_notification(
                (
                    f"Queued complex ambient task: {classification.summary}"
                    + (f" | proactive topic: {topic.display_title}" if topic is not None else "")
                ),
                source="classifier",
            )
            return

        if classification.label in {"REMINDER", "TASK_SIMPLE"}:
            try:
                execution_result = await simple_executor.execute(
                    classification=classification,
                    transcript_text=content,
                    system_prompt=executor_prompt,
                    model=DEFAULT_MODEL,
                )
                logger.info("Simple executor result: %s", execution_result)
            except Exception as exc:
                logger.exception("Simple executor failed for %s", classification.summary)
                notification_store.add_notification(
                    f"Simple execution failed for '{classification.summary}': {exc}",
                    source="simple_executor",
                )
                fallback_payload = (
                    f"Fallback queued from simple executor failure.\n"
                    f"Classification: {classification.label}\n"
                    f"Summary: {classification.summary}\n"
                    f"Transcript source: {transcript_path}"
                )
                task_queue.add_task(fallback_payload, priority="medium")
            return

    async def run_loop(self):
        (
            llm_adapter,
            tool_bridge,
            llm_service,
            task_queue,
            notification_store,
            agenda_store,
            proactive_topic_queue,
            memory_store,
            research_vault,
            memory_context_builder,
            speaker_resolution,
            classifier,
            evidence_service,
            session_tracker,
            open_loop_service,
            user_profile_service,
            simple_executor,
            proactive_topic_detector,
            proactive_research_service,
            memory_consolidation,
            ambient_reflection,
        ) = self._build_services()
        proactive_research_prompt = memory_context_builder.build_prompt(
            base_prompt_filename=PROACTIVE_RESEARCH_PROMPT,
            skills_summary=build_available_skills_summary(),
            participants=[],
            include_skills=False,
        )
        idle_cycle_interval = 30
        last_idle_cycle_at = 0.0
        services_initialized = False

        try:
            logger.info("Starting ambient LLM in continuous idle mode.")
            await self._ensure_llm_ready(llm_adapter, tool_bridge, llm_service)
            services_initialized = True

            while True:
                if self.audio_active_event.is_set():
                    logger.info("ASR claimed GPU. Unloading ambient LLM until audio processing completes.")
                    if services_initialized:
                        await self._release_llm(llm_adapter, tool_bridge)
                        services_initialized = False
                    while self.audio_active_event.is_set():
                        await asyncio.sleep(0.5)
                    logger.info("ASR finished. Reloading ambient LLM.")
                    await self._ensure_llm_ready(llm_adapter, tool_bridge, llm_service)
                    services_initialized = True
                    last_idle_cycle_at = 0.0

                processed_transcript = False
                while True:
                    try:
                        file_path = self.queue.get_nowait()
                    except queue.Empty:
                        break

                    processed_transcript = True
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
                                task_queue=task_queue,
                                notification_store=notification_store,
                                proactive_topic_detector=proactive_topic_detector,
                                memory_store=memory_store,
                                memory_context_builder=memory_context_builder,
                                speaker_resolution=speaker_resolution,
                                classifier=classifier,
                                evidence_service=evidence_service,
                                session_tracker=session_tracker,
                                open_loop_service=open_loop_service,
                                user_profile_service=user_profile_service,
                                simple_executor=simple_executor,
                                content=content,
                                transcript_path=file_path,
                            )
                    except KeyboardInterrupt:
                        logger.info("TranscriptionService shutting down.")
                        return
                    except Exception:
                        logger.exception("Failed to process transcript: %s", file_path)
                        await asyncio.sleep(5)
                    finally:
                        self.queue.task_done()

                now = time.monotonic()
                if not processed_transcript and now - last_idle_cycle_at >= idle_cycle_interval:
                    logger.info("Running ambient idle cycle.")
                    try:
                        with self.gpu_lock:
                            await self._run_idle_cycle(
                                memory_consolidation=memory_consolidation,
                                proactive_research_service=proactive_research_service,
                                ambient_reflection=ambient_reflection,
                                memory_store=memory_store,
                                proactive_research_prompt=proactive_research_prompt,
                            )
                    except Exception:
                        logger.exception("Ambient idle cycle failed.")
                    last_idle_cycle_at = now

                await asyncio.sleep(1)
        finally:
            if services_initialized:
                consolidated = memory_consolidation.consolidate()
                logger.info("Memory consolidation processed %s events during shutdown.", consolidated)
                await self._release_llm(llm_adapter, tool_bridge)

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
