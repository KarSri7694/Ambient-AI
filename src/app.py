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
from application.services.activity_ledger_service import ActivityLedgerService
from application.services.ambient_reflection_service import AmbientReflectionService
from application.services.agenda_scoring_service import AgendaScoringService
from application.services.memory_consolidation_service import MemoryConsolidationService
from application.services.memory_context_builder import MemoryContextBuilder
from application.services.night_mode_service import NightModeService
from application.services.proactive_research_service import ProactiveResearchService
from application.services.proactive_topic_detection_service import ProactiveTopicDetectionService
from application.services.research_vault_service import ResearchVaultService
from application.services.session_tracker_service import SessionTrackerService
from application.services.simple_task_execution_service import SimpleTaskExecutionService
from application.services.speaker_resolution_service import SpeakerResolutionService
from application.services.transcript_evidence_service import TranscriptEvidenceService
from application.services.transcript_classification_service import TranscriptClassificationService
from application.services.transcript_normalization_service import TranscriptNormalizationService
from application.services.open_loop_service import OpenLoopService
from application.services.passive_observer_followup_service import PassiveObserverFollowupService
from application.services.passive_observer_service import PassiveObserverService
from application.services.screenshot_queue_service import ScreenshotQueueService
from application.services.system_idle_service import SystemIdleService
from application.services.user_profile_service import UserProfileService
from application.services.visual_user_fact_service import VisualUserFactService
from audio_agent import AudioAgentService
from infrastructure.adapter.llamaCppAdapter import LlamaCppAdapter
from infrastructure.adapter.LoggingLLMProvider import LoggingLLMProvider
from infrastructure.adapter.MCPToolAdapter import MCPToolAdapter
from infrastructure.adapter.MSSScreenCaptureAdapter import MssScreenCaptureAdapter
from infrastructure.adapter.SQLiteMemoryAdapter import SQLiteMemoryAdapter
from infrastructure.adapter.SQLiteAmbientAgendaAdapter import SQLiteAmbientAgendaAdapter
from infrastructure.adapter.SQLiteActivityLedgerAdapter import SQLiteActivityLedgerAdapter
from infrastructure.adapter.SQLiteInteractionLogAdapter import SQLiteInteractionLogAdapter
from infrastructure.adapter.SQLiteNotificationAdapter import SQLiteNotificationAdapter
from infrastructure.adapter.SQLiteProactiveTopicQueueAdapter import SQLiteProactiveTopicQueueAdapter
from infrastructure.adapter.SQLiteTaskQueueAdapter import SQLiteTaskQueueAdapter
from infrastructure.adapter.TodoistTaskAdapter import TodoistTaskAdapter
from database_bootstrap import ensure_runtime_databases
from core.models import MemoryEvent, TranscriptClassificationResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8080"
DEFAULT_MODEL = "Qwen-3.5-9B-Fable-Distilled-Q4_K_M-Vision"
MCP_CONFIG_PATH = "mcp.json"
PROJECT_ROOT = Path(__file__).parent.parent
PROMPTS_ROOT = PROJECT_ROOT / "prompts"
USER_DATA_DIR = Path(os.getenv("USER_DATA_DIR", "D:\\USER_DATA"))
MEMORY_ROOT = USER_DATA_DIR / "memory"
MEMORY_DB_PATH = PROJECT_ROOT / "database" / "memory.db"
PROACTIVE_TOPICS_DB_PATH = PROJECT_ROOT / "database" / "proactive_topics.db"
AMBIENT_AGENDA_DB_PATH = PROJECT_ROOT / "database" / "ambient_agenda.db"
INTERACTION_LOG_DB_PATH = PROJECT_ROOT / "database" / "interaction_logs.db"
ACTIVITY_LEDGER_DB_PATH = PROJECT_ROOT / "database" / "activity_ledger.db"
CURRENT_RESPONSE_PATH = PROJECT_ROOT / "database" / "current_llm_response.md"
VOICE_DB_PATH = PROJECT_ROOT / "database" / "voice_database.db"
FINANCE_DB_PATH = PROJECT_ROOT / "database" / "finance.db"
FACTS_DB_PATH = PROJECT_ROOT / "database" / "facts.db"
RESEARCH_VAULT_ROOT = USER_DATA_DIR / "research_vault"
PASSIVE_OBSERVER_ROOT = USER_DATA_DIR / "passive_observer"
CLASSIFIER_PROMPT = "TRANSCRIPT_CLASSIFIER.md"
SIMPLE_EXECUTOR_PROMPT = "SIMPLE_EXECUTOR.md"
PROACTIVE_RESEARCH_PROMPT = "PROACTIVE_RESEARCH.md"
PASSIVE_OBSERVER_ENABLED = os.getenv("PASSIVE_OBSERVER_ENABLED", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
USER_IDLE_THRESHOLD_SECONDS = 120
SCREENSHOT_QUEUE_MAXLEN = 180
PASSIVE_OBSERVER_SSIM_THRESHOLD = float(os.getenv("PASSIVE_OBSERVER_SSIM_THRESHOLD", "0.92"))
PASSIVE_OBSERVER_SSIM_COMPARE_COUNT = int(os.getenv("PASSIVE_OBSERVER_SSIM_COMPARE_COUNT", "4"))
MAX_SCREENSHOTS_PER_IDLE_CYCLE = 6
NIGHT_MODE_START_HOUR = 0
NIGHT_MODE_END_HOUR = 6


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

    async def _ensure_ambient_runtime(
        self,
        *,
        llm_adapter,
        tool_bridge,
        llm_service,
        services_initialized: bool,
        reason: str,
    ) -> bool:
        if services_initialized:
            return True
        logger.info("Loading ambient runtime: %s", reason)
        await self._ensure_llm_ready(llm_adapter, tool_bridge, llm_service)
        return True

    async def _release_ambient_runtime(
        self,
        *,
        llm_adapter,
        tool_bridge,
        services_initialized: bool,
        reason: str,
    ) -> bool:
        if not services_initialized:
            return False
        logger.info("Unloading ambient runtime: %s", reason)
        await self._release_llm(llm_adapter, tool_bridge)
        return False

    async def _run_idle_cycle(
        self,
        *,
        memory_consolidation: MemoryConsolidationService,
        proactive_research_service: ProactiveResearchService,
        ambient_reflection: AmbientReflectionService,
        passive_followup: PassiveObserverFollowupService | None,
        visual_user_fact_service: VisualUserFactService | None,
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
        if passive_followup is not None:
            followup_result = await passive_followup.maybe_queue_followup(model=DEFAULT_MODEL)
            if followup_result.get("action") == "queue_task":
                logger.info(
                    "Passive observer queued deferred task: %s",
                    followup_result.get("title", ""),
                )
            else:
                logger.info("Passive observer idle follow-up found no task to queue.")
        else:
            reflection_result = await ambient_reflection.reflect_with_metadata(
                model=DEFAULT_MODEL,
                recent_context=memory_store.get_recent_context(),
            )
            if not reflection_result.llm_invoked:
                logger.info(
                    "Ambient reflection skipped: no candidates (candidate_count=%s).",
                    reflection_result.candidate_count,
                )
            else:
                logger.info(
                    "Ambient reflection invoked with %s candidate(s) and produced %s action(s): %s",
                    reflection_result.candidate_count,
                    len(reflection_result.actions),
                    ", ".join(action.action_type for action in reflection_result.actions),
                )

    def _is_night_mode_window(self, current_time: datetime | None = None) -> bool:
        now = current_time or datetime.now()
        return NIGHT_MODE_START_HOUR <= now.hour < NIGHT_MODE_END_HOUR

    def _build_services(self):
        ensure_runtime_databases(
            memory_db_path=str(MEMORY_DB_PATH),
            memory_root=str(MEMORY_ROOT),
            ambient_agenda_db_path=str(AMBIENT_AGENDA_DB_PATH),
            interaction_log_db_path=str(INTERACTION_LOG_DB_PATH),
            proactive_topics_db_path=str(PROACTIVE_TOPICS_DB_PATH),
            activity_ledger_db_path=str(ACTIVITY_LEDGER_DB_PATH),
            voice_db_path=str(VOICE_DB_PATH),
            finance_db_path=str(FINANCE_DB_PATH),
            facts_db_path=str(FACTS_DB_PATH),
        )
        llm_adapter = LlamaCppAdapter(base_url=API_BASE_URL)
        interaction_log_store = SQLiteInteractionLogAdapter(
            db_path=str(INTERACTION_LOG_DB_PATH),
        )
        activity_ledger_store = SQLiteActivityLedgerAdapter(
            db_path=str(ACTIVITY_LEDGER_DB_PATH),
            interaction_log_db_path=str(INTERACTION_LOG_DB_PATH),
        )
        activity_ledger = ActivityLedgerService(activity_ledger_store)
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
        todoist_task_provider = TodoistTaskAdapter()
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
        transcript_normalizer = TranscriptNormalizationService(llm_provider=logged_llm)
        speaker_resolution = SpeakerResolutionService(memory=memory_store)
        classifier = TranscriptClassificationService(llm_provider=logged_llm)
        evidence_service = TranscriptEvidenceService(llm_provider=logged_llm)
        session_tracker = SessionTrackerService(memory=memory_store, llm_provider=logged_llm)
        open_loop_service = OpenLoopService(memory=memory_store, llm_provider=logged_llm)
        user_profile_service = UserProfileService(memory=memory_store, llm_provider=logged_llm)
        simple_executor = SimpleTaskExecutionService(
            llm_service=llm_service,
            activity_ledger=activity_ledger,
        )
        proactive_topic_detector = ProactiveTopicDetectionService(
            queue=proactive_topic_queue,
            vault=research_vault,
        )
        proactive_research_service = ProactiveResearchService(
            llm_service=llm_service,
            topic_queue=proactive_topic_queue,
            vault=research_vault,
            notifications=notification_store,
            activity_ledger=activity_ledger,
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
        night_mode_service = NightModeService(
            task_queue=task_queue,
            task_provider=todoist_task_provider,
            notification_port=notification_store,
            llm_service=llm_service,
            memory_consolidator=memory_consolidation,
            proactive_research_service=proactive_research_service,
            ambient_reflection_service=ambient_reflection,
            activity_ledger=activity_ledger,
            model=DEFAULT_MODEL,
        )
        passive_followup = (
            PassiveObserverFollowupService(
                memory=memory_store,
                task_queue=task_queue,
                llm_provider=logged_llm,
                activity_ledger=activity_ledger,
            )
            if PASSIVE_OBSERVER_ENABLED
            else None
        )
        visual_user_fact_service = (
            VisualUserFactService(memory=memory_store)
            if PASSIVE_OBSERVER_ENABLED
            else None
        )
        system_idle_service = SystemIdleService(
            idle_threshold_seconds=USER_IDLE_THRESHOLD_SECONDS,
        )
        screenshot_queue = (
            ScreenshotQueueService(
                maxlen=SCREENSHOT_QUEUE_MAXLEN,
                ssim_threshold=PASSIVE_OBSERVER_SSIM_THRESHOLD,
                ssim_compare_count=PASSIVE_OBSERVER_SSIM_COMPARE_COUNT,
            )
            if PASSIVE_OBSERVER_ENABLED
            else None
        )
        passive_observer = (
            PassiveObserverService(
                memory=memory_store,
                llm_provider=logged_llm,
                screen_capture=MssScreenCaptureAdapter(output_dir=str(PASSIVE_OBSERVER_ROOT / "screenshots")),
                screenshot_root=str(PASSIVE_OBSERVER_ROOT / "screenshots"),
            )
            if PASSIVE_OBSERVER_ENABLED
            else None
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
            transcript_normalizer,
            speaker_resolution,
            classifier,
            evidence_service,
            session_tracker,
            open_loop_service,
            user_profile_service,
            simple_executor,
            proactive_topic_detector,
            proactive_research_service,
            activity_ledger,
            memory_consolidation,
            ambient_reflection,
            night_mode_service,
            passive_followup,
            visual_user_fact_service,
            system_idle_service,
            screenshot_queue,
            passive_observer,
        )

    async def _process_one(
        self,
        llm_service: LLMInteractionService,
        task_queue: SQLiteTaskQueueAdapter,
        notification_store: SQLiteNotificationAdapter,
        proactive_topic_detector: ProactiveTopicDetectionService,
        memory_store: SQLiteMemoryAdapter,
        memory_context_builder: MemoryContextBuilder,
        transcript_normalizer: TranscriptNormalizationService,
        speaker_resolution: SpeakerResolutionService,
        classifier: TranscriptClassificationService,
        evidence_service: TranscriptEvidenceService,
        session_tracker: SessionTrackerService,
        open_loop_service: OpenLoopService,
        user_profile_service: UserProfileService,
        simple_executor: SimpleTaskExecutionService,
        activity_ledger: ActivityLedgerService,
        content: str,
        transcript_path: str,
    ):
        normalized_content = await transcript_normalizer.normalize(
            content,
            model=DEFAULT_MODEL,
        )
        if normalized_content.strip() and normalized_content != content:
            logger.info("Transcript normalized for %s before downstream processing.", transcript_path)
            content = normalized_content
            Path(transcript_path).write_text(content, encoding="utf-8")
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
        durable_participants = {
            participant.speaker_label: participant
            for participant in participants.values()
            if participant.durable
        }
        durable_evidence_items = [
            item for item in evidence_items
            if participants.get(item.speaker_label) is not None and participants[item.speaker_label].durable
        ]
        for item in durable_evidence_items:
            memory_store.append_evidence(item)
        session_tracker.refresh_digest()
        open_loops = await open_loop_service.process(
            session=session,
            evidence_items=durable_evidence_items,
            model=DEFAULT_MODEL,
        )
        if open_loops:
            logger.info("Updated %s open loop(s) from transcript state.", len(open_loops))
        user_speaker = user_profile_service.infer_user_speaker()
        facets = await user_profile_service.update_from_evidence(
            user_speaker=user_speaker,
            evidence_items=durable_evidence_items,
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
                participants=durable_participants,
                simple_executor=simple_executor,
                activity_ledger=activity_ledger,
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
        activity_ledger: ActivityLedgerService,
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
            run = activity_ledger.queue_run(
                source_kind="transcript_task",
                trigger_kind="transcript",
                title=classification.summary[:120],
                summary=classification.summary,
                output_text="Queued for later execution.",
                metadata={
                    "classification_label": classification.label,
                    "speaker_label": classification.speaker_label,
                    "transcript_path": transcript_path,
                },
                tags=["transcript", "complex_task", "queued_task"],
            )
            activity_ledger.link_entity(
                run_id=run.run_id,
                entity_type="transcript",
                entity_id=transcript_path,
                relation="derived_from",
            )
            activity_ledger.attach_artifact(
                run_id=run.run_id,
                artifact_kind="transcript",
                title="Transcript source",
                path=transcript_path,
                mime_type="text/plain",
                text_preview=content[:400],
            )
            if topic is not None:
                activity_ledger.link_entity(
                    run_id=run.run_id,
                    entity_type="proactive_topic",
                    entity_id=topic.topic_id,
                    relation="queued_related_topic",
                )
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
            transcript_normalizer,
            speaker_resolution,
            classifier,
            evidence_service,
            session_tracker,
            open_loop_service,
            user_profile_service,
            simple_executor,
            proactive_topic_detector,
            proactive_research_service,
            activity_ledger,
            memory_consolidation,
            ambient_reflection,
            night_mode_service,
            passive_followup,
            visual_user_fact_service,
            system_idle_service,
            screenshot_queue,
            passive_observer,
        ) = self._build_services()
        proactive_research_prompt = memory_context_builder.build_prompt(
            base_prompt_filename=PROACTIVE_RESEARCH_PROMPT,
            skills_summary=build_available_skills_summary(),
            participants=[],
            include_skills=False,
        )
        idle_cycle_interval = 30
        passive_observer_interval = 10
        last_idle_cycle_at = 0.0
        last_passive_observer_at = 0.0
        user_idle_now = False
        services_initialized = False

        try:
            logger.info("Starting ambient runtime manager.")

            while True:
                current_user_idle = system_idle_service.is_user_idle() if PASSIVE_OBSERVER_ENABLED else False
                if current_user_idle != user_idle_now:
                    user_idle_now = current_user_idle
                    if user_idle_now:
                        logger.info("User idle detected (>= %ss). Ambient idle mode enabled.", USER_IDLE_THRESHOLD_SECONDS)
                        last_idle_cycle_at = 0.0
                    else:
                        logger.info("User activity detected. Ambient work will stay queued until idle resumes.")

                if self.audio_active_event.is_set():
                    logger.info("ASR claimed GPU. Unloading ambient runtime until audio processing completes.")
                    services_initialized = await self._release_ambient_runtime(
                        llm_adapter=llm_adapter,
                        tool_bridge=tool_bridge,
                        services_initialized=services_initialized,
                        reason="ASR pipeline is using the GPU",
                    )
                    while self.audio_active_event.is_set():
                        await asyncio.sleep(0.5)
                    last_idle_cycle_at = 0.0

                processed_transcript = False
                if user_idle_now:
                    try:
                        services_initialized = await self._ensure_ambient_runtime(
                            llm_adapter=llm_adapter,
                            tool_bridge=tool_bridge,
                            llm_service=llm_service,
                            services_initialized=services_initialized,
                            reason="processing explicit Ambient AI Tasks",
                        )
                        with self.gpu_lock:
                            processed_explicit_tasks = await night_mode_service.run_external_task_cycle()
                        if processed_explicit_tasks:
                            logger.info(
                                "Processed %s explicit Ambient AI Tasks before transcript drain.",
                                processed_explicit_tasks,
                            )
                    except Exception:
                        logger.exception("Explicit Ambient AI Tasks cycle failed.")

                while user_idle_now:
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
                        services_initialized = await self._ensure_ambient_runtime(
                            llm_adapter=llm_adapter,
                            tool_bridge=tool_bridge,
                            llm_service=llm_service,
                            services_initialized=services_initialized,
                            reason=f"processing transcript {file_path}",
                        )
                        with self.gpu_lock:
                            await self._process_one(
                                llm_service=llm_service,
                                task_queue=task_queue,
                                notification_store=notification_store,
                                proactive_topic_detector=proactive_topic_detector,
                                memory_store=memory_store,
                                memory_context_builder=memory_context_builder,
                                transcript_normalizer=transcript_normalizer,
                                speaker_resolution=speaker_resolution,
                                classifier=classifier,
                                evidence_service=evidence_service,
                                session_tracker=session_tracker,
                                open_loop_service=open_loop_service,
                                user_profile_service=user_profile_service,
                                simple_executor=simple_executor,
                                activity_ledger=activity_ledger,
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

                if (
                    PASSIVE_OBSERVER_ENABLED
                    and passive_observer is not None
                    and screenshot_queue is not None
                    and now - last_passive_observer_at >= passive_observer_interval
                ):
                    try:
                        screenshot_path = passive_observer.capture_screenshot()
                        queued = screenshot_queue.enqueue(screenshot_path)
                        if queued is None:
                            logger.info("Skipped screenshot for passive observer due to SSIM similarity filter: %s (queue_size=%s)", screenshot_path, screenshot_queue.size())
                        else:
                            logger.info("Queued screenshot for passive observer: %s (queue_size=%s)", queued.screenshot_path, screenshot_queue.size())
                    except Exception:
                        logger.exception("Passive observer screenshot capture failed.")
                    last_passive_observer_at = now

                if (
                    PASSIVE_OBSERVER_ENABLED
                    and passive_observer is not None
                    and screenshot_queue is not None
                    and user_idle_now
                    and not screenshot_queue.is_empty()
                ):
                    processed_screenshots = 0
                    while (
                        processed_screenshots < MAX_SCREENSHOTS_PER_IDLE_CYCLE
                        and not screenshot_queue.is_empty()
                    ):
                        if not system_idle_service.is_user_idle():
                            user_idle_now = False
                            logger.info("User activity resumed during screenshot backlog processing. Pausing idle mode.")
                            break
                        job = screenshot_queue.dequeue()
                        if job is None:
                            break
                        try:
                            if not Path(job.screenshot_path).exists():
                                logger.warning("Queued screenshot no longer exists, skipping: %s", job.screenshot_path)
                                processed_screenshots += 1
                                continue
                            services_initialized = await self._ensure_ambient_runtime(
                                llm_adapter=llm_adapter,
                                tool_bridge=tool_bridge,
                                llm_service=llm_service,
                                services_initialized=services_initialized,
                                reason="processing queued passive-observer screenshots",
                            )
                            with self.gpu_lock:
                                observation = await passive_observer.process_screenshot(
                                    screenshot_path=job.screenshot_path,
                                    model=DEFAULT_MODEL,
                                    recent_context=memory_store.get_recent_context(),
                                    captured_at=job.captured_at,
                                )
                            if observation is not None:
                                if visual_user_fact_service is not None:
                                    updated_facts = visual_user_fact_service.update_from_observation(observation)
                                    if updated_facts:
                                        logger.info(
                                            "Updated %s visual user fact(s) from passive observation.",
                                            len(updated_facts),
                                        )
                                logger.info(
                                    "Processed queued screenshot for %s.",
                                    observation.app_name or observation.page_hint or "screen",
                                )
                        except Exception:
                            logger.exception("Queued passive observer screenshot processing failed.")
                        finally:
                            try:
                                Path(job.screenshot_path).unlink(missing_ok=True)
                            except OSError:
                                logger.debug("Failed to remove processed screenshot %s", job.screenshot_path)
                        processed_screenshots += 1

                if user_idle_now and now - last_idle_cycle_at >= idle_cycle_interval:
                    try:
                        services_initialized = await self._ensure_ambient_runtime(
                            llm_adapter=llm_adapter,
                            tool_bridge=tool_bridge,
                            llm_service=llm_service,
                            services_initialized=services_initialized,
                            reason="running ambient idle work",
                        )
                        with self.gpu_lock:
                            if self._is_night_mode_window():
                                logger.info("Night mode window active during idle. Running night mode cycle.")
                                cycle_result = await night_mode_service.run_night_cycle()
                                logger.info("Night mode cycle result: %s", cycle_result)
                            else:
                                logger.info("Running ambient idle cycle.")
                                await self._run_idle_cycle(
                                    memory_consolidation=memory_consolidation,
                                    proactive_research_service=proactive_research_service,
                                    ambient_reflection=ambient_reflection,
                                    passive_followup=passive_followup,
                                    visual_user_fact_service=visual_user_fact_service,
                                    memory_store=memory_store,
                                    proactive_research_prompt=proactive_research_prompt,
                                )
                    except Exception:
                        logger.exception("Idle cycle failed.")
                    last_idle_cycle_at = now

                if not user_idle_now:
                    services_initialized = await self._release_ambient_runtime(
                        llm_adapter=llm_adapter,
                        tool_bridge=tool_bridge,
                        services_initialized=services_initialized,
                        reason="user is active; defer audio/transcript/ambient work and keep VRAM clear",
                    )

                await asyncio.sleep(1)
        finally:
            if services_initialized:
                consolidated = memory_consolidation.consolidate()
                logger.info("Memory consolidation processed %s events during shutdown.", consolidated)
                await self._release_ambient_runtime(
                    llm_adapter=llm_adapter,
                    tool_bridge=tool_bridge,
                    services_initialized=services_initialized,
                    reason="application shutdown",
                )

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
