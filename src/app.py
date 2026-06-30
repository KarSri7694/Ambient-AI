import asyncio
import gc
import json
import logging
from pathlib import Path
import queue
import re
import threading
import time
from typing import Optional

from application.services.passive_observer_followup_service import PassiveObserverFollowupService
from application.services.reflection_service import ReflectionService
from application.services.llm_interaction_service import LLMInteractionService
from application.services.passive_observer_service import PassiveObserverService
from application.services.semantic_deduplication_service import SemanticDeduplicationService
from application.services.semantic_memory_service import SemanticMemoryService
from application.services.screenshot_queue_service import ScreenshotQueueService
from application.services.system_idle_service import SystemIdleService
from application.services.training_data_service import TrainingDataService
from application.services.user_bio_data_service import UserBioDataService
from audio_agent import AudioAgentService
from infrastructure.adapter.LlamaCppSemanticAdapter import LlamaCppSemanticAdapter
from infrastructure.adapter.llamaCppAdapter import LlamaCppAdapter
from infrastructure.adapter.LoggingLLMProvider import LoggingLLMProvider
from infrastructure.adapter.MCPToolAdapter import MCPToolAdapter
from infrastructure.adapter.MSSScreenCaptureAdapter import MssScreenCaptureAdapter
from infrastructure.adapter.SQLiteBenchmarkAdapter import SQLiteBenchmarkAdapter
from infrastructure.adapter.SQLiteInteractionLogAdapter import SQLiteInteractionLogAdapter
from infrastructure.adapter.SQLiteMemoryAdapter import SQLiteMemoryAdapter
from infrastructure.adapter.SQLiteTaskQueueAdapter import SQLiteTaskQueueAdapter
from infrastructure.adapter.SQLiteTrainingDataAdapter import SQLiteTrainingDataAdapter
from infrastructure.adapter.SQLiteVoiceAdapter import SQLiteVoiceAdapter
from infrastructure.adapter.TodoistTaskAdapter import TodoistTaskAdapter
from infrastructure.adapter.UIATAdapter import UIATAdapter
from infrastructure.runtime_log_server import (
    configure_runtime_log_streaming,
    shutdown_runtime_log_server,
    start_runtime_log_server,
)
from config import CONFIG
import night_mode
from utils.todoist_helper import TodoistHelper


DEBUG_MODE = CONFIG.get_bool("runtime", "debug_mode", False)

logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

API_BASE_URL = CONFIG.get_str("runtime", "api_base_url", "http://localhost:8080")
API_KEY = CONFIG.get_str("runtime", "api_key", "testkey")
DEFAULT_MODEL = CONFIG.get_str("runtime", "default_model", "Qwen-3.5-9B-Mythos-Distilled-Q4_K_M-Vision")
PASSIVE_OBSERVER_MODEL = CONFIG.get_model("passive_observer_model", DEFAULT_MODEL)
FULL_PASSIVE_OBSERVER_MODEL = CONFIG.get_model("full_passive_observer_model", DEFAULT_MODEL)
PASSIVE_FOLLOWUP_MODEL = CONFIG.get_model("passive_followup_model", DEFAULT_MODEL)
USER_BIODATA_MODEL = CONFIG.get_model("user_biodata_model", DEFAULT_MODEL)
FOLLOWUP_EXECUTION_MODEL = CONFIG.get_model("followup_execution_model", DEFAULT_MODEL)
REPORTER_MODEL = CONFIG.get_model("reporter_model", DEFAULT_MODEL)
TODOIST_EXECUTION_MODEL = CONFIG.get_model("todoist_execution_model", DEFAULT_MODEL)
REFLECTION_MODEL = CONFIG.get_model("reflection_model", DEFAULT_MODEL)
TRANSCRIPT_PROCESSING_MODEL = CONFIG.get_model("transcript_processing_model", FOLLOWUP_EXECUTION_MODEL)
USER_DATA_DIR = Path(CONFIG.get_str("runtime", "user_data_dir", "D:\\USER_DATA"))
PROJECT_ROOT = Path(__file__).parent.parent
MEMORY_ROOT = USER_DATA_DIR / "memory"
MEMORY_DB_PATH = USER_DATA_DIR / "database" / "memory.db"
INTERACTION_LOG_DB_PATH = USER_DATA_DIR / "database" / "interaction_logs.db"
BENCHMARK_DB_PATH = Path(CONFIG.get_str("benchmarking", "db_path", str(PROJECT_ROOT / "database" / "benchmarking.db")))
TRAINING_DATA_ROOT = Path(CONFIG.get_str("training_data", "root", "D:\\TRAINING_DATA"))
TRAINING_DATA_DB_PATH = Path(
    CONFIG.get_str("training_data", "db_path", str(TRAINING_DATA_ROOT / "database" / "training_data.db"))
)
CURRENT_RESPONSE_PATH = PROJECT_ROOT / "database" / "current_llm_response.md"
ARTIFACTS_ROOT = USER_DATA_DIR / "artifacts"
VOICE_DB_PATH = USER_DATA_DIR / "database" / "voice_database.db"
PASSIVE_OBSERVER_ROOT = USER_DATA_DIR / "passive_observer"
PASSIVE_OBSERVER_ENABLED = CONFIG.get_bool("passive_observer", "enabled", False)
USER_IDLE_THRESHOLD_SECONDS = CONFIG.get_int("runtime", "user_idle_threshold_seconds", 20)
ALWAYS_ON_MODE = CONFIG.get_bool("runtime", "always_on", False)
PERFORM_QUEUE_TASKS = CONFIG.get_bool("runtime", "perform_queue_tasks", False)
SCREENSHOT_QUEUE_MAXLEN = CONFIG.get_int("passive_observer", "screenshot_queue_maxlen", 180)
PASSIVE_OBSERVER_SSIM_THRESHOLD = CONFIG.get_float("passive_observer", "ssim_threshold", 0.92)
PASSIVE_OBSERVER_SSIM_COMPARE_COUNT = CONFIG.get_int("passive_observer", "ssim_compare_count", 4)
MAX_SCREENSHOTS_PER_IDLE_CYCLE = CONFIG.get_int("passive_observer", "max_screenshots_per_idle_cycle", 6)
PASSIVE_OBSERVER_FAST_ROUTING_ENABLED = CONFIG.get_bool("passive_observer", "fast_routing_enabled", True)
PASSIVE_OBSERVER_FULL_VLM_SSIM_THRESHOLD = CONFIG.get_float("passive_observer", "full_vlm_ssim_threshold", 0.70)
PASSIVE_OBSERVER_FAST_MODEL_RETRY_COUNT = CONFIG.get_int("passive_observer", "fast_model_retry_count", 2)
PASSIVE_OBSERVER_UIAT_MODE = CONFIG.get_str("passive_observer", "uiat_mode", "screen_content")
LOG_API_ENABLED = CONFIG.get_bool("log_api", "enabled", True)
LOG_API_HOST = CONFIG.get_str("log_api", "host", "0.0.0.0")
LOG_API_PORT = CONFIG.get_int("log_api", "port", 8765)
LOG_API_BUFFER_SIZE = CONFIG.get_int("log_api", "buffer_size", 2000)
MCP_CONFIG_PATH = CONFIG.get_str("runtime", "mcp_config_path", "mcp.json")
TODOIST_ENABLED = CONFIG.get_bool("todoist", "enabled", True)
UPLOADS_DIR = Path(CONFIG.get_str("audio", "uploads_dir", str(USER_DATA_DIR / "uploads")))
TRANSCRIPTIONS_DIR = Path(CONFIG.get_str("audio", "transcriptions_dir", str(USER_DATA_DIR / "transcriptions")))
CLEANED_AUDIO_DIR = Path(CONFIG.get_str("audio", "cleaned_audio_dir", str(USER_DATA_DIR / "cleaned_audio")))
REFLECTION_ENABLED = CONFIG.get_bool("reflection", "enabled", True)
REFLECTION_CADENCE_MODE = CONFIG.get_str("reflection", "cadence_mode", "daily")
REFLECTION_INTERVAL_HOURS = CONFIG.get_int("reflection", "interval_hours", 24)
REFLECTION_MAX_GENERATED_TASKS = CONFIG.get_int("reflection", "max_generated_tasks", 8)
REFLECTION_HISTORY_PATH = CONFIG.get_str(
    "reflection",
    "history_path",
    str(USER_DATA_DIR / "reflection" / "reflection_history.json"),
)
SEMANTIC_MEMORY_ENABLED = CONFIG.get_bool("semantic_memory", "enabled", True)
EMBEDDING_API_BASE_URL = CONFIG.get_str("semantic_memory", "embedding_api_base_url", "http://localhost:8081")
EMBEDDING_MODEL = CONFIG.get_model("embedding_model", "", section="semantic_memory")
RERANKER_API_BASE_URL = CONFIG.get_str("semantic_memory", "reranker_api_base_url", EMBEDDING_API_BASE_URL)
RERANKER_MODEL = CONFIG.get_model("reranker_model", "", section="semantic_memory")
SEMANTIC_VECTOR_LIMIT = CONFIG.get_int("semantic_memory", "vector_limit", 12)
SEMANTIC_RERANK_LIMIT = CONFIG.get_int("semantic_memory", "rerank_limit", 6)
SEMANTIC_SYNC_BATCH_SIZE = CONFIG.get_int("semantic_memory", "sync_batch_size", 32)
SEMANTIC_DEDUPE_ENABLED = CONFIG.get_bool("semantic_dedupe", "enabled", True)
SEMANTIC_DEDUPE_MODEL = CONFIG.get_model("model", DEFAULT_MODEL, section="semantic_dedupe")
SEMANTIC_DEDUPE_CANDIDATE_LIMIT = CONFIG.get_int("semantic_dedupe", "candidate_limit", 8)
SEMANTIC_DEDUPE_DEFAULT_TTL_SECONDS = CONFIG.get_int("semantic_dedupe", "default_ttl_seconds", 604800)
SEMANTIC_DEDUPE_DEBUG_LOG_REASONING = CONFIG.get_bool("semantic_dedupe", "debug_log_reasoning", False)
SEMANTIC_DEDUPE_TTL_BY_KIND = {
    "todoist_reminder": CONFIG.get_int("semantic_dedupe", "todoist_reminder_ttl_seconds", 7 * 24 * 60 * 60),
    "internal_task": CONFIG.get_int("semantic_dedupe", "internal_task_ttl_seconds", 24 * 60 * 60),
    "reflection_task": CONFIG.get_int("semantic_dedupe", "reflection_task_ttl_seconds", 7 * 24 * 60 * 60),
    "do_now_action": CONFIG.get_int("semantic_dedupe", "do_now_action_ttl_seconds", 2 * 60 * 60),
    "calendar_event": CONFIG.get_int("semantic_dedupe", "calendar_event_ttl_seconds", 14 * 24 * 60 * 60),
}


def _parse_json_list(section: str, option: str) -> list[str]:
    raw = CONFIG.get_str(section, option, "[]")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [str(item).strip() for item in payload if str(item).strip()]


PASSIVE_OBSERVER_IGNORE_APPS = _parse_json_list("passive_observer", "ignore_apps_json")
PASSIVE_OBSERVER_IGNORE_DOMAINS = _parse_json_list("passive_observer", "ignore_domains_json")
PASSIVE_OBSERVER_ALWAYS_FULL_APPS = _parse_json_list("passive_observer", "always_full_apps_json")
PASSIVE_OBSERVER_ALWAYS_FULL_DOMAINS = _parse_json_list("passive_observer", "always_full_domains_json")

configure_runtime_log_streaming(max_entries=LOG_API_BUFFER_SIZE, debug_enabled=DEBUG_MODE)


def ensure_runtime_databases() -> None:
    MEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    VOICE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    SQLiteMemoryAdapter(db_path=str(MEMORY_DB_PATH), memory_root=str(MEMORY_ROOT))
    SQLiteInteractionLogAdapter(db_path=str(INTERACTION_LOG_DB_PATH))
    SQLiteBenchmarkAdapter(db_path=str(BENCHMARK_DB_PATH))
    SQLiteTrainingDataAdapter(db_path=str(TRAINING_DATA_DB_PATH))
    SQLiteVoiceAdapter(str(VOICE_DB_PATH))
    night_mode.init_db()


class AmbientRuntime:
    TRANSCRIPT_EXECUTION_PROMPT = (
        "You are an ambient assistant processing a freshly produced audio transcript. "
        "Treat the transcript as recent conversational context from the user environment. "
        "Use tools when needed to take useful actions. "
        "If the transcript does not require any action, respond with a short factual summary of what matters."
    )
    CONTEXT_OVERFLOW_STATUS = "failed_context_overflow"
    FOLLOWUP_EXECUTION_PROMPT = (
        "You are an ambient assistant executing a queued follow-up task. "
        "Treat the user message as an execution brief, not as a question. "
        "Use the available tools when needed. "
        "Do not ask the user for clarification if the brief provides enough context to act. "
        "When the task is complete, state the concrete result."
    )
    TODOIST_EXECUTION_PROMPT = (
        "You are an ambient assistant executing an explicit task the user queued for the agent. "
        "Treat the user message as a direct execution brief from the user. "
        "This task has higher priority than passive observation or background analysis. "
        "Use the available tools when needed. "
        "Do not ask the user for clarification if the brief already provides enough context to act. "
        "When the task is complete, state the concrete result."
    )

    def __init__(   
        self,
        transcription_queue: queue.Queue,
        gpu_lock: Optional[threading.Lock] = None,
        audio_active_event: Optional[threading.Event] = None,
        llm_active_event: Optional[threading.Event] = None,
    ):
        self.queue = transcription_queue
        self.gpu_lock = gpu_lock or threading.Lock()
        self.audio_active_event = audio_active_event or threading.Event()
        self.llm_active_event = llm_active_event or threading.Event()
        self.stop_event = threading.Event()
        self._screenshot_capture_stop_event = threading.Event()
        self._screenshot_capture_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._tool_bridge: Optional[MCPToolAdapter] = None

    def _drain_queue(self, target_queue: queue.Queue) -> None:
        while True:
            try:
                target_queue.get_nowait()
            except queue.Empty:
                break
            else:
                target_queue.task_done()

    def stop_service(self) -> None:
        self.stop_event.set()
        self._screenshot_capture_stop_event.set()

    async def _sleep_or_stop(self, seconds: float) -> bool:
        deadline = time.monotonic() + max(0.0, seconds)
        while not self.stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            try:
                await asyncio.sleep(min(0.25, remaining))
            except asyncio.CancelledError:
                if self.stop_event.is_set():
                    return True
                raise
        return True

    def _is_context_overflow_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return (
            "exceed_context_size_error" in message
            or "exceeds the available context size" in message
            or "context window" in message
        )

    def _summarize_context_overflow(self, exc: Exception) -> str:
        text = str(exc)
        match = re.search(
            r"request\s*\((?P<prompt_tokens>\d+)\s+tokens\)\s+exceeds the available context size\s*\((?P<context_tokens>\d+)\s+tokens\)",
            text,
            re.IGNORECASE,
        )
        if not match:
            return "Task failed because the assembled prompt exceeded the loaded model context window."
        prompt_tokens = match.group("prompt_tokens")
        context_tokens = match.group("context_tokens")
        return (
            "Task failed because the assembled prompt exceeded the loaded model context window "
            f"({prompt_tokens} prompt tokens vs {context_tokens} available)."
        )

    def _quarantine_overflowed_task(self, task_queue: SQLiteTaskQueueAdapter, task, exc: Exception) -> None:
        reason = self._summarize_context_overflow(exc)
        logger.error("Queued follow-up task %s quarantined: %s", task.id, reason)
        task_queue.mark_task_complete(task.id, status=self.CONTEXT_OVERFLOW_STATUS)
        night_mode.add_notification(
            message=(
                f"Queued follow-up task {task.id} was skipped. {reason} "
                "Reduce retrieved context or narrow the task scope before re-queueing it."
            ),
            source="ambient_runtime",
        )

    def _mark_dedupe_item_completed(self, semantic_dedupe: SemanticDeduplicationService | None, task) -> None:
        if semantic_dedupe is None:
            return
        metadata_json = getattr(task, "metadata_json", None)
        if not metadata_json:
            return
        try:
            payload = json.loads(metadata_json)
        except (TypeError, json.JSONDecodeError):
            return
        dedupe_item_id = str(payload.get("dedupe_item_id", "")).strip()
        if dedupe_item_id:
            semantic_dedupe.mark_completed(dedupe_item_id)

    async def _ensure_llm_ready(
        self,
        llm_adapter: LlamaCppAdapter,
        model_name: str,
    ) -> None:
        await llm_adapter.load_model(model_name)
        self.llm_active_event.set()

    async def _release_llm(self, llm_adapter: LlamaCppAdapter) -> None:
        await llm_adapter.unload_model()
        self.llm_active_event.clear()

    async def _initialize_mcp_tools(
        self,
        tool_bridge: MCPToolAdapter,
        llm_service: LLMInteractionService,
    ) -> None:
        await tool_bridge.start_servers(MCP_CONFIG_PATH)
        await llm_service.initialize_tools()

    async def _ensure_runtime(
        self,
        *,
        llm_adapter: LlamaCppAdapter,
        services_initialized: bool,
        reason: str,
        model_name: str,
    ) -> bool:
        if not services_initialized:
            logger.info("Loading ambient runtime: %s", reason)
        await self._ensure_llm_ready(llm_adapter, model_name)
        return True

    async def _release_runtime(
        self,
        *,
        llm_adapter: LlamaCppAdapter,
        services_initialized: bool,
        reason: str,
    ) -> bool:
        if not services_initialized:
            return False
        logger.info("Unloading ambient runtime: %s", reason)
        await self._release_llm(llm_adapter)
        return False

    def _build_services(self):
        ensure_runtime_databases()
        llm_adapter = LlamaCppAdapter(base_url=API_BASE_URL, api_key=API_KEY)
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
            reporter_model=REPORTER_MODEL,
            artifact_root=str(ARTIFACTS_ROOT),
        )
        memory_store = SQLiteMemoryAdapter(
            db_path=str(MEMORY_DB_PATH),
            memory_root=str(MEMORY_ROOT),
        )
        semantic_adapter = None
        semantic_memory = None
        if SEMANTIC_MEMORY_ENABLED and EMBEDDING_MODEL:
            semantic_adapter = LlamaCppSemanticAdapter(
                embedding_base_url=EMBEDDING_API_BASE_URL,
                embedding_model=EMBEDDING_MODEL,
                reranker_base_url=RERANKER_API_BASE_URL,
                reranker_model=RERANKER_MODEL,
            )
            semantic_memory = SemanticMemoryService(
                memory=memory_store,
                semantic_adapter=semantic_adapter,
                sync_batch_size=SEMANTIC_SYNC_BATCH_SIZE,
                vector_limit=SEMANTIC_VECTOR_LIMIT,
                rerank_limit=SEMANTIC_RERANK_LIMIT,
            )
        semantic_dedupe = SemanticDeduplicationService(
            memory=memory_store,
            llm_provider=logged_llm,
            enabled=SEMANTIC_DEDUPE_ENABLED,
            model=SEMANTIC_DEDUPE_MODEL,
            candidate_limit=SEMANTIC_DEDUPE_CANDIDATE_LIMIT,
            default_ttl_seconds=SEMANTIC_DEDUPE_DEFAULT_TTL_SECONDS,
            per_entity_ttl_seconds=SEMANTIC_DEDUPE_TTL_BY_KIND,
            debug_log_reasoning=SEMANTIC_DEDUPE_DEBUG_LOG_REASONING,
        )
        task_queue = SQLiteTaskQueueAdapter()
        reflection_service = (
            ReflectionService(
                memory=memory_store,
                task_queue=task_queue,
                llm_provider=logged_llm,
                semantic_memory=semantic_memory,
                semantic_dedupe_service=semantic_dedupe,
                history_path=REFLECTION_HISTORY_PATH,
                cadence_mode=REFLECTION_CADENCE_MODE,
                interval_hours=REFLECTION_INTERVAL_HOURS,
                max_generated_tasks=REFLECTION_MAX_GENERATED_TASKS,
            )
            if REFLECTION_ENABLED
            else None
        )
        todoist_provider = None
        candidate = TodoistTaskAdapter()
        reminder_helper = TodoistHelper()
        if TODOIST_ENABLED and candidate.is_enabled():
            todoist_provider = candidate
        elif TODOIST_ENABLED and not candidate.is_enabled():
            logger.warning("Todoist integration is enabled but no usable Todoist API token was found.")
        elif not TODOIST_ENABLED and candidate.is_enabled():
            logger.info("Todoist API token is available but Todoist integration is disabled in config.")
        passive_followup = (
            PassiveObserverFollowupService(
                memory=memory_store,
                task_queue=task_queue,
                llm_provider=logged_llm,
                semantic_memory=semantic_memory,
                semantic_dedupe_service=semantic_dedupe,
                activity_ledger=None,
                reminder_helper=reminder_helper if TODOIST_ENABLED and reminder_helper.is_enabled() else None,
            )
            if PASSIVE_OBSERVER_ENABLED
            else None
        )
        user_biodata_service = (
            UserBioDataService(
                memory=memory_store,
                llm_provider=logged_llm,
                semantic_memory=semantic_memory,
            )
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
                fast_model=PASSIVE_OBSERVER_MODEL,
                full_model=FULL_PASSIVE_OBSERVER_MODEL,
                full_vlm_ssim_threshold=PASSIVE_OBSERVER_FULL_VLM_SSIM_THRESHOLD,
                ignore_apps=PASSIVE_OBSERVER_IGNORE_APPS,
                ignore_domains=PASSIVE_OBSERVER_IGNORE_DOMAINS,
                always_full_apps=PASSIVE_OBSERVER_ALWAYS_FULL_APPS,
                always_full_domains=PASSIVE_OBSERVER_ALWAYS_FULL_DOMAINS,
                fast_model_retry_count=PASSIVE_OBSERVER_FAST_MODEL_RETRY_COUNT,
                uiat_adapter=UIATAdapter(mode=PASSIVE_OBSERVER_UIAT_MODE) if PASSIVE_OBSERVER_FAST_ROUTING_ENABLED else None,
                persist_observations=not ALWAYS_ON_MODE,
            )
            if PASSIVE_OBSERVER_ENABLED
            else None
        )
        return (
            llm_adapter,
            tool_bridge,
            llm_service,
            memory_store,
            task_queue,
            semantic_dedupe,
            reflection_service,
            todoist_provider,
            passive_followup,
            user_biodata_service,
            system_idle_service,
            screenshot_queue,
            passive_observer,
        )

    def _start_screenshot_capture_loop(
        self,
        *,
        passive_observer: PassiveObserverService | None,
        screenshot_queue: ScreenshotQueueService | None,
        system_idle_service: SystemIdleService,
        capture_interval_seconds: float,
    ) -> None:
        if (
            not PASSIVE_OBSERVER_ENABLED
            or passive_observer is None
            or screenshot_queue is None
            or self._screenshot_capture_thread is not None
        ):
            return

        self._screenshot_capture_stop_event.clear()

        def _worker() -> None:
            logger.info(
                "Passive observer screenshot capture loop started (interval=%ss).",
                capture_interval_seconds,
            )
            while not self._screenshot_capture_stop_event.is_set():
                try:
                    if system_idle_service.is_user_idle():
                        self._screenshot_capture_stop_event.wait(capture_interval_seconds)
                        continue

                    screenshot_path = passive_observer.capture_screenshot()
                    queued = screenshot_queue.enqueue(screenshot_path)
                    if queued is None:
                        logger.info(
                            "Skipped screenshot for passive observer due to SSIM similarity filter: %s (queue_size=%s)",
                            screenshot_path,
                            screenshot_queue.size(),
                        )
                    else:
                        logger.info(
                            "Queued screenshot for passive observer: %s (queue_size=%s)",
                            queued.screenshot_path,
                            screenshot_queue.size(),
                        )
                        logger.debug(
                            "Queued screenshot similarity_score=%s path=%s",
                            queued.similarity_score,
                            queued.screenshot_path,
                        )
                except Exception:
                    logger.exception("Passive observer screenshot capture failed.")
                self._screenshot_capture_stop_event.wait(capture_interval_seconds)

        self._screenshot_capture_thread = threading.Thread(
            target=_worker,
            daemon=True,
            name="PassiveObserverCaptureThread",
        )
        self._screenshot_capture_thread.start()

    def _stop_screenshot_capture_loop(self) -> None:
        self._screenshot_capture_stop_event.set()
        if self._screenshot_capture_thread is not None:
            self._screenshot_capture_thread.join(timeout=5.0)
            self._screenshot_capture_thread = None

    async def run_loop(self):
        (
            llm_adapter,
            tool_bridge,
            llm_service,
            memory_store,
            task_queue,
            semantic_dedupe,
            reflection_service,
            todoist_provider,
            passive_followup,
            user_biodata_service,
            system_idle_service,
            screenshot_queue,
            passive_observer,
        ) = self._build_services()
        idle_cycle_interval = 30
        passive_observer_interval = 10
        last_idle_cycle_at = 0.0
        user_idle_now = False
        services_initialized = False

        try:
            self.stop_event.clear()
            self._tool_bridge = tool_bridge
            await self._initialize_mcp_tools(tool_bridge, llm_service)
            self._start_screenshot_capture_loop(
                passive_observer=passive_observer,
                screenshot_queue=screenshot_queue,
                system_idle_service=system_idle_service,
                capture_interval_seconds=passive_observer_interval,
            )
            logger.info("Starting reduced ambient runtime manager.")

            while not self.stop_event.is_set():
                current_user_idle = system_idle_service.is_user_idle()
                if current_user_idle != user_idle_now:
                    user_idle_now = current_user_idle
                    if user_idle_now:
                        logger.info("User idle detected (>= %ss).", USER_IDLE_THRESHOLD_SECONDS)
                        last_idle_cycle_at = 0.0
                    else:
                        logger.info("User activity detected. Passive follow-up will wait until idle resumes.")

                if self.audio_active_event.is_set():
                    logger.info("ASR claimed GPU. Unloading ambient runtime until audio processing completes.")
                    services_initialized = await self._release_runtime(
                        llm_adapter=llm_adapter,
                        services_initialized=services_initialized,
                        reason="ASR pipeline is using the GPU",
                    )
                    while self.audio_active_event.is_set() and not self.stop_event.is_set():
                        await asyncio.sleep(0.25)
                    last_idle_cycle_at = 0.0
                    if self.stop_event.is_set():
                        break

                while True:
                    try:
                        transcript_path = self.queue.get_nowait()
                    except queue.Empty:
                        break
                    if ALWAYS_ON_MODE:
                        logger.info("Transcript produced at %s. Processing immediately in always_on mode.", transcript_path)
                        try:
                            transcript_text = Path(transcript_path).read_text(encoding="utf-8").strip()
                            if transcript_text:
                                services_initialized = await self._ensure_runtime(
                                    llm_adapter=llm_adapter,
                                    services_initialized=services_initialized,
                                    reason="processing transcript in always_on mode",
                                    model_name=TRANSCRIPT_PROCESSING_MODEL,
                                )
                                llm_service.reset_context()
                                try:
                                    with self.gpu_lock:
                                        result = await llm_service.run_interaction(
                                            user_input=transcript_text,
                                            system_prompt=self.TRANSCRIPT_EXECUTION_PROMPT,
                                            model=TRANSCRIPT_PROCESSING_MODEL,
                                            report_policy="auto_surface",
                                        )
                                    logger.info(
                                        "Transcript %s processed with result: %s",
                                        transcript_path,
                                        result[:500],
                                    )
                                finally:
                                    llm_service.reset_context()
                            else:
                                logger.info("Transcript %s was empty after read; skipping.", transcript_path)
                        except Exception:
                            logger.exception("Transcript processing failed for %s", transcript_path)
                    else:
                        logger.info(
                            "Transcript produced at %s. Downstream transcript reasoning has been removed from the runtime.",
                            transcript_path,
                        )
                    self.queue.task_done()

                now = time.monotonic()
                runtime_active_now = user_idle_now or ALWAYS_ON_MODE

                if runtime_active_now and todoist_provider is not None:
                    try:
                        explicit_tasks = todoist_provider.get_tasks()
                    except Exception:
                        logger.exception("Failed to fetch Todoist tasks.")
                        explicit_tasks = []
                    if explicit_tasks:
                        task = explicit_tasks[0]
                        logger.info(
                            "Executing highest-priority Todoist task ID %s while idle: %s",
                            task.get("id"),
                            task.get("content"),
                        )
                        try:
                            services_initialized = await self._ensure_runtime(
                                llm_adapter=llm_adapter,
                                services_initialized=services_initialized,
                                reason="executing explicit Todoist task",
                                model_name=TODOIST_EXECUTION_MODEL,
                            )
                            llm_service.reset_context()
                            try:
                                with self.gpu_lock:
                                    result = await llm_service.run_interaction(
                                        user_input=task.get("content", ""),
                                        system_prompt=self.TODOIST_EXECUTION_PROMPT,
                                        model=TODOIST_EXECUTION_MODEL,
                                        report_policy="auto_surface",
                                    )
                            finally:
                                llm_service.reset_context()
                            logger.info("Todoist task %s completed with result: %s", task.get("id"), result[:500])
                            todoist_provider.complete_task(task["id"])
                        except Exception:
                            logger.exception("Todoist task execution failed.")
                        last_idle_cycle_at = now
                        if await self._sleep_or_stop(1):
                            break
                        continue

                if (
                    PASSIVE_OBSERVER_ENABLED
                    and passive_observer is not None
                    and screenshot_queue is not None
                    and runtime_active_now
                    and not screenshot_queue.is_empty()
                ):
                    processed_screenshots = 0
                    while processed_screenshots < MAX_SCREENSHOTS_PER_IDLE_CYCLE and not screenshot_queue.is_empty():
                        if not ALWAYS_ON_MODE and not system_idle_service.is_user_idle():
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
                            logger.debug(
                                "Processing queued screenshot similarity_score=%s path=%s",
                                job.similarity_score,
                                job.screenshot_path,
                            )
                            services_initialized = await self._ensure_runtime(
                                llm_adapter=llm_adapter,
                                services_initialized=services_initialized,
                                reason="processing queued passive-observer screenshots",
                                model_name=PASSIVE_OBSERVER_MODEL,
                            )
                            with self.gpu_lock:
                                observation = await passive_observer.process_screenshot(
                                    screenshot_path=job.screenshot_path,
                                    model=PASSIVE_OBSERVER_MODEL,
                                    recent_context=memory_store.get_recent_context(),
                                    captured_at=job.captured_at,
                                    similarity_score=job.similarity_score,
                                )
                            if observation is not None:
                                logger.info(
                                    "Processed queued screenshot for %s.",
                                    observation.app_name or observation.page_hint or "screen",
                                )
                                if ALWAYS_ON_MODE and passive_followup is not None:
                                    followup_result = await passive_followup.process_observations(
                                        observations=[observation],
                                        model=PASSIVE_FOLLOWUP_MODEL,
                                        mark_sent=False,
                                        apply_memory_updates=False,
                                    )
                                    logger.info(
                                        "Always-on follow-up processed observation %s with %s queued and %s do-now activities.",
                                        observation.observation_id,
                                        len(followup_result.get("queued_activities", [])),
                                        len(followup_result.get("do_now_activities", [])),
                                    )
                                    for activity in followup_result.get("do_now_activities", []):
                                        logger.info("Executing always-on passive follow-up do-now activity: %s", activity)
                                        services_initialized = await self._ensure_runtime(
                                            llm_adapter=llm_adapter,
                                            services_initialized=services_initialized,
                                            reason="executing always-on passive follow-up do-now activity",
                                            model_name=FOLLOWUP_EXECUTION_MODEL,
                                        )
                                        llm_service.reset_context()
                                        try:
                                            with self.gpu_lock:
                                                result = await llm_service.run_interaction(
                                                    user_input=activity,
                                                    system_prompt=self.FOLLOWUP_EXECUTION_PROMPT,
                                                    model=FOLLOWUP_EXECUTION_MODEL,
                                                    report_policy="auto_surface",
                                                )
                                        finally:
                                            llm_service.reset_context()
                                        logger.info("Always-on do-now activity completed with result: %s", result[:500])
                        except Exception:
                            logger.exception("Queued passive observer screenshot processing failed.")
                        processed_screenshots += 1

                if runtime_active_now and now - last_idle_cycle_at >= idle_cycle_interval:
                    try:
                        if reflection_service is not None and not ALWAYS_ON_MODE:
                            services_initialized = await self._ensure_runtime(
                                llm_adapter=llm_adapter,
                                services_initialized=services_initialized,
                                reason="running reflection service",
                                model_name=REFLECTION_MODEL,
                            )
                            with self.gpu_lock:
                                reflection_result = await reflection_service.run_if_due(model=REFLECTION_MODEL)
                            if reflection_result.get("ran"):
                                logger.info(
                                    "Reflection service ran: cleaned_changed=%s, generated=%s, queued=%s, skipped=%s.",
                                    reflection_result.get("cleaned_user_info_changed"),
                                    len(reflection_result.get("generated_tasks", [])),
                                    len(reflection_result.get("queued_tasks", [])),
                                    len(reflection_result.get("skipped_tasks", [])),
                                )
                        if user_biodata_service is not None and not ALWAYS_ON_MODE:
                            services_initialized = await self._ensure_runtime(
                                llm_adapter=llm_adapter,
                                services_initialized=services_initialized,
                                reason="updating user biodata from passive observations",
                                model_name=USER_BIODATA_MODEL,
                            )
                            with self.gpu_lock:
                                biodata_result = await user_biodata_service.update_biodata(model=USER_BIODATA_MODEL)
                            logger.info(
                                "User BioData update processed %s observations and appended %s entries.",
                                len(biodata_result.get("processed_observation_ids", [])),
                                len(biodata_result.get("entries", [])),
                            )
                        if passive_followup is not None and not ALWAYS_ON_MODE:
                            services_initialized = await self._ensure_runtime(
                                llm_adapter=llm_adapter,
                                services_initialized=services_initialized,
                                reason="running passive observer follow-up",
                                model_name=PASSIVE_FOLLOWUP_MODEL,
                            )
                            with self.gpu_lock:
                                followup_result = await passive_followup.maybe_queue_followup(model=PASSIVE_FOLLOWUP_MODEL)
                            logger.info(
                                "Passive observer follow-up processed %s observations, %s unique activities, %s useful activities, %s queued, %s do-now.",
                                len(followup_result.get("processed_observation_ids", [])),
                                len(followup_result.get("unique_activities", [])),
                                len(followup_result.get("useful_activities", [])),
                                len(followup_result.get("queued_activities", [])),
                                len(followup_result.get("do_now_activities", [])),
                            )
                            for activity in followup_result.get("do_now_activities", []):
                                logger.info("Executing passive follow-up do-now activity: %s", activity)
                                services_initialized = await self._ensure_runtime(
                                    llm_adapter=llm_adapter,
                                    services_initialized=services_initialized,
                                    reason="executing passive follow-up do-now activity",
                                    model_name=FOLLOWUP_EXECUTION_MODEL,
                                )
                                llm_service.reset_context()
                                try:
                                    with self.gpu_lock:
                                        result = await llm_service.run_interaction(
                                            user_input=activity,
                                            system_prompt=self.FOLLOWUP_EXECUTION_PROMPT,
                                            model=FOLLOWUP_EXECUTION_MODEL,
                                            report_policy="auto_surface",
                                        )
                                finally:
                                    llm_service.reset_context()
                                logger.info("Do-now activity completed with result: %s", result[:500])
                        queue_task_execution_allowed = user_idle_now or PERFORM_QUEUE_TASKS
                        pending_tasks = task_queue.get_pending_tasks() if queue_task_execution_allowed else []
                        if pending_tasks:
                            task = pending_tasks[0]
                            logger.info("Executing queued follow-up task ID %s with tool-enabled LLM interaction.", task.id)
                            services_initialized = await self._ensure_runtime(
                                llm_adapter=llm_adapter,
                                services_initialized=services_initialized,
                                reason="executing queued follow-up task",
                                model_name=FOLLOWUP_EXECUTION_MODEL,
                            )
                            llm_service.reset_context()
                            try:
                                with self.gpu_lock:
                                    result = await llm_service.run_interaction(
                                        user_input=task.description,
                                        system_prompt=self.FOLLOWUP_EXECUTION_PROMPT,
                                        model=FOLLOWUP_EXECUTION_MODEL,
                                        report_policy="auto_surface",
                                    )
                            except Exception as exc:
                                if self._is_context_overflow_error(exc):
                                    self._quarantine_overflowed_task(task_queue, task, exc)
                                else:
                                    raise
                            else:
                                logger.info("Queued follow-up task %s completed with result: %s", task.id, result[:500])
                                task_queue.mark_task_complete(task.id)
                                self._mark_dedupe_item_completed(semantic_dedupe, task)
                            finally:
                                llm_service.reset_context()
                    except Exception:
                        logger.exception("Idle follow-up cycle failed.")
                    last_idle_cycle_at = now

                if not user_idle_now and not ALWAYS_ON_MODE:
                    services_initialized = await self._release_runtime(
                        llm_adapter=llm_adapter,
                        services_initialized=services_initialized,
                        reason="user is active; keep VRAM clear",
                    )

                if await self._sleep_or_stop(1):
                    break
        except asyncio.CancelledError:
            self.stop_event.set()
            logger.info("Ambient runtime loop cancelled during shutdown.")
        finally:
            self._stop_screenshot_capture_loop()
            llm_service.reset_context()
            if services_initialized:
                await self._release_runtime(
                    llm_adapter=llm_adapter,
                    services_initialized=services_initialized,
                    reason="application shutdown",
                )
            await tool_bridge.cleanup()
            self._tool_bridge = None
            self.llm_active_event.clear()
            self._drain_queue(self.queue)

    def start_service(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self.run_loop())
        finally:
            self._loop = None
            loop.close()


if __name__ == "__main__":
    runtime_log_started = False
    if LOG_API_ENABLED:
        interaction_store = SQLiteInteractionLogAdapter(db_path=str(INTERACTION_LOG_DB_PATH))
        training_store = SQLiteTrainingDataAdapter(db_path=str(TRAINING_DATA_DB_PATH))
        training_service = TrainingDataService(
            store=training_store,
            interaction_store=interaction_store,
            training_root=str(TRAINING_DATA_ROOT),
            user_data_dir=str(USER_DATA_DIR),
            uploads_dir=str(UPLOADS_DIR),
            transcripts_dir=str(TRANSCRIPTIONS_DIR),
            cleaned_audio_dir=str(CLEANED_AUDIO_DIR),
        )
        start_runtime_log_server(
            host=LOG_API_HOST,
            port=LOG_API_PORT,
            max_entries=LOG_API_BUFFER_SIZE,
            report_store=interaction_store,
            task_store=SQLiteTaskQueueAdapter(),
            benchmark_store=SQLiteBenchmarkAdapter(db_path=str(BENCHMARK_DB_PATH)),
            training_store=training_store,
            training_service=training_service,
            media_roots=[str(USER_DATA_DIR), str(TRAINING_DATA_ROOT)],
        )
        logger.info("Runtime log server started at http://%s:%s/logs", LOG_API_HOST, LOG_API_PORT)
        runtime_log_started = True

    gpu_lock = threading.Lock()
    audio_active_event = threading.Event()
    llm_active_event = threading.Event()

    audio_agent = AudioAgentService(
        gpu_lock=gpu_lock,
        audio_active_event=audio_active_event,
        llm_active_event=llm_active_event,
    )
    transcription_queue = audio_agent.get_transcription_queue()

    ambient_runtime = AmbientRuntime(
        transcription_queue=transcription_queue,
        gpu_lock=gpu_lock,
        audio_active_event=audio_active_event,
        llm_active_event=llm_active_event,
    )

    audio_thread = threading.Thread(
        target=audio_agent.start_service,
        name="AudioPipelineThread",
    )
    runtime_thread = threading.Thread(
        target=ambient_runtime.start_service,
        name="AmbientRuntimeThread",
    )

    audio_thread.start()
    runtime_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Stopping Ambient AI services.")
    finally:
        ambient_runtime.stop_service()
        audio_agent.stop_service()
        runtime_thread.join(timeout=15.0)
        audio_thread.join(timeout=15.0)
        if runtime_thread.is_alive():
            logger.warning("Ambient runtime thread did not exit cleanly before timeout.")
        if audio_thread.is_alive():
            logger.warning("Audio pipeline thread did not exit cleanly before timeout.")
        if runtime_log_started:
            shutdown_runtime_log_server(join_timeout=10.0, remove_log_handler=True)
        gc.collect()
