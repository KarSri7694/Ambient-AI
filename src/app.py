import asyncio
import gc
import json
import logging
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import queue
import re
import threading
import time
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from application.services.passive_observer_followup_service import PassiveObserverFollowupService
from application.services.reflection_service import ReflectionService
from application.services.llm_interaction_service import LLMInteractionService
from application.services.interaction_trace import interaction_trace
from application.services.scheduled_task_service import ScheduledTaskService
from application.services.passive_observer_service import PassiveObserverService
from application.services.semantic_deduplication_service import SemanticDeduplicationService
from application.services.semantic_memory_service import SemanticMemoryService
from application.services.screenshot_queue_service import ScreenshotQueueService
from application.services.system_idle_service import SystemIdleService
from application.services.training_data_service import TrainingDataService
from application.services.user_bio_data_service import UserBioDataService
from application.services.autonomy_coordinator_service import AutonomyCoordinatorService
from application.services.capability_policy_service import AutonomyBudget, CapabilityPolicyService
from application.services.opportunity_judgment_service import OpportunityJudgmentService
from application.services.capture_control_service import CaptureControlService
from application.services.resource_governor_service import (
    ModelResidencyManager,
    ResourceGovernorService,
    ResourceUnavailableError,
)
from core.models import InferenceRequest
from audio_agent import AudioAgentService
from infrastructure.adapter.LlamaCppSemanticAdapter import LlamaCppSemanticAdapter
from infrastructure.adapter.llamaCppAdapter import LlamaCppAdapter
from infrastructure.adapter.LoggingLLMProvider import LoggingLLMProvider
from infrastructure.adapter.BrowserMCPToolAdapter import BrowserMCPToolAdapter
from infrastructure.adapter.MCPToolAdapter import MCPToolAdapter
from infrastructure.adapter.MSSScreenCaptureAdapter import MssScreenCaptureAdapter
from infrastructure.adapter.SQLiteBenchmarkAdapter import SQLiteBenchmarkAdapter
from infrastructure.adapter.SQLiteChatAdapter import ChatEventBroker, SQLiteChatAdapter
from infrastructure.adapter.SQLiteInteractionLogAdapter import SQLiteInteractionLogAdapter
from infrastructure.adapter.SQLiteMemoryAdapter import SQLiteMemoryAdapter
from infrastructure.adapter.SQLiteTaskQueueAdapter import SQLiteTaskQueueAdapter
from infrastructure.adapter.SQLiteTrainingDataAdapter import SQLiteTrainingDataAdapter
from infrastructure.adapter.SQLiteVoiceAdapter import SQLiteVoiceAdapter
from infrastructure.adapter.SQLiteAutonomyAdapter import SQLiteAutonomyAdapter
from infrastructure.adapter.TodoistTaskAdapter import TodoistTaskAdapter
from infrastructure.adapter.UIATAdapter import UIATAdapter
from infrastructure.windows_resource_monitor import WindowsResourceMonitor
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
BROWSER_AGENT_MODEL = CONFIG.get_model("browser_agent_model", FOLLOWUP_EXECUTION_MODEL)
CHAT_MODEL = CONFIG.get_model("chat_model", FOLLOWUP_EXECUTION_MODEL)
LIGHTWEIGHT_CHAT_MODEL = CONFIG.get_model("lightweight_chat_model", "")
USER_DATA_DIR = Path(CONFIG.get_str("runtime", "user_data_dir", "D:\\USER_DATA"))
PROJECT_ROOT = Path(__file__).parent.parent
MEMORY_ROOT = USER_DATA_DIR / "memory"
MEMORY_DB_PATH = USER_DATA_DIR / "database" / "memory.db"
INTERACTION_LOG_DB_PATH = USER_DATA_DIR / "database" / "interaction_logs.db"
AUTONOMY_DB_PATH = USER_DATA_DIR / "database" / "autonomy.db"
CHAT_DB_PATH = Path(CONFIG.get_str("chat", "db_path", str(USER_DATA_DIR / "database" / "chat.db")))
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
PASSIVE_OBSERVER_CAPTURE_INTERVAL_SECONDS = CONFIG.get_float(
    "passive_observer", "capture_interval_seconds", 10.0
)
PASSIVE_OBSERVER_FAST_ROUTING_ENABLED = CONFIG.get_bool("passive_observer", "fast_routing_enabled", True)
PASSIVE_OBSERVER_FULL_VLM_SSIM_THRESHOLD = CONFIG.get_float("passive_observer", "full_vlm_ssim_threshold", 0.70)
PASSIVE_OBSERVER_FAST_MODEL_RETRY_COUNT = CONFIG.get_int("passive_observer", "fast_model_retry_count", 2)
PASSIVE_OBSERVER_UIAT_MODE = CONFIG.get_str("passive_observer", "uiat_mode", "screen_content")
LOG_API_ENABLED = CONFIG.get_bool("log_api", "enabled", True)
LOG_API_HOST = CONFIG.get_str("log_api", "host", "0.0.0.0")
LOG_API_PORT = CONFIG.get_int("log_api", "port", 8765)
LOG_API_BUFFER_SIZE = CONFIG.get_int("log_api", "buffer_size", 2000)
MCP_CONFIG_PATH = CONFIG.get_str("runtime", "mcp_config_path", "mcp.json")
BROWSER_MCP_SERVER_NAME = CONFIG.get_str("browser", "server_name", "playwright")
BROWSER_TASK_TIMEOUT_SECONDS = CONFIG.get_float("browser", "task_timeout_seconds", 180.0)
BROWSER_HEADLESS = CONFIG.get_bool("browser", "headless", False)
BROWSER_PROFILE_DIR = CONFIG.get_str(
    "browser",
    "persistent_profile_dir",
    str(USER_DATA_DIR / "browser" / "profile"),
)
from infrastructure.plain_capture_store import PlainCaptureStore
CHAT_HISTORY_MESSAGE_LIMIT = CONFIG.get_int("chat", "history_message_limit", 40)
CHAT_STREAM_CHECKPOINT_SECONDS = CONFIG.get_float("chat", "stream_checkpoint_seconds", 0.25)
CHAT_RESPONSE_WAIT_SECONDS = max(
    0.0,
    CONFIG.get_float("chat", "response_wait_seconds", 120.0),
)
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
# The unified coordinator is now the only production autonomy path. Legacy idle/night
# services remain importable for historical tests and data migration, but cannot be
# selected by runtime configuration.
AUTONOMY_COORDINATOR_ENABLED = True
AUTONOMY_MODE = CONFIG.get_str("autonomy", "mode", "shadow").strip().lower()
AUTONOMY_EVENT_LEASE_SECONDS = CONFIG.get_int("ambient", "event_lease_seconds", 180)
AUTONOMY_MAX_TOOL_CALLS_PER_HOUR = CONFIG.get_int("autonomy", "max_tool_calls_per_hour", 120)
AUTONOMY_MAX_WEB_QUERIES_PER_DAY = CONFIG.get_int("autonomy", "max_web_queries_per_day", 60)
AUTONOMY_MAX_INBOX_ITEMS_PER_DAY = CONFIG.get_int("autonomy", "max_inbox_items_per_day", 30)
CAPTURE_STORAGE_ROOT = Path(
    CONFIG.get_str("privacy", "capture_root", str(USER_DATA_DIR / "captures"))
)
RESOURCE_PRESET = CONFIG.get_str("resource_governor", "preset", "balanced").strip().lower()
RESOURCE_CRITICAL_RAM_MB = CONFIG.get_int("resource_governor", "critical_ram_mb", 2048)
RESOURCE_CRITICAL_RAM_PERCENT = CONFIG.get_float("resource_governor", "critical_ram_percent", 15.0)
RESOURCE_CRITICAL_VRAM_MB = CONFIG.get_int("resource_governor", "critical_vram_mb", 512)
RESOURCE_RECOVERY_STABLE_SECONDS = CONFIG.get_float("resource_governor", "recovery_stable_seconds", 30.0)
RESOURCE_TRANSITION_COOLDOWN_SECONDS = CONFIG.get_float("resource_governor", "transition_cooldown_seconds", 60.0)
RESOURCE_BATCH_MAX_EVENTS = CONFIG.get_int("resource_governor", "batch_max_events", 8)
RESOURCE_BATCH_MAX_SECONDS = CONFIG.get_float("resource_governor", "batch_max_seconds", 90.0)
RESOURCE_DEFER_SECONDS = CONFIG.get_int("resource_governor", "defer_seconds", 30)
DEFERRED_ASR_MAX_AUDIO_SECONDS = CONFIG.get_float("resource_governor", "asr_batch_audio_seconds", 300.0)


def check_model_server() -> tuple[bool, str]:
    """Check the manually managed llama.cpp router without loading a model."""
    request = Request(
        f"{API_BASE_URL.rstrip('/')}/v1/models",
        headers={"Authorization": f"Bearer {API_KEY}"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=2.0) as response:
            return 200 <= int(response.status) < 500, f"HTTP {response.status}"
    except HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except (URLError, OSError, TimeoutError) as exc:
        return False, str(exc.reason if isinstance(exc, URLError) else exc)


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
BROWSER_DENIED_TOOL_NAMES = set(
    _parse_json_list("browser", "denied_tools_json")
    or ["browser_run_code", "browser_run_code_unsafe"]
)

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
    CHAT_SYSTEM_PROMPT = (
        "You are Ambient AI in a direct conversation with the local user. "
        "Answer informational questions clearly and use read-only tools when they help. "
        "Use state-changing tools only when the user's current message explicitly requests the action. "
        "For a clear request to do something at an exact future time, call schedule_task_at with a detailed "
        "standalone task and an absolute ISO 8601 date-time. If the requested time is ambiguous, ask a short "
        "clarifying question instead of guessing. Execute explicitly requested immediate tasks now."
    )

    def __init__(   
        self,
        transcription_queue: queue.Queue,
        gpu_lock: Optional[threading.Lock] = None,
        audio_active_event: Optional[threading.Event] = None,
        llm_active_event: Optional[threading.Event] = None,
        chat_store: Optional[SQLiteChatAdapter] = None,
        chat_event_broker: Optional[ChatEventBroker] = None,
        capture_store: Optional[PlainCaptureStore] = None,
        capture_control: Optional[CaptureControlService] = None,
        resource_governor: Optional[ResourceGovernorService] = None,
    ):
        self.queue = transcription_queue
        self.gpu_lock = gpu_lock or threading.Lock()
        self.audio_active_event = audio_active_event or threading.Event()
        self.llm_active_event = llm_active_event or threading.Event()
        self.chat_store = chat_store
        self.chat_event_broker = chat_event_broker
        self.capture_store = capture_store
        self.capture_control = capture_control or CaptureControlService()
        self.resource_governor = resource_governor or ResourceGovernorService(
            monitor=WindowsResourceMonitor(), preset=RESOURCE_PRESET,
            critical_ram_mb=RESOURCE_CRITICAL_RAM_MB,
            critical_ram_percent=RESOURCE_CRITICAL_RAM_PERCENT,
            critical_vram_mb=RESOURCE_CRITICAL_VRAM_MB,
        )
        self.stop_event = threading.Event()
        self._screenshot_capture_stop_event = threading.Event()
        self._screenshot_capture_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._tool_bridge: Optional[MCPToolAdapter] = None
        self._chat_response_deadline: Optional[float] = None
        self._chat_resource_backoff_until = 0.0
        self._deferred_followup_observations: deque = deque()
        self._deferred_followup_activities: deque[str] = deque()

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

    def _start_chat_response_window(self, now: Optional[float] = None) -> None:
        started_at = time.monotonic() if now is None else now
        self._chat_response_deadline = started_at + CHAT_RESPONSE_WAIT_SECONDS
        logger.info(
            "Chat model response window opened for %ss.",
            CHAT_RESPONSE_WAIT_SECONDS,
        )

    def _chat_response_window_active(self, now: Optional[float] = None) -> bool:
        if self._chat_response_deadline is None:
            return False
        current = time.monotonic() if now is None else now
        return current < self._chat_response_deadline

    async def _restore_chat_residency(
        self,
        *,
        llm_adapter: ModelResidencyManager,
        services_initialized: bool,
        reason: str,
        open_response_window: bool = True,
    ) -> bool:
        """Restore only the configured tiny chat model when memory is healthy."""
        try:
            # Keep the legacy model-manager port usable for focused unit tests and
            # external adapters. Production always supplies ModelResidencyManager.
            if not hasattr(llm_adapter, "ensure_lightweight_resident"):
                await llm_adapter.load_model(CHAT_MODEL)
                services_initialized = True
                self.llm_active_event.set()
            elif not LIGHTWEIGHT_CHAT_MODEL:
                if services_initialized and hasattr(llm_adapter, "ensure_lightweight_resident"):
                    await self._release_llm(llm_adapter)
                    services_initialized = False
            else:
                services_initialized = await llm_adapter.ensure_lightweight_resident(
                    user_active=True,
                    startup=(not services_initialized and "startup" in reason.lower()),
                )
                if services_initialized:
                    self.llm_active_event.set()
        except Exception:
            logger.exception("Failed to restore the lightweight chat model: %s", reason)
            return services_initialized
        if open_response_window and services_initialized:
            self._start_chat_response_window()
        elif not services_initialized:
            self._chat_response_deadline = None
        return services_initialized

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
        llm_adapter: ModelResidencyManager,
        model_name: str,
        *,
        role: str = "interactive",
        background: bool = False,
        user_active: bool = True,
    ) -> None:
        try:
            decision = await llm_adapter.load_model(
                model_name,
                role=role,
                background=background,
                user_active=user_active,
            )
        except TypeError:
            await llm_adapter.load_model(model_name)
            decision = None
        if decision is not None and not decision.allowed:
            raise ResourceUnavailableError(decision)
        self.llm_active_event.set()

    async def _release_llm(self, llm_adapter: ModelResidencyManager) -> None:
        try:
            await llm_adapter.unload_model(reason="ambient work unit completed")
        except TypeError:
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
        llm_adapter: ModelResidencyManager,
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
        llm_adapter: ModelResidencyManager,
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
        raw_llm_adapter = LlamaCppAdapter(base_url=API_BASE_URL, api_key=API_KEY)
        autonomy_store = SQLiteAutonomyAdapter(str(AUTONOMY_DB_PATH))
        if self.resource_governor.audit is None:
            self.resource_governor.audit = autonomy_store.audit
        residency_manager = ModelResidencyManager(
            provider=raw_llm_adapter,
            governor=self.resource_governor,
            lightweight_chat_model=LIGHTWEIGHT_CHAT_MODEL,
            recovery_stable_seconds=RESOURCE_RECOVERY_STABLE_SECONDS,
            transition_cooldown_seconds=RESOURCE_TRANSITION_COOLDOWN_SECONDS,
        )
        self.resource_governor.set_residency_status_provider(residency_manager.status)
        interaction_log_store = SQLiteInteractionLogAdapter(
            db_path=str(INTERACTION_LOG_DB_PATH),
        )
        logged_llm = LoggingLLMProvider(
            provider=raw_llm_adapter,
            log_store=interaction_log_store,
            current_response_path=str(CURRENT_RESPONSE_PATH),
            capture_store=self.capture_store,
            residency_manager=residency_manager,
        )
        capability_policy = CapabilityPolicyService(
            store=autonomy_store,
            budget=AutonomyBudget(
                max_tool_calls_per_hour=AUTONOMY_MAX_TOOL_CALLS_PER_HOUR,
                max_web_queries_per_day=AUTONOMY_MAX_WEB_QUERIES_PER_DAY,
            ),
        )
        tool_bridge = MCPToolAdapter()
        task_queue = SQLiteTaskQueueAdapter()
        scheduled_task_service = ScheduledTaskService(task_queue)
        browser_tool_bridge = BrowserMCPToolAdapter(
            config_path=MCP_CONFIG_PATH,
            server_name=BROWSER_MCP_SERVER_NAME,
            profile_dir=BROWSER_PROFILE_DIR,
            denied_tool_names=BROWSER_DENIED_TOOL_NAMES,
        )
        llm_service = LLMInteractionService(
            llm_provider=logged_llm,
            tool_bridge=tool_bridge,
            browser_tool_bridge=browser_tool_bridge,
            browser_agent_model=BROWSER_AGENT_MODEL,
            browser_task_timeout_seconds=BROWSER_TASK_TIMEOUT_SECONDS,
            browser_headless=BROWSER_HEADLESS,
            scheduled_task_service=scheduled_task_service,
            reporter_model=REPORTER_MODEL,
            artifact_root=str(ARTIFACTS_ROOT),
            capability_policy=capability_policy,
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
                activity_ledger=autonomy_store,
                reminder_helper=None,
            )
            if PASSIVE_OBSERVER_ENABLED and not AUTONOMY_COORDINATOR_ENABLED
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
                discard_handler=(
                    lambda path, reason: self.capture_store.store_file(
                        path, kind=f"screenshot_{reason}", delete_source=True
                    )
                    if self.capture_store is not None and Path(path).exists()
                    else None
                ),
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
                fast_model=(
                    FOLLOWUP_EXECUTION_MODEL
                    if AUTONOMY_COORDINATOR_ENABLED
                    else PASSIVE_OBSERVER_MODEL
                ),
                # Coordinator batches deliberately use one model across fast/full
                # screen analysis, judgment, and research to avoid model swaps.
                full_model=(
                    FOLLOWUP_EXECUTION_MODEL
                    if AUTONOMY_COORDINATOR_ENABLED
                    else FULL_PASSIVE_OBSERVER_MODEL
                ),
                full_vlm_ssim_threshold=PASSIVE_OBSERVER_FULL_VLM_SSIM_THRESHOLD,
                ignore_apps=PASSIVE_OBSERVER_IGNORE_APPS,
                ignore_domains=PASSIVE_OBSERVER_IGNORE_DOMAINS,
                always_full_apps=PASSIVE_OBSERVER_ALWAYS_FULL_APPS,
                always_full_domains=PASSIVE_OBSERVER_ALWAYS_FULL_DOMAINS,
                fast_model_retry_count=PASSIVE_OBSERVER_FAST_MODEL_RETRY_COUNT,
                uiat_adapter=UIATAdapter(mode=PASSIVE_OBSERVER_UIAT_MODE) if PASSIVE_OBSERVER_FAST_ROUTING_ENABLED else None,
                persist_observations=not ALWAYS_ON_MODE,
                capture_store=self.capture_store,
                persist_payloads=AUTONOMY_COORDINATOR_ENABLED,
                capture_control=self.capture_control,
            )
            if PASSIVE_OBSERVER_ENABLED
            else None
        )
        autonomy_coordinator = (
            AutonomyCoordinatorService(
                store=autonomy_store,
                judgment=OpportunityJudgmentService(llm_provider=logged_llm),
                policy=capability_policy,
                mode=AUTONOMY_MODE,
                event_lease_seconds=AUTONOMY_EVENT_LEASE_SECONDS,
                max_inbox_items_per_day=AUTONOMY_MAX_INBOX_ITEMS_PER_DAY,
                capture_store=self.capture_store,
                visual_observer=passive_observer,
                # Use one vision-capable model for enrichment, judgment, and research
                # so an ambient batch never swaps presets between its phases.
                visual_model=FOLLOWUP_EXECUTION_MODEL,
            )
            if AUTONOMY_COORDINATOR_ENABLED
            else None
        )
        return (
            residency_manager,
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
            autonomy_store,
            autonomy_coordinator,
        )

    def _start_screenshot_capture_loop(
        self,
        *,
        passive_observer: PassiveObserverService | None,
        screenshot_queue: ScreenshotQueueService | None,
        system_idle_service: SystemIdleService,
        capture_interval_seconds: float,
        autonomy_coordinator: AutonomyCoordinatorService | None = None,
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
                if self.capture_control.is_paused():
                    self._screenshot_capture_stop_event.wait(0.5)
                    continue
                try:
                    lightweight_context = passive_observer.capture_lightweight_context()
                    if self.capture_control.is_excluded(
                        app_name=str(lightweight_context.get("app_name") or ""),
                        domain=str(lightweight_context.get("domain") or ""),
                    ):
                        self._screenshot_capture_stop_event.wait(capture_interval_seconds)
                        continue
                    if system_idle_service.is_user_idle():
                        self._screenshot_capture_stop_event.wait(capture_interval_seconds)
                        continue

                    screenshot_path = passive_observer.capture_screenshot()
                    queued = screenshot_queue.enqueue(
                        screenshot_path,
                        retain=autonomy_coordinator is None,
                    )
                    if queued is None:
                        logger.info(
                            "Skipped screenshot for passive observer due to SSIM similarity filter: %s (queue_size=%s)",
                            screenshot_path,
                            screenshot_queue.size(),
                        )
                    else:
                        if autonomy_coordinator is not None:
                            screenshot_ref = screenshot_path
                            if self.capture_store is not None:
                                screenshot_ref = self.capture_store.store_file(
                                    screenshot_path,
                                    kind="screenshot",
                                    delete_source=True,
                                )
                            ambient_event = autonomy_coordinator.enqueue_lightweight_visual(
                                screenshot_ref=screenshot_ref,
                                captured_at=queued.captured_at,
                                context=lightweight_context,
                                similarity_score=queued.similarity_score,
                            )
                        logger.info(
                            "Captured lightweight screen context: %s (ambient_event=%s, backlog=%s)",
                            screenshot_ref if autonomy_coordinator is not None else queued.screenshot_path,
                            ambient_event.event_id if autonomy_coordinator is not None else "legacy_queue",
                            autonomy_coordinator.event_counts() if autonomy_coordinator is not None else screenshot_queue.size(),
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

    def _publish_chat_event(self, message_id: str, event: dict) -> None:
        if self.chat_event_broker is not None:
            self.chat_event_broker.publish(message_id, event)

    async def _process_pending_chat_turn(
        self,
        *,
        llm_adapter: ModelResidencyManager,
        llm_service: LLMInteractionService,
        services_initialized: bool,
    ) -> tuple[bool, bool]:
        if self.chat_store is None:
            return False, services_initialized
        if time.monotonic() < self._chat_resource_backoff_until:
            return False, services_initialized
        turn = self.chat_store.claim_next_turn()
        if turn is None:
            return False, services_initialized

        message_id = turn["id"]
        session_id = turn["session_id"]
        user_message = turn["user_message"]
        self._publish_chat_event(message_id, {"type": "status", "status": "running"})
        streamed_parts: list[str] = []
        last_checkpoint = time.monotonic()
        chat_model = LIGHTWEIGHT_CHAT_MODEL or CHAT_MODEL
        deferred_for_resources = False

        def on_event(event: dict) -> None:
            nonlocal last_checkpoint
            if event.get("type") == "delta":
                streamed_parts.append(str(event.get("content", "")))
                now = time.monotonic()
                if now - last_checkpoint >= CHAT_STREAM_CHECKPOINT_SECONDS:
                    self.chat_store.update_partial(message_id, "".join(streamed_parts))
                    last_checkpoint = now
            self._publish_chat_event(message_id, event)

        try:
            services_initialized = await self._ensure_runtime(
                llm_adapter=llm_adapter,
                services_initialized=services_initialized,
                reason="answering direct chat message",
                model_name=chat_model,
            )
            history = self.chat_store.conversation_history(
                session_id,
                before_message_id=user_message["id"],
                limit=CHAT_HISTORY_MESSAGE_LIMIT,
            )
            llm_service.restore_conversation(
                system_prompt=self.CHAT_SYSTEM_PROMPT,
                messages=history,
            )
            with interaction_trace(
                "direct_chat",
                {
                    "chat_session_id": session_id,
                    "chat_message_id": message_id,
                },
            ):
                with self.gpu_lock:
                    result = await llm_service.run_interaction(
                        user_input=user_message["content"],
                        system_prompt=self.CHAT_SYSTEM_PROMPT,
                        model=chat_model,
                        report_policy="silent",
                        event_callback=on_event,
                    )
            self.chat_store.complete_message(message_id, result)
            self._publish_chat_event(
                message_id,
                {
                    "type": "done",
                    "message": self.chat_store.get_message(message_id),
                },
            )
        except ResourceUnavailableError as exc:
            deferred_for_resources = True
            self._chat_resource_backoff_until = time.monotonic() + RESOURCE_DEFER_SECONDS
            self.chat_store.defer_message(message_id, f"Queued until resources are available: {exc}")
            self._publish_chat_event(
                message_id,
                {"type": "status", "status": "resource_deferred", "reason": str(exc)},
            )
        except Exception as exc:
            logger.exception("Direct chat message %s failed.", message_id)
            self.chat_store.fail_message(message_id, str(exc))
            self._publish_chat_event(
                message_id,
                {"type": "error", "error": str(exc)},
            )
        finally:
            llm_service.reset_context()
            if not deferred_for_resources:
                self._start_chat_response_window()
        return True, services_initialized

    async def _process_due_scheduled_task(
        self,
        *,
        llm_adapter: ModelResidencyManager,
        llm_service: LLMInteractionService,
        task_queue: SQLiteTaskQueueAdapter,
        services_initialized: bool,
    ) -> tuple[bool, bool]:
        now_utc = datetime.now(timezone.utc)
        due_tasks = task_queue.get_due_tasks(now_utc.isoformat())
        if not due_tasks:
            return False, services_initialized
        task = due_tasks[0]
        if not task_queue.claim_task(task.id):
            return False, services_initialized

        metadata = {}
        if task.metadata_json:
            try:
                metadata = json.loads(task.metadata_json)
            except json.JSONDecodeError:
                metadata = {}
        session_id = str(metadata.get("origin_chat_session_id") or "").strip()
        scheduled_at = datetime.fromisoformat(task.run_at_utc) if task.run_at_utc else now_utc
        lateness_seconds = max(0, int((now_utc - scheduled_at).total_seconds()))

        try:
            services_initialized = await self._ensure_runtime(
                llm_adapter=llm_adapter,
                services_initialized=services_initialized,
                reason="executing exact-time scheduled task",
                model_name=FOLLOWUP_EXECUTION_MODEL,
            )
            llm_service.reset_context()
            with interaction_trace(
                "scheduled_chat_task",
                {
                    "scheduled_task_id": task.id,
                    "chat_session_id": session_id or None,
                    "scheduled_lateness_seconds": lateness_seconds,
                },
            ):
                with self.gpu_lock:
                    result = await llm_service.run_interaction(
                        user_input=task.description,
                        system_prompt=self.FOLLOWUP_EXECUTION_PROMPT,
                        model=FOLLOWUP_EXECUTION_MODEL,
                        report_policy="auto_surface",
                    )
            task_queue.mark_task_complete(task.id)
            if self.chat_store is not None and session_id and self.chat_store.get_session(session_id):
                late_note = f" (started {lateness_seconds}s late)" if lateness_seconds > 5 else ""
                self.chat_store.append_scheduled_result(
                    session_id=session_id,
                    task_id=task.id,
                    content=f"Scheduled task completed{late_note}:\n\n{result}",
                )
        except Exception as exc:
            logger.exception("Scheduled task %s failed.", task.id)
            task_queue.mark_task_complete(task.id, status="failed")
            if self.chat_store is not None and session_id and self.chat_store.get_session(session_id):
                self.chat_store.append_scheduled_result(
                    session_id=session_id,
                    task_id=task.id,
                    content=f"Scheduled task failed: {exc}",
                    failed=True,
                )
        finally:
            llm_service.reset_context()
        return True, services_initialized

    def _enqueue_due_scheduled_tasks(
        self,
        *,
        task_queue: SQLiteTaskQueueAdapter,
        autonomy_coordinator: AutonomyCoordinatorService,
    ) -> int:
        """Move due exact-time work into the durable autonomy event stream."""
        now_utc = datetime.now(timezone.utc)
        delegated = 0
        for task in task_queue.get_due_tasks(now_utc.isoformat()):
            if not task_queue.claim_task(task.id):
                continue
            try:
                autonomy_coordinator.enqueue_scheduled_task(
                    task_id=task.id,
                    description=task.description,
                    run_at_utc=task.run_at_utc or now_utc.isoformat(),
                    metadata_json=task.metadata_json,
                )
            except Exception:
                task_queue.mark_task_complete(task.id, status="failed_event_enqueue")
                raise
            else:
                task_queue.mark_task_complete(task.id, status="delegated_to_autonomy")
                delegated += 1
        return delegated

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
            autonomy_store,
            autonomy_coordinator,
        ) = self._build_services()
        idle_cycle_interval = 30
        passive_observer_interval = PASSIVE_OBSERVER_CAPTURE_INTERVAL_SECONDS
        last_idle_cycle_at = 0.0
        user_idle_now = False
        services_initialized = False

        try:
            self.stop_event.clear()
            self._tool_bridge = tool_bridge
            await self._initialize_mcp_tools(tool_bridge, llm_service)
            services_initialized = await self._restore_chat_residency(
                llm_adapter=llm_adapter,
                services_initialized=services_initialized,
                reason="keeping direct chat ready at runtime startup",
                open_response_window=False,
            )
            self._start_screenshot_capture_loop(
                passive_observer=passive_observer,
                screenshot_queue=screenshot_queue,
                system_idle_service=system_idle_service,
                capture_interval_seconds=passive_observer_interval,
                autonomy_coordinator=autonomy_coordinator,
            )
            logger.info("Starting ambient runtime manager.")

            while not self.stop_event.is_set():
                current_user_idle = system_idle_service.is_user_idle()
                if current_user_idle != user_idle_now:
                    user_idle_now = current_user_idle
                    if user_idle_now:
                        logger.info(
                            "User idle detected (>= %ss); background work may use the available resource window.",
                            USER_IDLE_THRESHOLD_SECONDS,
                        )
                        last_idle_cycle_at = 0.0
                    else:
                        logger.info("User activity detected; context judgment remains active.")

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
                    services_initialized = await self._restore_chat_residency(
                        llm_adapter=llm_adapter,
                        services_initialized=services_initialized,
                        reason="ASR pipeline finished",
                    )

                handled_chat, services_initialized = await self._process_pending_chat_turn(
                    llm_adapter=llm_adapter,
                    llm_service=llm_service,
                    services_initialized=services_initialized,
                )
                if handled_chat:
                    if await self._sleep_or_stop(0.1):
                        break
                    continue

                if self._chat_response_window_active():
                    if await self._sleep_or_stop(0.25):
                        break
                    continue

                autonomy_backlog_ready = bool(
                    autonomy_coordinator is not None and autonomy_coordinator.has_ready_work()
                )
                if autonomy_backlog_ready:
                    # Background work owns residency until its ready backlog drains.
                    # Restoring chat here would steal headroom and cause load/unload thrash.
                    services_initialized = bool(llm_adapter.status().get("loaded_model"))
                    if services_initialized:
                        self.llm_active_event.set()
                    else:
                        self.llm_active_event.clear()
                else:
                    services_initialized = await llm_adapter.settle_to_lightweight(
                        user_active=not user_idle_now
                    )
                    if services_initialized:
                        self.llm_active_event.set()
                    else:
                        self.llm_active_event.clear()

                if autonomy_coordinator is not None:
                    delegated = self._enqueue_due_scheduled_tasks(
                        task_queue=task_queue,
                        autonomy_coordinator=autonomy_coordinator,
                    )
                    if delegated:
                        logger.info("Delegated %s due scheduled task(s) to the autonomy coordinator.", delegated)
                else:
                    handled_scheduled, services_initialized = await self._process_due_scheduled_task(
                        llm_adapter=llm_adapter,
                        llm_service=llm_service,
                        task_queue=task_queue,
                        services_initialized=services_initialized,
                    )
                    if handled_scheduled:
                        services_initialized = await self._restore_chat_residency(
                            llm_adapter=llm_adapter,
                            services_initialized=services_initialized,
                            reason="scheduled task work unit finished",
                        )
                        if await self._sleep_or_stop(0.1):
                            break
                        continue

                try:
                    transcript_path = self.queue.get_nowait()
                except queue.Empty:
                    transcript_path = None
                if transcript_path is not None:
                    if autonomy_coordinator is not None:
                        try:
                            transcript_text = Path(transcript_path).read_text(encoding="utf-8").strip()
                            if transcript_text:
                                autonomy_coordinator.enqueue_transcript(
                                    transcript_path=str(transcript_path),
                                    transcript_text=transcript_text,
                                )
                                logger.info("Transcript %s added to continuous context evaluation.", transcript_path)
                        except Exception:
                            logger.exception("Failed to enqueue transcript %s for autonomy judgment.", transcript_path)
                    elif ALWAYS_ON_MODE:
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
                    if ALWAYS_ON_MODE:
                        services_initialized = await self._restore_chat_residency(
                            llm_adapter=llm_adapter,
                            services_initialized=services_initialized,
                            reason="transcript work unit finished",
                        )
                        if await self._sleep_or_stop(0.1):
                            break
                        continue

                now = time.monotonic()
                runtime_active_now = user_idle_now or ALWAYS_ON_MODE

                if not AUTONOMY_COORDINATOR_ENABLED and runtime_active_now and todoist_provider is not None:
                    try:
                        explicit_tasks = todoist_provider.get_tasks()
                    except Exception:
                        logger.exception("Failed to fetch Todoist tasks.")
                        explicit_tasks = []
                    if explicit_tasks:
                        task = explicit_tasks[0]
                        logger.info(
                            "Executing user-given Todoist task while idle: %s",
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
                            logger.info("Todoist task %s completed", task.get("id"))
                            todoist_provider.complete_task(task["id"])
                        except Exception:
                            logger.exception("Todoist task execution failed.")
                        last_idle_cycle_at = now
                        services_initialized = await self._restore_chat_residency(
                            llm_adapter=llm_adapter,
                            services_initialized=services_initialized,
                            reason="Todoist work unit finished",
                        )
                        if await self._sleep_or_stop(0.1):
                            break
                        continue

                if (
                    PASSIVE_OBSERVER_ENABLED
                    and passive_observer is not None
                    and screenshot_queue is not None
                    and not screenshot_queue.is_empty()
                ):
                    job = screenshot_queue.dequeue()
                    if job is not None and Path(job.screenshot_path).exists():
                        try:
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
                                    self._deferred_followup_observations.append(observation)
                                if autonomy_coordinator is not None:
                                    autonomy_coordinator.enqueue_visual_observation(observation)
                        except Exception:
                            logger.exception("Queued passive observer screenshot processing failed.")
                        services_initialized = await self._restore_chat_residency(
                            llm_adapter=llm_adapter,
                            services_initialized=services_initialized,
                            reason="screenshot work unit finished",
                        )
                        if await self._sleep_or_stop(0.1):
                            break
                        continue
                    if job is not None:
                        logger.warning("Queued screenshot no longer exists, skipping: %s", job.screenshot_path)

                if autonomy_coordinator is not None and autonomy_coordinator.has_ready_work():
                    inference_request = InferenceRequest(
                        workload="ambient_inference_batch",
                        model_name=FOLLOWUP_EXECUTION_MODEL,
                        background=True,
                        user_active=not user_idle_now,
                        priority=60,
                    )
                    lease = self.resource_governor.request_lease(inference_request)
                    if not lease.acquired:
                        deferred = autonomy_coordinator.defer_next(
                            reason=lease.decision.reason,
                            delay_seconds=RESOURCE_DEFER_SECONDS,
                        )
                        await llm_adapter.evict_if_critical()
                        if deferred and await self._sleep_or_stop(0.25):
                            break
                    else:
                        try:
                            with interaction_trace(
                                "autonomy_resource_batch",
                                {"user_idle": user_idle_now, "resource_preset": self.resource_governor.preset},
                            ):
                                with self.gpu_lock:
                                    autonomy_result = await autonomy_coordinator.process_batch(
                                        model=FOLLOWUP_EXECUTION_MODEL,
                                        llm_service=llm_service,
                                        personalization_context=memory_store.get_recent_context(),
                                        max_events=RESOURCE_BATCH_MAX_EVENTS,
                                        max_seconds=RESOURCE_BATCH_MAX_SECONDS,
                                        should_preempt=(
                                            chat_store.has_queued_turn if chat_store is not None else None
                                        ),
                                    )
                            if autonomy_result.get("processed"):
                                logger.info(
                                    "Autonomy coordinator processed %s event(s) in %.2fs.",
                                    autonomy_result.get("count"),
                                    autonomy_result.get("elapsed_seconds", 0.0),
                                )
                        finally:
                            lease.__exit__(None, None, None)
                            backlog_remaining = autonomy_coordinator.has_ready_work()
                            chat_waiting = bool(
                                chat_store is not None and chat_store.has_queued_turn()
                            )
                            if backlog_remaining and not chat_waiting:
                                services_initialized = bool(
                                    llm_adapter.status().get("loaded_model")
                                )
                                if services_initialized:
                                    self.llm_active_event.set()
                                    logger.info(
                                        "Keeping %s resident for remaining ambient backlog %s.",
                                        llm_adapter.status().get("loaded_model"),
                                        autonomy_coordinator.event_counts(),
                                    )
                            else:
                                services_initialized = await self._release_runtime(
                                    llm_adapter=llm_adapter,
                                    services_initialized=bool(llm_adapter.status().get("loaded_model")),
                                    reason="ambient backlog drained or chat preempted",
                                )
                                services_initialized = await self._restore_chat_residency(
                                    llm_adapter=llm_adapter,
                                    services_initialized=services_initialized,
                                    reason="ambient backlog drained or chat preempted",
                                    open_response_window=False,
                                )
                        if autonomy_result.get("processed"):
                            if await self._sleep_or_stop(0.1):
                                break
                            continue

                if autonomy_coordinator is None and runtime_active_now and now - last_idle_cycle_at >= idle_cycle_interval:
                    idle_unit_attempted = False
                    idle_unit_completed = False
                    try:
                        if self._deferred_followup_observations and passive_followup is not None:
                            observation = self._deferred_followup_observations.popleft()
                            idle_unit_attempted = True
                            services_initialized = await self._ensure_runtime(
                                llm_adapter=llm_adapter,
                                services_initialized=services_initialized,
                                reason="processing deferred always-on passive follow-up",
                                model_name=PASSIVE_FOLLOWUP_MODEL,
                            )
                            with self.gpu_lock:
                                followup_result = await passive_followup.process_observations(
                                    observations=[observation],
                                    model=PASSIVE_FOLLOWUP_MODEL,
                                    mark_sent=False,
                                    apply_memory_updates=False,
                                )
                            self._deferred_followup_activities.extend(
                                str(activity)
                                for activity in followup_result.get("do_now_activities", [])
                                if str(activity).strip()
                            )
                            idle_unit_completed = True
                            logger.info(
                                "Always-on follow-up processed observation %s with %s queued and %s do-now activities.",
                                observation.observation_id,
                                len(followup_result.get("queued_activities", [])),
                                len(followup_result.get("do_now_activities", [])),
                            )

                        if not idle_unit_attempted and self._deferred_followup_activities:
                            activity = self._deferred_followup_activities.popleft()
                            idle_unit_attempted = True
                            logger.info("Executing deferred passive follow-up activity: %s", activity)
                            services_initialized = await self._ensure_runtime(
                                llm_adapter=llm_adapter,
                                services_initialized=services_initialized,
                                reason="executing deferred passive follow-up activity",
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
                                idle_unit_completed = True
                                logger.info("Deferred do-now activity completed with result: %s", result[:500])
                            finally:
                                llm_service.reset_context()

                        if (
                            not idle_unit_attempted
                            and passive_followup is not None
                            and not ALWAYS_ON_MODE
                        ):
                            services_initialized = await self._ensure_runtime(
                                llm_adapter=llm_adapter,
                                services_initialized=services_initialized,
                                reason="running passive observer follow-up",
                                model_name=PASSIVE_FOLLOWUP_MODEL,
                            )
                            with self.gpu_lock:
                                followup_result = await passive_followup.maybe_queue_followup(
                                    model=PASSIVE_FOLLOWUP_MODEL
                                )
                            self._deferred_followup_activities.extend(
                                str(activity)
                                for activity in followup_result.get("do_now_activities", [])
                                if str(activity).strip()
                            )
                            idle_unit_attempted = bool(
                                followup_result.get("processed_observation_ids")
                                or followup_result.get("unique_activities")
                                or followup_result.get("queued_activities")
                                or followup_result.get("do_now_activities")
                            )
                            idle_unit_completed = idle_unit_attempted
                            logger.info(
                                "Passive observer follow-up processed %s observations, %s unique activities, %s useful activities, %s queued, %s do-now.",
                                len(followup_result.get("processed_observation_ids", [])),
                                len(followup_result.get("unique_activities", [])),
                                len(followup_result.get("useful_activities", [])),
                                len(followup_result.get("queued_activities", [])),
                                len(followup_result.get("do_now_activities", [])),
                            )

                        queue_task_execution_allowed = user_idle_now or PERFORM_QUEUE_TASKS
                        pending_tasks = (
                            task_queue.get_pending_tasks()
                            if not idle_unit_attempted and queue_task_execution_allowed
                            else []
                        )
                        if pending_tasks:
                            idle_unit_attempted = True
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
                                    idle_unit_completed = True
                                else:
                                    raise
                            else:
                                idle_unit_completed = True
                                logger.info("Queued follow-up task %s completed with result: %s", task.id, result[:500])
                                task_queue.mark_task_complete(task.id)
                                self._mark_dedupe_item_completed(semantic_dedupe, task)
                            finally:
                                llm_service.reset_context()

                        if (
                            not idle_unit_attempted
                            and reflection_service is not None
                            and not ALWAYS_ON_MODE
                        ):
                            services_initialized = await self._ensure_runtime(
                                llm_adapter=llm_adapter,
                                services_initialized=services_initialized,
                                reason="running reflection service",
                                model_name=REFLECTION_MODEL,
                            )
                            with self.gpu_lock:
                                reflection_result = await reflection_service.run_if_due(model=REFLECTION_MODEL)
                            idle_unit_attempted = bool(reflection_result.get("ran"))
                            idle_unit_completed = idle_unit_attempted
                            if idle_unit_attempted:
                                logger.info(
                                    "Reflection service ran: cleaned_changed=%s, generated=%s, queued=%s, skipped=%s.",
                                    reflection_result.get("cleaned_user_info_changed"),
                                    len(reflection_result.get("generated_tasks", [])),
                                    len(reflection_result.get("queued_tasks", [])),
                                    len(reflection_result.get("skipped_tasks", [])),
                                )

                        if (
                            not idle_unit_attempted
                            and user_biodata_service is not None
                            and not ALWAYS_ON_MODE
                        ):
                            services_initialized = await self._ensure_runtime(
                                llm_adapter=llm_adapter,
                                services_initialized=services_initialized,
                                reason="updating user biodata from passive observations",
                                model_name=USER_BIODATA_MODEL,
                            )
                            with self.gpu_lock:
                                biodata_result = await user_biodata_service.update_biodata(
                                    model=USER_BIODATA_MODEL
                                )
                            idle_unit_attempted = bool(
                                biodata_result.get("processed_observation_ids")
                                or biodata_result.get("entries")
                            )
                            idle_unit_completed = idle_unit_attempted
                            logger.info(
                                "User BioData update processed %s observations and appended %s entries.",
                                len(biodata_result.get("processed_observation_ids", [])),
                                len(biodata_result.get("entries", [])),
                            )
                    except Exception:
                        idle_unit_attempted = True
                        logger.exception("Idle follow-up work unit failed.")
                    last_idle_cycle_at = now

                    services_initialized = await self._restore_chat_residency(
                        llm_adapter=llm_adapter,
                        services_initialized=services_initialized,
                        reason="idle background work check finished",
                        open_response_window=idle_unit_attempted,
                    )
                    if idle_unit_attempted or idle_unit_completed:
                        if await self._sleep_or_stop(0.1):
                            break
                        continue

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
            await llm_service.cleanup_browser_sessions()
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
    model_server_ready, model_server_status = check_model_server()
    if model_server_ready:
        logger.info("Manual llama.cpp router is reachable at %s (%s).", API_BASE_URL, model_server_status)
    else:
        logger.error(
            "Manual llama.cpp router is unavailable at %s (%s). Capture will continue, but all model inference will remain deferred until the router is started.",
            API_BASE_URL,
            model_server_status,
        )
    capture_store = PlainCaptureStore(str(CAPTURE_STORAGE_ROOT))
    capture_control = CaptureControlService(
        excluded_apps=PASSIVE_OBSERVER_IGNORE_APPS,
        excluded_domains=PASSIVE_OBSERVER_IGNORE_DOMAINS,
    )
    autonomy_api_store = SQLiteAutonomyAdapter(str(AUTONOMY_DB_PATH))
    resource_governor = ResourceGovernorService(
        monitor=WindowsResourceMonitor(),
        preset=RESOURCE_PRESET,
        critical_ram_mb=RESOURCE_CRITICAL_RAM_MB,
        critical_ram_percent=RESOURCE_CRITICAL_RAM_PERCENT,
        critical_vram_mb=RESOURCE_CRITICAL_VRAM_MB,
        audit=autonomy_api_store.audit,
    )
    chat_store = SQLiteChatAdapter(db_path=str(CHAT_DB_PATH))
    recovered_chat_messages = chat_store.recover_interrupted()
    if recovered_chat_messages:
        logger.warning("Marked %s interrupted chat response(s) as failed.", recovered_chat_messages)
    chat_event_broker = ChatEventBroker()
    recovered_scheduled_tasks = SQLiteTaskQueueAdapter().recover_interrupted()
    if recovered_scheduled_tasks:
        logger.warning("Marked %s interrupted scheduled task(s) as failed.", recovered_scheduled_tasks)
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
            chat_store=chat_store,
            chat_event_broker=chat_event_broker,
            autonomy_store=autonomy_api_store,
            capture_store=capture_store,
            capture_control=capture_control,
            resource_governor=resource_governor,
        )
        logger.info(
            "Runtime log server started at %s://%s:%s/logs",
            "http",
            LOG_API_HOST,
            LOG_API_PORT,
        )
        runtime_log_started = True

    gpu_lock = threading.Lock()
    audio_active_event = threading.Event()
    llm_active_event = threading.Event()

    audio_agent = AudioAgentService(
        gpu_lock=gpu_lock,
        audio_active_event=audio_active_event,
        llm_active_event=llm_active_event,
        capture_store=capture_store,
        capture_control=capture_control,
        autonomy_store=autonomy_api_store,
        resource_governor=resource_governor,
        deferred_asr_max_audio_seconds=DEFERRED_ASR_MAX_AUDIO_SECONDS,
    )
    transcription_queue = audio_agent.get_transcription_queue()

    ambient_runtime = AmbientRuntime(
        transcription_queue=transcription_queue,
        gpu_lock=gpu_lock,
        audio_active_event=audio_active_event,
        llm_active_event=llm_active_event,
        chat_store=chat_store,
        chat_event_broker=chat_event_broker,
        capture_store=capture_store,
        capture_control=capture_control,
        resource_governor=resource_governor,
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
