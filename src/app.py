import asyncio
import logging
from pathlib import Path
import queue
import threading
import time
from typing import Optional

from application.services.passive_observer_followup_service import PassiveObserverFollowupService
from application.services.passive_observer_service import PassiveObserverService
from application.services.screenshot_queue_service import ScreenshotQueueService
from application.services.system_idle_service import SystemIdleService
from audio_agent import AudioAgentService
from infrastructure.adapter.llamaCppAdapter import LlamaCppAdapter
from infrastructure.adapter.LoggingLLMProvider import LoggingLLMProvider
from infrastructure.adapter.MSSScreenCaptureAdapter import MssScreenCaptureAdapter
from infrastructure.adapter.SQLiteInteractionLogAdapter import SQLiteInteractionLogAdapter
from infrastructure.adapter.SQLiteMemoryAdapter import SQLiteMemoryAdapter
from infrastructure.adapter.SQLiteTaskQueueAdapter import SQLiteTaskQueueAdapter
from infrastructure.adapter.SQLiteVoiceAdapter import SQLiteVoiceAdapter
from infrastructure.runtime_log_server import (
    configure_runtime_log_streaming,
    start_runtime_log_server,
)
from config import CONFIG
import night_mode


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

API_BASE_URL = CONFIG.get_str("runtime", "api_base_url", "http://localhost:8080")
DEFAULT_MODEL = CONFIG.get_str("runtime", "default_model", "Qwen-3.5-9B-Mythos-Distilled-Q4_K_M-Vision")
USER_DATA_DIR = Path(CONFIG.get_str("runtime", "user_data_dir", "D:\\USER_DATA"))
PROJECT_ROOT = Path(__file__).parent.parent
MEMORY_ROOT = USER_DATA_DIR / "memory"
MEMORY_DB_PATH = USER_DATA_DIR / "database" / "memory.db"
INTERACTION_LOG_DB_PATH = USER_DATA_DIR / "database" / "interaction_logs.db"
CURRENT_RESPONSE_PATH = USER_DATA_DIR / "database" / "current_llm_response.md"
VOICE_DB_PATH = USER_DATA_DIR / "database" / "voice_database.db"
PASSIVE_OBSERVER_ROOT = USER_DATA_DIR / "passive_observer"
PASSIVE_OBSERVER_ENABLED = CONFIG.get_bool("passive_observer", "enabled", False)
USER_IDLE_THRESHOLD_SECONDS = CONFIG.get_int("runtime", "user_idle_threshold_seconds", 20)
SCREENSHOT_QUEUE_MAXLEN = CONFIG.get_int("passive_observer", "screenshot_queue_maxlen", 180)
PASSIVE_OBSERVER_SSIM_THRESHOLD = CONFIG.get_float("passive_observer", "ssim_threshold", 0.92)
PASSIVE_OBSERVER_SSIM_COMPARE_COUNT = CONFIG.get_int("passive_observer", "ssim_compare_count", 4)
MAX_SCREENSHOTS_PER_IDLE_CYCLE = CONFIG.get_int("passive_observer", "max_screenshots_per_idle_cycle", 6)
LOG_API_ENABLED = CONFIG.get_bool("log_api", "enabled", True)
LOG_API_HOST = CONFIG.get_str("log_api", "host", "0.0.0.0")
LOG_API_PORT = CONFIG.get_int("log_api", "port", 8765)
LOG_API_BUFFER_SIZE = CONFIG.get_int("log_api", "buffer_size", 2000)

configure_runtime_log_streaming(max_entries=LOG_API_BUFFER_SIZE)


def ensure_runtime_databases() -> None:
    MEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    VOICE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    SQLiteMemoryAdapter(db_path=str(MEMORY_DB_PATH), memory_root=str(MEMORY_ROOT))
    SQLiteInteractionLogAdapter(db_path=str(INTERACTION_LOG_DB_PATH))
    SQLiteVoiceAdapter(str(VOICE_DB_PATH))
    night_mode.init_db()


class AmbientRuntime:
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

    async def _ensure_llm_ready(self, llm_adapter: LlamaCppAdapter) -> None:
        await llm_adapter.load_model(DEFAULT_MODEL)
        self.llm_active_event.set()

    async def _release_llm(self, llm_adapter: LlamaCppAdapter) -> None:
        await llm_adapter.unload_model()
        self.llm_active_event.clear()

    async def _ensure_runtime(
        self,
        *,
        llm_adapter: LlamaCppAdapter,
        services_initialized: bool,
        reason: str,
    ) -> bool:
        if services_initialized:
            return True
        logger.info("Loading ambient runtime: %s", reason)
        await self._ensure_llm_ready(llm_adapter)
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
        llm_adapter = LlamaCppAdapter(base_url=API_BASE_URL)
        interaction_log_store = SQLiteInteractionLogAdapter(
            db_path=str(INTERACTION_LOG_DB_PATH),
        )
        logged_llm = LoggingLLMProvider(
            provider=llm_adapter,
            log_store=interaction_log_store,
            current_response_path=str(CURRENT_RESPONSE_PATH),
        )
        memory_store = SQLiteMemoryAdapter(
            db_path=str(MEMORY_DB_PATH),
            memory_root=str(MEMORY_ROOT),
        )
        task_queue = SQLiteTaskQueueAdapter()
        passive_followup = (
            PassiveObserverFollowupService(
                memory=memory_store,
                task_queue=task_queue,
                llm_provider=logged_llm,
                activity_ledger=None,
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
            )
            if PASSIVE_OBSERVER_ENABLED
            else None
        )
        return (
            llm_adapter,
            memory_store,
            passive_followup,
            system_idle_service,
            screenshot_queue,
            passive_observer,
        )

    async def run_loop(self):
        (
            llm_adapter,
            memory_store,
            passive_followup,
            system_idle_service,
            screenshot_queue,
            passive_observer,
        ) = self._build_services()
        idle_cycle_interval = 30
        passive_observer_interval = 10
        last_idle_cycle_at = 0.0
        last_passive_observer_at = 0.0
        user_idle_now = False
        services_initialized = False

        try:
            logger.info("Starting reduced ambient runtime manager.")

            while True:
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
                    while self.audio_active_event.is_set():
                        await asyncio.sleep(0.5)
                    last_idle_cycle_at = 0.0

                while True:
                    try:
                        transcript_path = self.queue.get_nowait()
                    except queue.Empty:
                        break
                    logger.info(
                        "Transcript produced at %s. Downstream transcript reasoning has been removed from the runtime.",
                        transcript_path,
                    )
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
                    while processed_screenshots < MAX_SCREENSHOTS_PER_IDLE_CYCLE and not screenshot_queue.is_empty():
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
                            services_initialized = await self._ensure_runtime(
                                llm_adapter=llm_adapter,
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
                        if passive_followup is not None:
                            services_initialized = await self._ensure_runtime(
                                llm_adapter=llm_adapter,
                                services_initialized=services_initialized,
                                reason="running passive observer follow-up",
                            )
                            with self.gpu_lock:
                                followup_result = await passive_followup.maybe_queue_followup(model=DEFAULT_MODEL)
                            if followup_result.get("action") == "queue_task":
                                logger.info(
                                    "Passive observer queued deferred task: %s",
                                    followup_result.get("title", ""),
                                )
                            else:
                                logger.info("Passive observer idle follow-up found no task to queue.")
                    except Exception:
                        logger.exception("Idle follow-up cycle failed.")
                    last_idle_cycle_at = now

                if not user_idle_now:
                    services_initialized = await self._release_runtime(
                        llm_adapter=llm_adapter,
                        services_initialized=services_initialized,
                        reason="user is active; keep VRAM clear",
                    )

                await asyncio.sleep(1)
        finally:
            if services_initialized:
                await self._release_runtime(
                    llm_adapter=llm_adapter,
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
    if LOG_API_ENABLED:
        start_runtime_log_server(
            host=LOG_API_HOST,
            port=LOG_API_PORT,
            max_entries=LOG_API_BUFFER_SIZE,
        )
        logger.info("Runtime log server started at http://%s:%s/logs", LOG_API_HOST, LOG_API_PORT)

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
        daemon=True,
        name="AudioPipelineThread",
    )
    runtime_thread = threading.Thread(
        target=ambient_runtime.start_service,
        daemon=True,
        name="AmbientRuntimeThread",
    )

    audio_thread.start()
    runtime_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Main thread shutting down.")
