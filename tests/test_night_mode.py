import asyncio
from datetime import datetime
import queue
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from application.services.activity_ledger_service import ActivityLedgerService
from application.services.night_mode_service import NightModeService
from infrastructure.adapter.SQLiteActivityLedgerAdapter import SQLiteActivityLedgerAdapter

sys.modules.setdefault(
    "audio_agent",
    types.SimpleNamespace(AudioAgentService=object),
)
from app import TranscriptionService


class _FakeTask:
    def __init__(self, task_id: str, description: str):
        self.id = task_id
        self.description = description


class _FakeTaskQueue:
    def __init__(self):
        self.pending = []
        self.completed = []

    def get_pending_tasks(self):
        return list(self.pending)

    def mark_task_complete(self, task_id):
        self.completed.append(task_id)


class _FakeTaskProvider:
    def __init__(self, tasks=None):
        self.tasks = list(tasks or [])
        self.completed = []

    def get_tasks(self):
        return list(self.tasks)

    def complete_task(self, task_id):
        self.completed.append(task_id)


class _FakeNotificationPort:
    def get_unread_notifications(self):
        return []


class _FakeLLMService:
    def __init__(self, should_fail: bool = False):
        self.calls = []
        self.should_fail = should_fail

    async def run_interaction(self, **kwargs):
        self.calls.append(kwargs)
        if self.should_fail:
            raise RuntimeError("llm failed")
        return {"ok": True}


class _FakeMemoryConsolidator:
    def __init__(self):
        self.calls = 0
        self.memory = self

    def consolidate(self):
        self.calls += 1
        return 1

    def get_recent_context(self):
        return "recent context"


class _FakeProactiveResearchService:
    def __init__(self):
        self.calls = 0

    async def process_topics(self, **kwargs):
        self.calls += 1
        return 2


class _FakeAction:
    def __init__(self, action_type: str):
        self.action_type = action_type


class _FakeAmbientReflectionService:
    def __init__(self):
        self.calls = 0

    async def reflect(self, **kwargs):
        self.calls += 1
        return [_FakeAction("SURFACE_ITEM")]


class _FakeAdapter:
    def __init__(self):
        self.loaded = 0
        self.unloaded = 0

    async def load_model(self, _model_name):
        self.loaded += 1

    async def unload_model(self):
        self.unloaded += 1


class _FakeToolBridge:
    def __init__(self):
        self.started = 0
        self.cleaned = 0

    async def start_servers(self, _config_path):
        self.started += 1

    async def cleanup(self):
        self.cleaned += 1


class _FakeLLMInitService:
    def __init__(self):
        self.initialized = 0

    async def initialize_tools(self):
        self.initialized += 1


class NightModeTests(unittest.TestCase):
    def test_night_mode_cycle_processes_bounded_work(self):
        task_queue = _FakeTaskQueue()
        task_queue.pending = [_FakeTask("internal-1", "Finish draft")]
        task_provider = _FakeTaskProvider(
            tasks=[{"id": "external-1", "content": "Todoist task"}]
        )
        llm_service = _FakeLLMService()
        memory_consolidator = _FakeMemoryConsolidator()
        proactive_research = _FakeProactiveResearchService()
        ambient_reflection = _FakeAmbientReflectionService()
        with tempfile.TemporaryDirectory() as tmpdir:
            activity_ledger = ActivityLedgerService(
                SQLiteActivityLedgerAdapter(db_path=str(Path(tmpdir) / "activity_ledger.db"))
            )

            service = NightModeService(
                task_queue=task_queue,
                task_provider=task_provider,
                notification_port=_FakeNotificationPort(),
                llm_service=llm_service,
                memory_consolidator=memory_consolidator,
                proactive_research_service=proactive_research,
                ambient_reflection_service=ambient_reflection,
                activity_ledger=activity_ledger,
                model="test-model",
            )

            result = asyncio.run(service.run_night_cycle())

            runs = activity_ledger.ledger.list_runs(limit=10)
            self.assertGreaterEqual(len(runs), 3)

        self.assertEqual(result["processed_internal_tasks"], 1)
        self.assertEqual(result["processed_external_tasks"], 1)
        self.assertEqual(result["processed_research_topics"], 2)
        self.assertEqual(result["reflection_actions"], 1)
        self.assertEqual(task_queue.completed, ["internal-1"])
        self.assertEqual(task_provider.completed, ["external-1"])
        self.assertEqual(len(llm_service.calls), 2)
        self.assertIn("Todoist task", llm_service.calls[0]["user_input"])
        self.assertIn("Finish draft", llm_service.calls[1]["user_input"])

    def test_external_task_failure_marks_run_failed_and_does_not_complete_task(self):
        task_queue = _FakeTaskQueue()
        task_provider = _FakeTaskProvider(tasks=[{"id": "external-1", "content": "Todoist task"}])
        llm_service = _FakeLLMService(should_fail=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            activity_ledger = ActivityLedgerService(
                SQLiteActivityLedgerAdapter(db_path=str(Path(tmpdir) / "activity_ledger.db"))
            )
            service = NightModeService(
                task_queue=task_queue,
                task_provider=task_provider,
                notification_port=_FakeNotificationPort(),
                llm_service=llm_service,
                activity_ledger=activity_ledger,
                model="test-model",
            )

            processed = asyncio.run(service.run_external_task_cycle())

            self.assertEqual(processed, 0)
            self.assertEqual(task_provider.completed, [])
            runs = activity_ledger.ledger.list_runs(source_kind="ambient_ai_task")
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0].status, "failed")

    def test_external_task_cycle_only_processes_ambient_ai_tasks(self):
        task_queue = _FakeTaskQueue()
        task_queue.pending = [_FakeTask("internal-1", "Finish draft")]
        task_provider = _FakeTaskProvider(
            tasks=[{"id": "external-1", "content": "Todoist task"}]
        )
        llm_service = _FakeLLMService()

        service = NightModeService(
            task_queue=task_queue,
            task_provider=task_provider,
            notification_port=_FakeNotificationPort(),
            llm_service=llm_service,
            model="test-model",
        )

        processed = asyncio.run(service.run_external_task_cycle())

        self.assertEqual(processed, 1)
        self.assertEqual(task_provider.completed, ["external-1"])
        self.assertEqual(task_queue.completed, [])
        self.assertEqual(len(llm_service.calls), 1)
        self.assertIn("Todoist task", llm_service.calls[0]["user_input"])

    def test_night_mode_window_is_after_midnight_and_before_six_am(self):
        service = TranscriptionService(
            transcription_queue=queue.Queue(),
            gpu_lock=threading.Lock(),
            audio_active_event=threading.Event(),
            llm_active_event=threading.Event(),
        )

        self.assertTrue(service._is_night_mode_window(datetime(2026, 6, 25, 0, 0, 0)))
        self.assertTrue(service._is_night_mode_window(datetime(2026, 6, 25, 3, 30, 0)))
        self.assertFalse(service._is_night_mode_window(datetime(2026, 6, 25, 6, 0, 0)))
        self.assertFalse(service._is_night_mode_window(datetime(2026, 6, 25, 23, 59, 0)))

    def test_ambient_runtime_can_load_and_unload_cleanly(self):
        service = TranscriptionService(
            transcription_queue=queue.Queue(),
            gpu_lock=threading.Lock(),
            audio_active_event=threading.Event(),
            llm_active_event=threading.Event(),
        )
        adapter = _FakeAdapter()
        tool_bridge = _FakeToolBridge()
        llm_service = _FakeLLMInitService()

        initialized = asyncio.run(
            service._ensure_ambient_runtime(
                llm_adapter=adapter,
                tool_bridge=tool_bridge,
                llm_service=llm_service,
                services_initialized=False,
                reason="test load",
            )
        )
        self.assertTrue(initialized)
        self.assertEqual(adapter.loaded, 1)
        self.assertEqual(tool_bridge.started, 1)
        self.assertEqual(llm_service.initialized, 1)
        self.assertTrue(service.llm_active_event.is_set())

        initialized = asyncio.run(
            service._release_ambient_runtime(
                llm_adapter=adapter,
                tool_bridge=tool_bridge,
                services_initialized=initialized,
                reason="test unload",
            )
        )
        self.assertFalse(initialized)
        self.assertEqual(adapter.unloaded, 1)
        self.assertEqual(tool_bridge.cleaned, 1)
        self.assertFalse(service.llm_active_event.is_set())


if __name__ == "__main__":
    unittest.main()
