import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from application.services.activity_ledger_service import ActivityLedgerService
from application.services.interaction_trace import interaction_trace
from core.models import TranscriptClassificationResult
from infrastructure.adapter.LoggingLLMProvider import LoggingLLMProvider
from infrastructure.adapter.SQLiteActivityLedgerAdapter import SQLiteActivityLedgerAdapter
from infrastructure.adapter.SQLiteInteractionLogAdapter import SQLiteInteractionLogAdapter
from server import create_app


class _FakeDelta:
    def __init__(self, content=None):
        self.content = content
        self.reasoning_content = None
        self.tool_calls = None


class _FakeChoice:
    def __init__(self, delta):
        self.delta = delta


class _FakeChunk:
    def __init__(self, content=None):
        self.choices = [_FakeChoice(_FakeDelta(content=content))]


class _FakeLLMProvider:
    async def chat_completion_stream(self, model, messages, tools=None, image="", temperature=0.7, top_p=0.95, top_k=0):
        async def _gen():
            yield _FakeChunk(content="trace body")
        return _gen()

    async def load_model(self, model_name: str):
        return None

    async def save_and_unload(self, messages):
        return None

    async def load_and_restore(self):
        return None


class _FakeTaskQueue:
    def __init__(self):
        self.items = []

    def add_task(self, description: str, priority: str = "medium") -> str:
        self.items.append((description, priority))
        return "queued"


class _FakeNotifications:
    def __init__(self):
        self.messages = []

    def add_notification(self, message: str, source: str = "system") -> None:
        self.messages.append((message, source))


class _FakeTopicDetector:
    def maybe_queue_topic(self, classification, transcript_text: str, source_ref: str):
        return None


class _FakeMemoryStore:
    def append_event(self, event):
        return None


class _FakeSimpleExecutor:
    async def execute(self, **kwargs):
        return "unused"


class ActivityLedgerTests(unittest.TestCase):
    def test_storage_search_and_trace_linking(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            interaction_store = SQLiteInteractionLogAdapter(str(temp_path / "interaction_logs.db"))
            ledger = SQLiteActivityLedgerAdapter(
                db_path=str(temp_path / "activity_ledger.db"),
                interaction_log_db_path=str(temp_path / "interaction_logs.db"),
            )
            service = ActivityLedgerService(ledger)
            run = service.start_run(
                source_kind="ambient_ai_task",
                trigger_kind="explicit_user_task",
                title="Test run",
                summary="Testing ledger persistence.",
                model="test-model",
                tags=["research", "todoist"],
            )
            step = service.start_step(run.run_id, step_kind="llm_interaction", title="Do work")
            service.attach_artifact(
                run_id=run.run_id,
                step_id=step.step_id,
                artifact_kind="tool_output",
                title="Output",
                text_preview="artifact preview",
            )
            service.link_entity(
                run_id=run.run_id,
                entity_type="todoist_task",
                entity_id="task-1",
                relation="executes",
            )
            provider = LoggingLLMProvider(_FakeLLMProvider(), interaction_store)

            async def _run_trace():
                with interaction_trace("unit_test", {"run_id": run.run_id, "step_id": step.step_id}):
                    stream = await provider.chat_completion_stream(
                        model="test-model",
                        messages=[{"role": "user", "content": "hello"}],
                    )
                    async for _ in stream:
                        pass

            asyncio.run(_run_trace())
            service.complete_step(step.step_id, output_ref="done")
            service.complete_run(run.run_id, summary="Finished", output_text="Final output")

            self.assertEqual(len(ledger.list_runs(status="completed")), 1)
            self.assertEqual(len(ledger.search_runs("artifact")), 1)
            detail = ledger.get_run_detail(run.run_id)
            self.assertIsNotNone(detail)
            self.assertEqual(len(detail.steps), 1)
            self.assertEqual(len(detail.artifacts), 1)
            self.assertEqual(len(detail.links), 1)
            self.assertEqual(len(detail.traces), 1)

    def test_fastapi_endpoints_and_dashboard_render(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = SQLiteActivityLedgerAdapter(db_path=str(Path(tmpdir) / "activity_ledger.db"))
            service = ActivityLedgerService(ledger)
            run = service.start_run(
                source_kind="proactive_research",
                trigger_kind="research_queue",
                title="Research run",
                summary="Saved a research package.",
                model="test-model",
                tags=["research"],
            )
            service.attach_artifact(
                run_id=run.run_id,
                artifact_kind="research_summary",
                title="Summary",
                path="D:/tmp/summary.md",
                text_preview="Saved package preview",
            )
            service.complete_run(run.run_id, summary="Saved a research package.", output_text="Research done")

            client = TestClient(create_app(activity_ledger=ledger))

            resp = client.get("/api/activity/runs")
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(len(resp.json()["items"]), 1)

            detail_resp = client.get(f"/api/activity/runs/{run.run_id}")
            self.assertEqual(detail_resp.status_code, 200)
            self.assertEqual(detail_resp.json()["run"]["title"], "Research run")

            search_resp = client.get("/api/activity/search", params={"q": "research"})
            self.assertEqual(search_resp.status_code, 200)
            self.assertEqual(len(search_resp.json()["items"]), 1)

            summary_resp = client.get("/api/activity/summary")
            self.assertEqual(summary_resp.status_code, 200)
            self.assertEqual(summary_resp.json()["completed"], 1)

            page = client.get("/activity/")
            self.assertEqual(page.status_code, 200)
            self.assertIn("Research run", page.text)

            detail_page = client.get(f"/activity/{run.run_id}")
            self.assertEqual(detail_page.status_code, 200)
            self.assertIn("Research done", detail_page.text)

    def test_complex_transcript_task_creates_queued_run(self):
        from app import TranscriptionService

        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = ActivityLedgerService(
                SQLiteActivityLedgerAdapter(db_path=str(Path(tmpdir) / "activity_ledger.db"))
            )
            service = TranscriptionService(transcription_queue=__import__("queue").Queue())
            task_queue = _FakeTaskQueue()
            notifications = _FakeNotifications()

            classification = TranscriptClassificationResult(
                label="TASK_COMPLEX",
                speaker_label="USER",
                summary="Please gather GRPO notes.",
                confidence=0.9,
                reason="matched_complex_task_tokens",
                suggested_action="Gather GRPO learning resources and notes.",
            )

            asyncio.run(
                service._dispatch_classification(
                    classification=classification,
                    content="[0.0 - 1.0] -> USER: please gather GRPO notes",
                    transcript_path="transcriptions/sample.txt",
                    task_queue=task_queue,
                    notification_store=notifications,
                    proactive_topic_detector=_FakeTopicDetector(),
                    memory_store=_FakeMemoryStore(),
                    participants={},
                    simple_executor=_FakeSimpleExecutor(),
                    activity_ledger=ledger,
                    executor_prompt="unused",
                )
            )

            runs = ledger.ledger.list_runs(source_kind="transcript_task")
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0].status, "queued")
            detail = ledger.ledger.get_run_detail(runs[0].run_id)
            self.assertEqual(len(detail.artifacts), 1)
            self.assertEqual(task_queue.items[0][1], "medium")


if __name__ == "__main__":
    unittest.main()
