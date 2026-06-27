import asyncio
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from application.services.reflection_service import ReflectionService
from infrastructure.adapter.SQLiteMemoryAdapter import SQLiteMemoryAdapter


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


class FakeReflectionLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def chat_completion_stream(self, model, messages, tools=None, image="", temperature=0.7, top_p=0.95, top_k=0):
        self.calls.append(
            {
                "model": model,
                "system": next((m["content"] for m in messages if m.get("role") == "system"), ""),
                "user": next((m["content"] for m in messages if m.get("role") == "user"), ""),
            }
        )
        response = self.responses.pop(0)

        async def _gen():
            yield _FakeChunk(content=response)

        return _gen()


class FakeTaskQueue:
    def __init__(self, pending=None):
        self.pending = list(pending or [])
        self.items = []

    def get_pending_tasks(self):
        return list(self.pending)

    def add_task(self, description: str, priority: str = "medium", metadata=None) -> str:
        task = type("Task", (), {"description": description, "priority": priority, "metadata": metadata})()
        self.items.append(task)
        return "queued"

    def mark_task_complete(self, task_id: int, status: str = "completed") -> None:
        return None


class ReflectionServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.memory = SQLiteMemoryAdapter(
            db_path=str(self.temp_path / "memory.db"),
            memory_root=str(self.temp_path / "memory"),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_reflection_service_cleans_user_info_and_queues_new_tasks(self):
        self.memory.save_user_info(
            "## BioData Update - 2026-06-28 10:00:00\n"
            "- [interest] User likes budget TV research.\n"
            "- [interest] User likes budget TV research.\n"
            "- [education] User may have an upcoming test.\n"
        )
        llm = FakeReflectionLLM(
            [
                "## Interests\n- User likes budget TV research.\n\n## Education\n- User may have an upcoming test.\n",
                json.dumps(
                    {
                        "tasks": [
                            {
                                "description": "compare current TV deals for the user's preferred budget range",
                                "priority": "medium",
                                "reason": "The user repeatedly researches TVs and could benefit from a fresh comparison.",
                            },
                            {
                                "description": "set a reminder about the upcoming test",
                                "priority": "high",
                                "reason": "The user may need help remembering the upcoming test.",
                            },
                            {
                                "description": "compare current TV deals for the user's preferred budget range",
                                "priority": "medium",
                                "reason": "Duplicate in same run.",
                            },
                        ]
                    }
                ),
            ]
        )
        pending = [type("Task", (), {"description": "set a reminder about the upcoming test"})()]
        task_queue = FakeTaskQueue(pending=pending)
        history_path = self.temp_path / "reflection" / "history.json"
        service = ReflectionService(
            memory=self.memory,
            task_queue=task_queue,
            llm_provider=llm,
            history_path=str(history_path),
            cadence_mode="daily",
            max_generated_tasks=8,
        )

        result = asyncio.run(service.run_if_due(model="reflection-model", now=datetime(2026, 6, 28, 12, 0, 0)))

        self.assertTrue(result["ran"])
        user_info = self.memory.get_user_info()
        self.assertEqual(user_info.count("User likes budget TV research."), 1)
        self.assertEqual(len(task_queue.items), 1)
        self.assertIn("compare current TV deals", task_queue.items[0].description)
        payload = json.loads(history_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["last_run_at"], "2026-06-28T12:00:00")
        self.assertEqual(len(payload["runs"]), 1)
        self.assertEqual(payload["runs"][0]["generated_at"], "2026-06-28T12:00:00")
        self.assertEqual(len(payload["runs"][0]["generated_tasks"]), 3)
        self.assertEqual(len(payload["runs"][0]["queued_tasks"]), 1)
        self.assertEqual(len(payload["runs"][0]["skipped_tasks"]), 2)

    def test_reflection_service_skips_when_not_due(self):
        self.memory.save_user_info("- [interest] User likes reading.\n")
        history_path = self.temp_path / "reflection" / "history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(
            json.dumps(
                {
                    "last_run_at": "2026-06-28T08:00:00",
                    "runs": [],
                }
            ),
            encoding="utf-8",
        )
        llm = FakeReflectionLLM([])
        service = ReflectionService(
            memory=self.memory,
            task_queue=FakeTaskQueue(),
            llm_provider=llm,
            history_path=str(history_path),
            cadence_mode="daily",
        )

        result = asyncio.run(service.run_if_due(model="reflection-model", now=datetime(2026, 6, 28, 12, 0, 0)))

        self.assertFalse(result["ran"])
        self.assertEqual(result["reason"], "not_due")
        self.assertEqual(llm.calls, [])


if __name__ == "__main__":
    unittest.main()
