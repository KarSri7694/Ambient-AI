import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from application.services.passive_observer_followup_service import PassiveObserverFollowupService
from application.services.reflection_service import ReflectionService
from application.services.semantic_deduplication_service import SemanticDeduplicationService
from core.models import VisualObservation
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


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def chat_completion_stream(self, model, messages, tools=None, image="", temperature=0.7, top_p=0.95, top_k=0):
        self.calls.append(
            {
                "model": model,
                "messages": messages,
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


class FakeReminderHelper:
    def __init__(self):
        self.created = []

    def is_enabled(self):
        return True

    def get_tasks(self):
        return []

    def add_task(self, content, due_datetime=None):
        self.created.append(content)
        return {"id": f"todo-{len(self.created)}", "content": content}


class SemanticDeduplicationServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.memory = SQLiteMemoryAdapter(
            db_path=str(self.temp_path / "memory.db"),
            memory_root=str(self.temp_path / "memory"),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_duplicate_skip_uses_prior_registry_item(self):
        llm = FakeLLM(
            [
                json.dumps(
                    {
                        "decision": "duplicate_skip",
                        "duplicate_of_item_id": "existing-1",
                        "reason": "Same reminder phrased differently.",
                        "confidence": 0.93,
                    }
                )
            ]
        )
        service = SemanticDeduplicationService(
            memory=self.memory,
            llm_provider=llm,
            enabled=True,
            model="dedupe-model",
            candidate_limit=5,
        )
        self.memory.add_semantic_dedupe_item(
            dedupe_item_id="existing-1",
            entity_kind="todoist_reminder",
            source_kind="seed",
            raw_text="Call mom tonight",
            status="created",
            ttl_expires_at="2099-01-01T00:00:00",
        )

        result = asyncio.run(
            service.evaluate_candidate(
                entity_kind="todoist_reminder",
                source_kind="test",
                text="Set a reminder to call mom tonight",
                metadata={"due_datetime": "2026-06-29T20:00:00"},
                model="dedupe-model",
            )
        )

        self.assertEqual(result["decision"], "duplicate_skip")
        self.assertEqual(result["duplicate_of_item_id"], "existing-1")
        self.assertEqual(result["matched_item"].raw_text, "Call mom tonight")

    def test_invalid_json_falls_back_to_create_new(self):
        llm = FakeLLM(["not json"])
        service = SemanticDeduplicationService(
            memory=self.memory,
            llm_provider=llm,
            enabled=True,
            model="dedupe-model",
        )
        self.memory.add_semantic_dedupe_item(
            dedupe_item_id="existing-1",
            entity_kind="internal_task",
            source_kind="seed",
            raw_text="compare TV prices",
            status="created",
            ttl_expires_at="2099-01-01T00:00:00",
        )

        result = asyncio.run(
            service.evaluate_candidate(
                entity_kind="internal_task",
                source_kind="test",
                text="compare TV prices again",
                model="dedupe-model",
            )
        )

        self.assertEqual(result["decision"], "create_new")
        self.assertEqual(result["reason"], "semantic_dedupe_invalid_json")

    def test_expired_items_do_not_participate(self):
        llm = FakeLLM([])
        service = SemanticDeduplicationService(
            memory=self.memory,
            llm_provider=llm,
            enabled=True,
            model="dedupe-model",
        )
        record = self.memory.add_semantic_dedupe_item(
            entity_kind="do_now_action",
            source_kind="seed",
            raw_text="check the cricket score",
            status="created",
            ttl_expires_at="2099-01-01T00:00:00",
        )
        self.memory.update_semantic_dedupe_item(record.dedupe_item_id, ttl_expires_at="2000-01-01T00:00:00")

        result = asyncio.run(
            service.evaluate_candidate(
                entity_kind="do_now_action",
                source_kind="test",
                text="check the cricket score",
                model="dedupe-model",
            )
        )

        self.assertEqual(result["decision"], "create_new")
        self.assertEqual(result["reason"], "no_recent_candidates")
        self.assertEqual(llm.calls, [])

    def test_passive_followup_skips_duplicate_direct_reminder(self):
        llm = FakeLLM(
            [
                json.dumps(
                    {
                        "decision": "duplicate_skip",
                        "duplicate_of_item_id": "existing-reminder",
                        "reason": "Same event reminder.",
                        "confidence": 0.95,
                    }
                ),
                json.dumps({"unique_activities": ["checking event details"]}),
                json.dumps({"useful_activities": []}),
                json.dumps(
                    {
                        "action": "nothing",
                        "task": "",
                        "memory_updates": [],
                        "user_info_updates": [],
                    }
                ),
            ]
        )
        dedupe_service = SemanticDeduplicationService(
            memory=self.memory,
            llm_provider=llm,
            enabled=True,
            model="dedupe-model",
        )
        self.memory.add_semantic_dedupe_item(
            dedupe_item_id="existing-reminder",
            entity_kind="todoist_reminder",
            source_kind="seed",
            raw_text="Code Autopsy 1.0 starts tonight at 9 PM",
            status="created",
            ttl_expires_at="2099-01-01T00:00:00",
        )
        observation = VisualObservation(
            observation_id="obs-1",
            screenshot_path="shot.png",
            created_at="2026-06-29T20:30:00",
            app_name="WhatsApp",
            summary="Event details are open.",
            detailed_description="A chat mentions Code Autopsy tonight at 9 PM.",
            inferred_user_activity="checking event details",
            raw_payload_json=json.dumps(
                {
                    "maybe_require_a_reminder": True,
                    "reminder_context": {
                        "message_to_user": "Code Autopsy 1.0 starts tonight at 9 PM",
                        "due_date": "2026-06-29T21:00:00+05:30",
                    },
                }
            ),
        )
        service = PassiveObserverFollowupService(
            memory=self.memory,
            task_queue=FakeTaskQueue(),
            llm_provider=llm,
            semantic_dedupe_service=dedupe_service,
            reminder_helper=FakeReminderHelper(),
        )

        result = asyncio.run(
            service.process_observations(
                observations=[observation],
                model="dedupe-model",
                mark_sent=False,
                apply_memory_updates=False,
            )
        )

        self.assertEqual(result["direct_reminders"], [])
        skipped = self.memory.list_semantic_dedupe_items(statuses=["duplicate_skipped"], limit=5)
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0].duplicate_of_item_id, "existing-reminder")

    def test_reflection_skips_semantic_duplicate_task(self):
        self.memory.save_user_info("- User researches TV deals.\n")
        llm = FakeLLM(
            [
                "- User researches TV deals.\n",
                json.dumps(
                    {
                        "tasks": [
                            {
                                "description": "compare the latest TV deals",
                                "priority": "medium",
                                "reason": "Fresh comparison could help.",
                            }
                        ]
                    }
                ),
                json.dumps(
                    {
                        "decision": "duplicate_skip",
                        "duplicate_of_item_id": "existing-reflection",
                        "reason": "Same research task.",
                        "confidence": 0.88,
                    }
                ),
            ]
        )
        dedupe_service = SemanticDeduplicationService(
            memory=self.memory,
            llm_provider=llm,
            enabled=True,
            model="reflection-model",
        )
        self.memory.add_semantic_dedupe_item(
            dedupe_item_id="existing-reflection",
            entity_kind="reflection_task",
            source_kind="seed",
            raw_text="compare the latest TV deals",
            status="created",
            ttl_expires_at="2099-01-01T00:00:00",
        )
        history_path = self.temp_path / "reflection" / "history.json"
        service = ReflectionService(
            memory=self.memory,
            task_queue=FakeTaskQueue(),
            llm_provider=llm,
            semantic_dedupe_service=dedupe_service,
            history_path=str(history_path),
            cadence_mode="every_idle_cycle",
        )

        result = asyncio.run(service.run(model="reflection-model"))

        self.assertEqual(result["queued_tasks"], [])
        self.assertEqual(result["skipped_tasks"][0]["skip_reason"], "semantic_duplicate")
        skipped = self.memory.list_semantic_dedupe_items(statuses=["duplicate_skipped"], limit=5)
        self.assertEqual(skipped[0].duplicate_of_item_id, "existing-reflection")


if __name__ == "__main__":
    unittest.main()
