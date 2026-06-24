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

from application.services.memory_context_builder import MemoryContextBuilder
from application.services.passive_observer_followup_service import PassiveObserverFollowupService
from application.services.passive_observer_service import PassiveObserverService
from application.services.visual_user_fact_service import VisualUserFactService
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


class FakeVisualLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    async def chat_completion_stream(self, model, messages, tools=None, image="", temperature=0.7, top_p=0.95, top_k=0):
        response = self.responses.pop(0)

        async def _gen():
            yield _FakeChunk(content=response)

        return _gen()


class FakeScreenCapture:
    def __init__(self, payloads):
        self.payloads = list(payloads)

    def capture_screenshot(self, output_path=None):
        path = Path(output_path)
        payload = self.payloads.pop(0)
        path.write_bytes(payload)
        return str(path)


class FakeTaskQueue:
    def __init__(self):
        self.items = []

    def get_pending_tasks(self):
        return list(self.items)

    def add_task(self, description: str, priority: str = "medium") -> str:
        task = type("Task", (), {"description": description, "priority": priority})()
        self.items.append(task)
        return "queued"

    def mark_task_complete(self, task_id: int, status: str = "completed") -> None:
        return None


class PassiveObserverTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        prompts_dir = self.temp_path / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        (prompts_dir / "AGENT.md").write_text("Agent prompt", encoding="utf-8")
        (prompts_dir / "USER.md").write_text("User prompt", encoding="utf-8")
        self.prompts_dir = prompts_dir
        self.memory = SQLiteMemoryAdapter(
            db_path=str(self.temp_path / "memory.db"),
            memory_root=str(self.temp_path / "memory"),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_observer_persists_visual_context(self):
        llm = FakeVisualLLM(
            [
                json.dumps(
                    {
                        "app_name": "Amazon",
                        "window_title": "Amazon cart",
                        "page_hint": "cart",
                        "summary": "Amazon cart is open with a product comparison in progress.",
                        "inferred_user_activity": "reviewing items before purchase",
                        "previous_activity_status": "left_midway",
                        "salient_entities": ["amazon", "cart", "laptop stand"],
                        "completed_items": [],
                        "open_loops": ["cart review left unfinished"],
                        "possible_next_task": "continue comparing laptop stands",
                        "user_fact_hypotheses": [
                            {
                                "category": "shopping_intent",
                                "title": "looking for laptop stand",
                                "summary": "User is currently looking for a laptop stand.",
                                "confidence": 0.82,
                                "scope": "temporary",
                                "evidence_strength": "medium",
                            }
                        ],
                        "suggested_research_topics": ["best laptop stand"],
                        "confidence": 0.82,
                        "worth_noting": True,
                    }
                )
            ]
        )
        capture = FakeScreenCapture([b"image-1"])
        service = PassiveObserverService(
            memory=self.memory,
            llm_provider=llm,
            screen_capture=capture,
            screenshot_root=str(self.temp_path / "shots"),
        )

        observation = asyncio.run(service.observe(model="test-model", recent_context=""))

        self.assertIsNotNone(observation)
        persisted = self.memory.get_recent_visual_observations(limit=5)
        self.assertEqual(len(persisted), 1)
        self.assertEqual(persisted[0].app_name, "Amazon")
        self.assertEqual(persisted[0].page_hint, "cart")
        self.assertEqual(persisted[0].previous_activity_status, "left_midway")
        self.assertEqual(persisted[0].possible_next_task, "continue comparing laptop stands")
        self.assertEqual(persisted[0].user_fact_hypotheses[0]["title"], "looking for laptop stand")
        self.assertEqual(len(self.memory.list_visual_sessions(statuses=["open"], limit=5)), 1)
        self.assertIn("cart review left unfinished", self.memory.get_visual_digest())

    def test_observer_skips_unchanged_screen(self):
        llm = FakeVisualLLM(
            [
                json.dumps(
                    {
                        "app_name": "Docs",
                        "summary": "Email draft is open.",
                        "inferred_user_activity": "drafting an email",
                        "previous_activity_status": "continued",
                        "salient_entities": ["email"],
                        "completed_items": [],
                        "open_loops": ["finish email draft"],
                        "possible_next_task": "finish the email draft",
                        "user_fact_hypotheses": [],
                        "suggested_research_topics": [],
                        "confidence": 0.75,
                        "worth_noting": True,
                    }
                )
            ]
        )
        capture = FakeScreenCapture([b"same-image", b"same-image"])
        service = PassiveObserverService(
            memory=self.memory,
            llm_provider=llm,
            screen_capture=capture,
            screenshot_root=str(self.temp_path / "shots"),
        )

        first = asyncio.run(service.observe(model="test-model", recent_context=""))
        second = asyncio.run(service.observe(model="test-model", recent_context=""))

        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertEqual(len(self.memory.get_recent_visual_observations(limit=5)), 1)

    def test_prompt_builder_includes_visual_digest(self):
        self.memory.save_visual_digest("# Passive Visual Context\n\n- Amazon cart review\n")
        self.memory.save_user_info("# USER_INFO\n\n- User likes strategy games.\n")
        builder = MemoryContextBuilder(self.memory, str(self.prompts_dir))
        prompt = builder.build_prompt(
            base_prompt_filename="AGENT.md",
            skills_summary="Skill summary",
            participants=[],
        )
        self.assertIn("Passive Visual Context", prompt)
        self.assertIn("Amazon cart review", prompt)
        self.assertIn("User likes strategy games.", prompt)

    def test_followup_service_queues_task_from_open_loop(self):
        observer_llm = FakeVisualLLM(
            [
                json.dumps(
                    {
                        "app_name": "Gmail",
                        "summary": "A draft email is open.",
                        "inferred_user_activity": "writing an email reply",
                        "previous_activity_status": "left_midway",
                        "salient_entities": ["gmail", "email"],
                        "completed_items": [],
                        "open_loops": ["finish the email reply"],
                        "possible_next_task": "complete the email reply draft",
                        "user_fact_hypotheses": [],
                        "suggested_research_topics": [],
                        "confidence": 0.8,
                        "worth_noting": True,
                    }
                ),
                json.dumps(
                    {
                        "action": "queue_task",
                        "title": "Complete email reply draft",
                        "description": "Return to the unfinished Gmail draft and complete the reply if the user is away.",
                        "source_observation_id": "",
                        "confidence": 0.8,
                    }
                ),
            ]
        )
        service = PassiveObserverService(
            memory=self.memory,
            llm_provider=observer_llm,
            screen_capture=FakeScreenCapture([b"gmail-draft"]),
            screenshot_root=str(self.temp_path / "shots"),
        )
        observation = asyncio.run(service.observe(model="test-model", recent_context=""))
        self.assertIsNotNone(observation)

        task_queue = FakeTaskQueue()
        followup = PassiveObserverFollowupService(
            memory=self.memory,
            task_queue=task_queue,
            llm_provider=observer_llm,
        )
        result = asyncio.run(followup.maybe_queue_followup(model="test-model"))

        self.assertEqual(result["action"], "queue_task")
        self.assertEqual(len(task_queue.items), 1)
        self.assertIn("Complete email reply draft", task_queue.items[0].description)

    def test_visual_user_fact_service_promotes_repeated_visual_interest(self):
        service = VisualUserFactService(memory=self.memory)
        first = VisualObservation(
            observation_id="obs-1",
            screenshot_path="shot1.png",
            created_at="2026-06-24T10:00:00",
            summary="Amazon is open with mobile phones.",
            inferred_user_activity="comparing mobile phones",
            user_fact_hypotheses=[
                {
                    "category": "device_interest",
                    "title": "interested in mobile phones",
                    "summary": "User may be interested in buying a mobile phone.",
                    "confidence": 1.0,
                    "scope": "temporary",
                    "evidence_strength": "strong",
                }
            ],
            session_id="session-a",
        )
        second = VisualObservation(
            observation_id="obs-2",
            screenshot_path="shot2.png",
            created_at="2026-06-25T10:00:00",
            summary="Amazon is open with more phone listings.",
            inferred_user_activity="researching smartphones again",
            user_fact_hypotheses=[
                {
                    "category": "device_interest",
                    "title": "interested in mobile phones",
                    "summary": "User may be interested in buying a mobile phone.",
                    "confidence": 1.0,
                    "scope": "temporary",
                    "evidence_strength": "strong",
                }
            ],
            session_id="session-b",
        )

        first_updates = service.update_from_observation(first)
        second_updates = service.update_from_observation(second)

        self.assertEqual(first_updates[0].status, "emerging")
        self.assertEqual(second_updates[0].status, "durable")
        stored = self.memory.get_visual_user_fact("device-interest-interested-in-mobile-phones")
        self.assertIsNotNone(stored)
        self.assertEqual(stored.status, "durable")
        self.assertIn("mobile phone", self.memory.get_user_info().lower())


if __name__ == "__main__":
    unittest.main()
