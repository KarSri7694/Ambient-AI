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
from application.services.passive_observer_service import PassiveObserverService
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
        self.currently_loaded_model = "main-model"
        self.events = []
        self.calls = []

    def get_current_model(self):
        return self.currently_loaded_model

    async def load_model(self, model_name: str):
        self.events.append(("load", model_name))
        self.currently_loaded_model = model_name

    async def unload_model(self):
        self.events.append(("unload", self.currently_loaded_model))
        self.currently_loaded_model = None

    async def chat_completion_stream(self, model, messages, tools=None, image="", temperature=0.7, top_p=0.95, top_k=0):
        user_message = next((item for item in reversed(messages) if item.get("role") == "user"), {})
        payload = {}
        try:
            payload = json.loads(user_message.get("content", "") or "{}")
        except json.JSONDecodeError:
            payload = {}
        self.calls.append(
            {
                "model": model,
                "image": image,
                "loaded_model": self.currently_loaded_model,
                "payload": payload,
            }
        )
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

    def add_task(self, description: str, priority: str = "medium", metadata=None) -> str:
        task = type("Task", (), {"description": description, "priority": priority, "metadata": metadata})()
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
                        "detailed_description": "Amazon cart shows laptop stand listings with comparison context, including visible product cards, pricing, ratings, and the user's focus on narrowing options before purchase.",
                        "inferred_user_activity": "reviewing items before purchase",
                        "previous_activity_status": "left_midway",
                        "salient_entities": ["amazon", "cart", "laptop stand"],
                        "visible_targets": ["laptop stands", "product cards"],
                        "selection_context": ["amazon cart", "comparison flow"],
                        "decision_factors": ["price", "rating", "design"],
                        "comparison_axes": ["price vs rating vs design"],
                        "artifact_state": "comparing",
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
        self.assertIn("screenshot_captured_at", llm.calls[0]["payload"])
        persisted = self.memory.get_recent_visual_observations(limit=5)
        self.assertEqual(len(persisted), 1)
        self.assertEqual(persisted[0].app_name, "Amazon")
        self.assertEqual(persisted[0].page_hint, "cart")
        self.assertIn("pricing, ratings", persisted[0].detailed_description)
        self.assertEqual(persisted[0].previous_activity_status, "left_midway")
        self.assertEqual(persisted[0].possible_next_task, "continue comparing laptop stands")
        self.assertEqual(persisted[0].user_fact_hypotheses[0]["title"], "looking for laptop stand")
        self.assertEqual(len(self.memory.list_visual_sessions(statuses=["open"], limit=5)), 1)
        self.assertIn("cart review left unfinished", self.memory.get_visual_digest())
        self.assertIn("pricing, ratings", self.memory.get_visual_digest())

    def test_observer_allows_repeated_capture_processing(self):
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
                ),
                json.dumps(
                    {
                        "app_name": "Docs",
                        "summary": "Email draft is still open.",
                        "inferred_user_activity": "still drafting an email",
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
                ),
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
        self.assertIsNotNone(second)
        self.assertEqual(len(self.memory.get_recent_visual_observations(limit=5)), 2)

    def test_process_screenshot_handles_deferred_queue_item(self):
        llm = FakeVisualLLM(
            [
                json.dumps(
                    {
                        "app_name": "Steam",
                        "summary": "Steam store is open on GTA 5.",
                        "inferred_user_activity": "checking a game page",
                        "previous_activity_status": "new",
                        "salient_entities": ["steam", "gta 5"],
                        "completed_items": [],
                        "open_loops": ["decide whether to buy gta 5"],
                        "possible_next_task": "compare GTA 5 editions later",
                        "user_fact_hypotheses": [
                            {
                                "category": "entertainment_preference",
                                "title": "interested in open world games",
                                "summary": "User may be interested in open-world games.",
                                "confidence": 0.7,
                                "scope": "temporary",
                                "evidence_strength": "medium",
                            }
                        ],
                        "suggested_research_topics": [],
                        "confidence": 0.75,
                        "worth_noting": True,
                    }
                )
            ]
        )
        screenshot = self.temp_path / "queued.png"
        screenshot.write_bytes(b"queued")
        service = PassiveObserverService(
            memory=self.memory,
            llm_provider=llm,
            screen_capture=FakeScreenCapture([]),
            screenshot_root=str(self.temp_path / "shots"),
        )

        observation = asyncio.run(
            service.process_screenshot(
                screenshot_path=str(screenshot),
                model="test-model",
                recent_context="",
                captured_at="2026-06-25T10:00:00",
            )
        )

        self.assertIsNotNone(observation)
        self.assertEqual(observation.app_name, "Steam")
        self.assertEqual(observation.created_at, "2026-06-25T10:00:00")
        self.assertEqual(
            llm.calls[0]["payload"]["screenshot_captured_at"],
            "2026-06-25T10:00:00",
        )
        self.assertEqual(self.memory.get_recent_visual_observations(limit=1)[0].app_name, "Steam")
        self.assertTrue(screenshot.exists())

    def test_process_screenshot_returns_none_when_file_is_missing(self):
        llm = FakeVisualLLM([json.dumps({"worth_noting": True})])
        missing = self.temp_path / "missing.png"
        service = PassiveObserverService(
            memory=self.memory,
            llm_provider=llm,
            screen_capture=FakeScreenCapture([]),
            screenshot_root=str(self.temp_path / "shots"),
        )

        observation = asyncio.run(
            service.process_screenshot(
                screenshot_path=str(missing),
                model="test-model",
                recent_context="",
            )
        )

        self.assertIsNone(observation)

    def test_followup_service_queues_task_from_open_loop(self):
        observer_llm = FakeVisualLLM(
            [
                json.dumps(
                    {
                        "app_name": "Gmail",
                        "summary": "A draft email is open.",
                        "detailed_description": "Gmail compose view is open with an unfinished reply draft and editable message body visible.",
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
        self.assertIn("Passive observation context:", task_queue.items[0].description)
        self.assertIn("Detailed description:", task_queue.items[0].description)
        self.assertIn("Inferred user activity:", task_queue.items[0].description)
        self.assertEqual(task_queue.items[0].metadata["task_kind"], "passive_observer_followup")
        self.assertEqual(task_queue.items[0].metadata["source_observation_id"], observation.observation_id)
        self.assertIn("unfinished reply draft", task_queue.items[0].metadata["detailed_description"])
        persisted = self.memory.get_visual_observation(observation.observation_id)
        self.assertIsNotNone(persisted)
        self.assertTrue(persisted.followup_sent_at)

    def test_followup_service_only_sends_unsent_observations(self):
        older = VisualObservation(
            observation_id="obs-sent",
            screenshot_path="sent.png",
            created_at="2026-06-25T10:00:00",
            app_name="Amazon",
            summary="Earlier shopping observation.",
            inferred_user_activity="comparing products",
            open_loops=["compare products"],
            possible_next_task="continue comparing products",
            followup_sent_at="2026-06-25T10:05:00",
            confidence=0.7,
        )
        newer = VisualObservation(
            observation_id="obs-unsent",
            screenshot_path="unsent.png",
            created_at="2026-06-25T10:10:00",
            app_name="Amazon",
            summary="Current shopping observation.",
            detailed_description="Product comparison page is still open.",
            inferred_user_activity="comparing shortlisted products",
            previous_activity_status="left_midway",
            open_loops=["compare shortlisted products"],
            possible_next_task="continue the product comparison",
            confidence=0.8,
        )
        self.memory.append_visual_observation(older)
        self.memory.append_visual_observation(newer)

        llm = FakeVisualLLM(
            [
                json.dumps(
                    {
                        "action": "nothing",
                        "title": "",
                        "description": "",
                        "source_observation_id": "",
                        "confidence": 0.2,
                    }
                )
            ]
        )
        followup = PassiveObserverFollowupService(
            memory=self.memory,
            task_queue=FakeTaskQueue(),
            llm_provider=llm,
        )

        result = asyncio.run(followup.maybe_queue_followup(model="test-model"))

        self.assertEqual(result["action"], "nothing")
        payload = llm.calls[0]["payload"]
        sent_ids = [item["observation_id"] for item in payload["recent_visual_observations"]]
        self.assertEqual(sent_ids, ["obs-unsent"])
        persisted_unsent = self.memory.get_visual_observation("obs-unsent")
        persisted_sent = self.memory.get_visual_observation("obs-sent")
        self.assertIsNotNone(persisted_unsent)
        self.assertTrue(persisted_unsent.followup_sent_at)
        self.assertEqual(persisted_sent.followup_sent_at, "2026-06-25T10:05:00")

    def test_followup_service_rejects_ephemeral_feed_scroll(self):
        observation = VisualObservation(
            observation_id="obs-scroll",
            screenshot_path="scroll.png",
            created_at="2026-06-25T10:00:00",
            app_name="Instagram",
            page_hint="feed",
            summary="Instagram feed is open.",
            inferred_user_activity="scrolling the Instagram feed",
            previous_activity_status="continued",
            open_loops=["scroll more through the feed"],
            possible_next_task="scroll further down the feed to see more posts",
            confidence=0.8,
        )
        self.memory.append_visual_observation(observation)

        task_queue = FakeTaskQueue()
        followup = PassiveObserverFollowupService(
            memory=self.memory,
            task_queue=task_queue,
            llm_provider=FakeVisualLLM([]),
        )
        result = asyncio.run(followup.maybe_queue_followup(model="test-model"))

        self.assertEqual(result["action"], "nothing")
        self.assertEqual(len(task_queue.items), 0)

    def test_followup_service_rejects_ephemeral_call_accept(self):
        observation = VisualObservation(
            observation_id="obs-call",
            screenshot_path="call.png",
            created_at="2026-06-25T10:00:00",
            app_name="WhatsApp",
            summary="Incoming video call from Milli is visible.",
            inferred_user_activity="deciding whether to accept a WhatsApp video call",
            previous_activity_status="new",
            open_loops=["accept the WhatsApp video call from Milli"],
            possible_next_task="click Accept to join the video call from Milli",
            confidence=0.85,
        )
        self.memory.append_visual_observation(observation)

        task_queue = FakeTaskQueue()
        followup = PassiveObserverFollowupService(
            memory=self.memory,
            task_queue=task_queue,
            llm_provider=FakeVisualLLM([]),
        )
        result = asyncio.run(followup.maybe_queue_followup(model="test-model"))

        self.assertEqual(result["action"], "nothing")
        self.assertEqual(len(task_queue.items), 0)

if __name__ == "__main__":
    unittest.main()
