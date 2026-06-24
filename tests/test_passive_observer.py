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
from application.services.passive_observer_service import PassiveObserverService
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


class FakeNotifications:
    def __init__(self):
        self.items = []

    def add_notification(self, message: str, source: str = "system") -> None:
        self.items.append((message, source))

    def peek_unread_notifications(self):
        return []

    def get_unread_notifications(self):
        return []

    def mark_read(self, notification_id: int) -> None:
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

    def test_observer_persists_visual_context_and_notifications(self):
        llm = FakeVisualLLM(
            [
                json.dumps(
                    {
                        "app_name": "Amazon",
                        "window_title": "Amazon cart",
                        "page_hint": "cart",
                        "summary": "Amazon cart is open with a product comparison in progress.",
                        "inferred_user_activity": "reviewing items before purchase",
                        "salient_entities": ["amazon", "cart", "laptop stand"],
                        "open_loops": ["cart review left unfinished"],
                        "suggested_research_topics": ["best laptop stand"],
                        "confidence": 0.82,
                        "worth_noting": True,
                    }
                )
            ]
        )
        capture = FakeScreenCapture([b"image-1"])
        notifications = FakeNotifications()
        service = PassiveObserverService(
            memory=self.memory,
            llm_provider=llm,
            screen_capture=capture,
            notifications=notifications,
            screenshot_root=str(self.temp_path / "shots"),
        )

        observation = asyncio.run(service.observe(model="test-model", recent_context=""))

        self.assertIsNotNone(observation)
        persisted = self.memory.get_recent_visual_observations(limit=5)
        self.assertEqual(len(persisted), 1)
        self.assertEqual(persisted[0].app_name, "Amazon")
        self.assertEqual(persisted[0].page_hint, "cart")
        self.assertEqual(len(self.memory.list_visual_sessions(statuses=["open"], limit=5)), 1)
        self.assertEqual(len(notifications.items), 1)
        self.assertIn("unfinished activity", notifications.items[0][0])
        self.assertIn("cart review left unfinished", self.memory.get_visual_digest())

    def test_observer_skips_unchanged_screen(self):
        llm = FakeVisualLLM(
            [
                json.dumps(
                    {
                        "app_name": "Docs",
                        "summary": "Email draft is open.",
                        "inferred_user_activity": "drafting an email",
                        "salient_entities": ["email"],
                        "open_loops": ["finish email draft"],
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
            notifications=FakeNotifications(),
            screenshot_root=str(self.temp_path / "shots"),
        )

        first = asyncio.run(service.observe(model="test-model", recent_context=""))
        second = asyncio.run(service.observe(model="test-model", recent_context=""))

        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertEqual(len(self.memory.get_recent_visual_observations(limit=5)), 1)

    def test_prompt_builder_includes_visual_digest(self):
        self.memory.save_visual_digest("# Passive Visual Context\n\n- Amazon cart review\n")
        builder = MemoryContextBuilder(self.memory, str(self.prompts_dir))
        prompt = builder.build_prompt(
            base_prompt_filename="AGENT.md",
            skills_summary="Skill summary",
            participants=[],
        )
        self.assertIn("Passive Visual Context", prompt)
        self.assertIn("Amazon cart review", prompt)


if __name__ == "__main__":
    unittest.main()
