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
from application.services.user_bio_data_service import UserBioDataService
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
                "user_content": user_message.get("content", ""),
                "system_content": next((item.get("content", "") for item in messages if item.get("role") == "system"), ""),
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


class FakeUIATAdapter:
    def __init__(self, payload):
        self.payload = payload

    def inspect_foreground_window(self):
        return dict(self.payload)


class FakeReminderHelper:
    def __init__(self, existing=None):
        self.existing = list(existing or [])
        self.created = []

    def is_enabled(self):
        return True

    def get_tasks(self):
        return [{"content": item} for item in self.existing + self.created]

    def add_task(self, content, due_datetime=None):
        self.created.append(content)
        return {"content": content, "due_datetime": due_datetime}


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
                        "app_page": "Amazon / cart",
                        "summary": "Amazon cart is open with a product comparison in progress.",
                        "detailed_description": "Amazon cart shows laptop stand listings with comparison context, including visible product cards, pricing, ratings, and the user's focus on narrowing options before purchase.",
                        "inferred_user_activity": "reviewing items before purchase",
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
        self.assertEqual(persisted[0].previous_activity_status, "unclear")
        self.assertIsNone(persisted[0].possible_next_task)
        self.assertEqual(persisted[0].user_fact_hypotheses, [])
        self.assertEqual(len(self.memory.list_visual_sessions(statuses=["open"], limit=5)), 1)
        self.assertIn("pricing, ratings", self.memory.get_visual_digest())

    def test_observer_allows_repeated_capture_processing(self):
        llm = FakeVisualLLM(
            [
                json.dumps(
                    {
                        "app_page": "Docs",
                        "summary": "Email draft is open.",
                        "inferred_user_activity": "drafting an email",
                    }
                ),
                json.dumps(
                    {
                        "app_page": "Docs",
                        "summary": "Email draft is still open.",
                        "inferred_user_activity": "still drafting an email",
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
                        "app_page": "Steam",
                        "summary": "Steam store is open on GTA 5.",
                        "inferred_user_activity": "checking a game page",
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

    def test_process_screenshot_uses_fast_model_for_medium_similarity(self):
        llm = FakeVisualLLM(
            [
                json.dumps(
                    {
                        "app_page": "Docs / report",
                        "summary": "A report remains open with small edits.",
                        "detailed_description": "The same report view is visible with minor textual updates.",
                        "inferred_user_activity": "editing a report",
                    }
                )
            ]
        )
        screenshot = self.temp_path / "fast.png"
        screenshot.write_bytes(b"queued")
        service = PassiveObserverService(
            memory=self.memory,
            llm_provider=llm,
            screen_capture=FakeScreenCapture([]),
            screenshot_root=str(self.temp_path / "shots"),
            fast_model="fast-model",
            full_model="full-model",
        )

        observation = asyncio.run(
            service.process_screenshot(
                screenshot_path=str(screenshot),
                model="fallback-model",
                recent_context="",
                captured_at="2026-06-25T10:00:00",
                similarity_score=0.8,
            )
        )

        self.assertIsNotNone(observation)
        self.assertEqual(llm.calls[0]["model"], "fast-model")
        payload = json.loads(observation.raw_payload_json or "{}")
        self.assertEqual(payload.get("_analysis_mode"), "fast_model")

    def test_process_screenshot_uses_full_model_for_large_change(self):
        llm = FakeVisualLLM(
            [
                json.dumps(
                    {
                        "app_page": "Amazon / product page",
                        "summary": "A new product page is open.",
                        "detailed_description": "A laptop product page is visible with pricing and reviews.",
                        "inferred_user_activity": "evaluating a product purchase",
                    }
                )
            ]
        )
        screenshot = self.temp_path / "full.png"
        screenshot.write_bytes(b"queued")
        service = PassiveObserverService(
            memory=self.memory,
            llm_provider=llm,
            screen_capture=FakeScreenCapture([]),
            screenshot_root=str(self.temp_path / "shots"),
            fast_model="fast-model",
            full_model="full-model",
        )

        observation = asyncio.run(
            service.process_screenshot(
                screenshot_path=str(screenshot),
                model="fallback-model",
                recent_context="",
                captured_at="2026-06-25T10:00:00",
                similarity_score=0.4,
            )
        )

        self.assertIsNotNone(observation)
        self.assertEqual(llm.calls[0]["model"], "full-model")
        payload = json.loads(observation.raw_payload_json or "{}")
        self.assertEqual(payload.get("_analysis_mode"), "full_vlm")

    def test_process_screenshot_skips_ignored_app_without_override(self):
        llm = FakeVisualLLM([])
        screenshot = self.temp_path / "ignored.png"
        screenshot.write_bytes(b"queued")
        service = PassiveObserverService(
            memory=self.memory,
            llm_provider=llm,
            screen_capture=FakeScreenCapture([]),
            screenshot_root=str(self.temp_path / "shots"),
            ignore_apps=["spotify"],
            uiat_adapter=FakeUIATAdapter(
                {
                    "ok": True,
                    "window_title": "Spotify Premium",
                    "window_class": "SpotifyMainWindow",
                    "visible_text_summary": "Spotify playlist and playback controls",
                    "contains_dialog": False,
                    "contains_notification": False,
                }
            ),
        )

        observation = asyncio.run(
            service.process_screenshot(
                screenshot_path=str(screenshot),
                model="fallback-model",
                recent_context="",
                captured_at="2026-06-25T10:00:00",
                similarity_score=0.8,
            )
        )

        self.assertIsNone(observation)
        self.assertEqual(llm.calls, [])

    def test_process_screenshot_can_disable_persistence(self):
        llm = FakeVisualLLM(
            [
                json.dumps(
                    {
                        "app_page": "Docs / transient",
                        "summary": "A transient screen was analyzed.",
                        "detailed_description": "Visible content was analyzed without memory persistence.",
                        "inferred_user_activity": "reviewing a transient screen",
                    }
                )
            ]
        )
        screenshot = self.temp_path / "transient.png"
        screenshot.write_bytes(b"queued")
        service = PassiveObserverService(
            memory=self.memory,
            llm_provider=llm,
            screen_capture=FakeScreenCapture([]),
            screenshot_root=str(self.temp_path / "shots"),
            persist_observations=False,
        )

        observation = asyncio.run(
            service.process_screenshot(
                screenshot_path=str(screenshot),
                model="fallback-model",
                recent_context="",
                captured_at="2026-06-25T10:00:00",
                similarity_score=0.4,
            )
        )

        self.assertIsNotNone(observation)
        self.assertEqual(len(self.memory.get_recent_visual_observations(limit=5)), 0)

    def test_followup_creates_direct_todoist_reminder_from_hint(self):
        observation = VisualObservation(
            observation_id="obs-reminder",
            screenshot_path="whatsapp.png",
            created_at="2026-06-29T20:30:00",
            app_name="WhatsApp",
            summary="Coding challenge announcement is visible.",
            detailed_description="The screen shows Code Autopsy 1.0 starting tonight at 9 PM IST.",
            inferred_user_activity="checking coding challenge details",
            raw_payload_json=json.dumps(
                {
                    "maybe_require_a_reminder": True,
                    "reminder_context": "Code Autopsy 1.0 starts tonight, June 29, at 9:00 PM IST.",
                }
            ),
        )
        llm = FakeVisualLLM(
            [
                json.dumps({"unique_activities": ["checking coding challenge details"]}),
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
        reminder_helper = FakeReminderHelper()
        service = PassiveObserverFollowupService(
            memory=self.memory,
            task_queue=FakeTaskQueue(),
            llm_provider=llm,
            reminder_helper=reminder_helper,
        )

        result = asyncio.run(
            service.process_observations(
                observations=[observation],
                model="test-model",
                mark_sent=False,
                apply_memory_updates=False,
            )
        )

        self.assertEqual(
            reminder_helper.created,
            ["Code Autopsy 1.0 starts tonight, June 29, at 9:00 PM IST."],
        )
        self.assertEqual(
            result["direct_reminders"],
            ["Code Autopsy 1.0 starts tonight, June 29, at 9:00 PM IST."],
        )

    def test_process_screenshot_returns_none_when_file_is_missing(self):
        llm = FakeVisualLLM([json.dumps({})])
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

    def test_followup_service_processes_activity_pipeline(self):
        first = VisualObservation(
            observation_id="obs-1",
            screenshot_path="gmail-1.png",
            created_at="2026-06-25T10:00:00",
            app_name="Gmail",
            summary="A reply draft is open in Gmail.",
            detailed_description="Gmail compose window shows an unfinished reply draft addressed to a contact.",
            inferred_user_activity="writing an email reply",
            raw_payload_json=json.dumps(
                {
                    "maybe_require_a_reminder": True,
                    "reminder_context": "follow up on the pending reply if not sent today",
                }
            ),
        )
        second = VisualObservation(
            observation_id="obs-2",
            screenshot_path="gmail-2.png",
            created_at="2026-06-25T10:02:00",
            app_name="Gmail",
            summary="The same email reply draft remains open.",
            detailed_description="Gmail compose view is still open with the draft body partially written.",
            inferred_user_activity="Writing an email reply",
        )
        third = VisualObservation(
            observation_id="obs-3",
            screenshot_path="amazon.png",
            created_at="2026-06-25T10:03:00",
            app_name="Amazon",
            page_hint="TV listings",
            summary="Amazon TV product listing is visible.",
            detailed_description="TV product cards with pricing and ratings are visible for side-by-side comparison.",
            inferred_user_activity="comparing TV options on Amazon",
        )
        fourth = VisualObservation(
            observation_id="obs-4",
            screenshot_path="feed.png",
            created_at="2026-06-25T10:04:00",
            app_name="Instagram",
            summary="Instagram feed is open.",
            detailed_description="A vertically scrolling social feed is visible with posts and reels.",
            inferred_user_activity="scrolling through a social media feed",
        )
        for item in (first, second, third, fourth):
            self.memory.append_visual_observation(item)

        llm = FakeVisualLLM(
            [
                json.dumps(
                    {
                        "unique_activities": [
                            "writing an email reply",
                            "comparing TV options on Amazon",
                            "scrolling through a social media feed",
                        ]
                    }
                ),
                json.dumps(
                    {
                        "useful_activities": [
                            "writing an email reply",
                            "comparing TV options on Amazon",
                        ]
                    }
                ),
                json.dumps(
                    {
                        "action": "queue_task",
                        "task": "draft a reply email",
                        "memory_updates": ["User has an unfinished email reply that may need follow-up soon."],
                        "user_info_updates": [],
                    }
                ),
                json.dumps(
                    {
                        "action": "do_now",
                        "task": "search current TV prices",
                        "memory_updates": ["User may want a near-term TV price check."],
                        "user_info_updates": ["User compares TV options carefully before purchase."],
                    }
                ),
            ]
        )

        task_queue = FakeTaskQueue()
        followup = PassiveObserverFollowupService(
            memory=self.memory,
            task_queue=task_queue,
            llm_provider=llm,
        )
        result = asyncio.run(followup.maybe_queue_followup(model="test-model"))

        self.assertEqual(result["unique_activities"], [
            "writing an email reply",
            "comparing TV options on Amazon",
            "scrolling through a social media feed",
        ])
        self.assertEqual(result["useful_activities"], [
            "writing an email reply",
            "comparing TV options on Amazon",
        ])
        self.assertEqual(result["queued_activities"], ["draft a reply email"])
        self.assertEqual(result["do_now_activities"], ["search current TV prices"])
        self.assertEqual(
            result["memory_updates"],
            [
                "User has an unfinished email reply that may need follow-up soon.",
                "User may want a near-term TV price check.",
            ],
        )
        self.assertEqual(result["user_info_updates"], ["User compares TV options carefully before purchase."])
        self.assertIn("scrolling through a social media feed", result["ignored_activities"])
        self.assertEqual(len(task_queue.items), 1)
        self.assertIn("draft a reply email", task_queue.items[0].description)
        self.assertEqual(task_queue.items[0].metadata["task_kind"], "passive_observer_followup")
        self.assertEqual(task_queue.items[0].metadata["source"], "inferred_user_activity")
        self.assertEqual(task_queue.items[0].metadata["activity"], "writing an email reply")
        self.assertEqual(task_queue.items[0].metadata["task"], "draft a reply email")
        self.assertCountEqual(task_queue.items[0].metadata["source_observation_ids"], ["obs-1", "obs-2"])
        self.assertIn("User compares TV options carefully before purchase.", self.memory.get_user_info())
        self.assertIn("User has an unfinished email reply", self.memory.get_working_memory())
        self.assertIn("near-term TV price check", self.memory.get_working_memory())
        decision_payload = llm.calls[2]["payload"]
        self.assertEqual(decision_payload["activity"], "writing an email reply")
        self.assertTrue(all(item["app_name"] == "Gmail" for item in decision_payload["observation_context"]))
        detailed_text = " ".join(item["detailed_description"] for item in decision_payload["observation_context"])
        self.assertIn("unfinished reply draft", detailed_text)
        self.assertTrue(any(item["maybe_require_a_reminder"] for item in decision_payload["observation_context"]))
        reminder_text = " ".join(item["reminder_context"] for item in decision_payload["observation_context"])
        self.assertIn("follow up on the pending reply", reminder_text)
        for observation_id in ("obs-1", "obs-2", "obs-3", "obs-4"):
            persisted = self.memory.get_visual_observation(observation_id)
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
            confidence=0.8,
        )
        self.memory.append_visual_observation(older)
        self.memory.append_visual_observation(newer)

        llm = FakeVisualLLM(
            [
                json.dumps(
                    {
                        "unique_activities": ["comparing shortlisted products"],
                    }
                ),
                json.dumps({"useful_activities": []}),
            ]
        )
        followup = PassiveObserverFollowupService(
            memory=self.memory,
            task_queue=FakeTaskQueue(),
            llm_provider=llm,
        )

        result = asyncio.run(followup.maybe_queue_followup(model="test-model"))

        self.assertEqual(result["queued_activities"], [])
        payload = llm.calls[0]["payload"]
        self.assertEqual(payload["activities"], ["comparing shortlisted products"])
        persisted_unsent = self.memory.get_visual_observation("obs-unsent")
        persisted_sent = self.memory.get_visual_observation("obs-sent")
        self.assertIsNotNone(persisted_unsent)
        self.assertTrue(persisted_unsent.followup_sent_at)
        self.assertEqual(persisted_sent.followup_sent_at, "2026-06-25T10:05:00")

    def test_followup_service_skips_blank_activities(self):
        self.memory.append_visual_observation(
            VisualObservation(
                observation_id="obs-blank",
                screenshot_path="blank.png",
                created_at="2026-06-25T10:00:00",
                inferred_user_activity="   ",
            )
        )

        followup = PassiveObserverFollowupService(
            memory=self.memory,
            task_queue=FakeTaskQueue(),
            llm_provider=FakeVisualLLM([]),
        )
        result = asyncio.run(followup.maybe_queue_followup(model="test-model"))

        self.assertEqual(result["processed_observation_ids"], [])
        self.assertEqual(result["reason"], "no unsent inferred activities")

    def test_followup_service_filters_ephemeral_activity_via_llm(self):
        observation = VisualObservation(
            observation_id="obs-scroll",
            screenshot_path="scroll.png",
            created_at="2026-06-25T10:00:00",
            app_name="Instagram",
            page_hint="feed",
            summary="Instagram feed is open.",
            inferred_user_activity="scrolling the Instagram feed",
            confidence=0.8,
        )
        self.memory.append_visual_observation(observation)

        task_queue = FakeTaskQueue()
        followup = PassiveObserverFollowupService(
            memory=self.memory,
            task_queue=task_queue,
            llm_provider=FakeVisualLLM(
                [
                    json.dumps({"unique_activities": ["scrolling the Instagram feed"]}),
                    json.dumps({"useful_activities": []}),
                ]
            ),
        )
        result = asyncio.run(followup.maybe_queue_followup(model="test-model"))

        self.assertEqual(result["useful_activities"], [])
        self.assertEqual(len(task_queue.items), 0)

    def test_user_biodata_service_appends_entries_and_marks_biodata_sent(self):
        first = VisualObservation(
            observation_id="obs-bio-1",
            screenshot_path="study.png",
            created_at="2026-06-25T10:00:00",
            app_name="WhatsApp",
            summary="Chat mentions an upcoming PLISP test.",
            detailed_description="WhatsApp conversation includes a reminder about an upcoming PLISP test.",
            inferred_user_activity="reviewing messages about an upcoming test",
            followup_sent_at="2026-06-25T10:05:00",
        )
        second = VisualObservation(
            observation_id="obs-bio-2",
            screenshot_path="shop.png",
            created_at="2026-06-25T10:10:00",
            app_name="Amazon",
            summary="TV comparison is visible.",
            detailed_description="Several TV product cards are being compared by price and rating.",
            inferred_user_activity="comparing TV options online",
        )
        self.memory.append_visual_observation(first)
        self.memory.append_visual_observation(second)
        self.memory.save_user_info("Existing user note.\n")

        llm = FakeVisualLLM(
            [
                json.dumps(
                    {
                        "entries": [
                            {
                                "note": "User may be preparing for an upcoming PLISP test.",
                                "bucket": "memory",
                                "category": "education",
                                "confidence": 0.91,
                            },
                            {
                                "note": "User is actively comparing TVs before purchase.",
                                "bucket": "user_info",
                                "category": "interest",
                                "confidence": 0.88,
                            },
                        ]
                    }
                )
            ]
        )
        service = UserBioDataService(memory=self.memory, llm_provider=llm)

        result = asyncio.run(service.update_biodata(model="test-model"))

        self.assertEqual(result["processed_observation_ids"], ["obs-bio-2", "obs-bio-1"])
        self.assertEqual(len(result["entries"]), 2)
        user_info = self.memory.get_user_info()
        working_memory = self.memory.get_working_memory()
        self.assertIn("Existing user note.", user_info)
        self.assertIn("[interest] User is actively comparing TVs before purchase.", user_info)
        self.assertIn("[education] User may be preparing for an upcoming PLISP test.", working_memory)
        persisted_first = self.memory.get_visual_observation("obs-bio-1")
        persisted_second = self.memory.get_visual_observation("obs-bio-2")
        self.assertIsNotNone(persisted_first)
        self.assertIsNotNone(persisted_second)
        self.assertTrue(persisted_first.biodata_sent_at)
        self.assertTrue(persisted_second.biodata_sent_at)
        self.assertEqual(persisted_first.followup_sent_at, "2026-06-25T10:05:00")

    def test_user_biodata_service_skips_already_processed_biodata_observations(self):
        done = VisualObservation(
            observation_id="obs-done",
            screenshot_path="done.png",
            created_at="2026-06-25T10:00:00",
            inferred_user_activity="reading about internships",
            biodata_sent_at="2026-06-25T10:04:00",
        )
        pending = VisualObservation(
            observation_id="obs-pending",
            screenshot_path="pending.png",
            created_at="2026-06-25T10:05:00",
            inferred_user_activity="researching internship options",
        )
        self.memory.append_visual_observation(done)
        self.memory.append_visual_observation(pending)

        llm = FakeVisualLLM(
            [
                json.dumps(
                    {
                        "entries": [
                            {
                                "note": "User is interested in internship opportunities.",
                                "bucket": "user_info",
                                "category": "work",
                                "confidence": 0.82,
                            }
                        ]
                    }
                )
            ]
        )
        service = UserBioDataService(memory=self.memory, llm_provider=llm)

        result = asyncio.run(service.update_biodata(model="test-model"))

        self.assertEqual(result["processed_observation_ids"], ["obs-pending"])
        payload = llm.calls[0]["payload"]
        self.assertEqual([item["observation_id"] for item in payload["observations"]], ["obs-pending"])

if __name__ == "__main__":
    unittest.main()
