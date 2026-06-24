import asyncio
import json
import sys
import tempfile
import unittest
import uuid
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from application.services.agenda_scoring_service import AgendaScoringService
from application.services.ambient_reflection_service import AmbientReflectionService
from core.models import AmbientAgendaItem, MemoryFact, Notification, NightTask, ProactiveTopicCandidate
from infrastructure.adapter.SQLiteAmbientAgendaAdapter import SQLiteAmbientAgendaAdapter
from infrastructure.adapter.SQLiteMemoryAdapter import SQLiteMemoryAdapter
from infrastructure.adapter.SQLiteProactiveTopicQueueAdapter import SQLiteProactiveTopicQueueAdapter


class FakeLLMService:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.calls = []

    async def run_interaction(self, **kwargs):
        self.calls.append(kwargs)
        return self.response_text


class FakeNotificationPort:
    def __init__(self):
        self.peeked = []
        self.sent = []

    def peek_unread_notifications(self):
        return list(self.peeked)

    def get_unread_notifications(self):
        return list(self.peeked)

    def add_notification(self, message: str, source: str = "system") -> None:
        self.sent.append(Notification(id=len(self.sent) + 1, message=message, source=source))

    def mark_read(self, notification_id: int) -> None:
        return None


class FakeTaskQueue:
    def __init__(self):
        self.pending = []
        self.added = []

    def get_pending_tasks(self):
        return list(self.pending)

    def add_task(self, description: str, priority: str = "medium") -> str:
        self.added.append((description, priority))
        return "Task queued."

    def mark_task_complete(self, task_id: int, status: str = "completed") -> None:
        return None


class AmbientReflectionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.memory = SQLiteMemoryAdapter(
            db_path=str(self.temp_path / "memory.db"),
            memory_root=str(self.temp_path / "memory"),
        )
        self.agenda = SQLiteAmbientAgendaAdapter(str(self.temp_path / "agenda.db"))
        self.topics = SQLiteProactiveTopicQueueAdapter(str(self.temp_path / "topics.db"))
        self.notifications = FakeNotificationPort()
        self.task_queue = FakeTaskQueue()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_scoring_raises_stale_commitment_above_low_value_topic(self):
        speaker = self.memory.upsert_speaker("User", "USER", is_user=True)
        fact = MemoryFact(
            fact_id=uuid.uuid4().hex,
            speaker_id=speaker.speaker_id,
            fact_text="I will send the GRPO notes tomorrow.",
            topic="commitment",
            valid_from=(datetime.now() - timedelta(days=1)).isoformat(),
            source_event_ids=[],
            updated_at=datetime.now().isoformat(),
        )
        self.memory.upsert_fact(fact)
        pending_topic = ProactiveTopicCandidate(
            topic_id=uuid.uuid4().hex,
            normalized_topic="generic-topic",
            display_title="Generic Topic",
            source_ref="source.txt",
            speaker_label="USER",
            summary_hint="Something low value",
            salience_score=0.1,
            status="pending",
            first_seen_at=datetime.now().isoformat(),
            last_seen_at=datetime.now().isoformat(),
        )

        scorer = AgendaScoringService(self.agenda)
        ranked = scorer.build_top_candidates(
            facts=[fact],
            topics=[pending_topic],
            open_agenda_items=[],
            pending_tasks=[],
            unread_notifications=[],
            limit=5,
        )

        self.assertGreaterEqual(len(ranked), 2)
        self.assertEqual(ranked[0].candidate_type, "stale_commitment")

    def test_new_research_package_creates_one_agenda_item_without_duplicates(self):
        llm = FakeLLMService(
            json.dumps(
                {
                    "actions": [
                        {
                            "action_type": "CREATE_AGENDA_ITEM",
                            "payload": {
                                "candidate_id": "topic:topic-1",
                            },
                        }
                    ]
                }
            )
        )
        service = AmbientReflectionService(
            llm_service=llm,
            memory=self.memory,
            agenda=self.agenda,
            notifications=self.notifications,
            task_queue=self.task_queue,
            topic_queue=self.topics,
        )
        self.topics.upsert_topic(
            ProactiveTopicCandidate(
                topic_id="topic-1",
                normalized_topic="blue-prince",
                display_title="Blue Prince",
                source_ref="transcript-1",
                speaker_label="USER",
                summary_hint="A game worth checking out.",
                salience_score=0.8,
                status="completed",
                first_seen_at=datetime.now().isoformat(),
                last_seen_at=datetime.now().isoformat(),
                artifact_path=str(self.temp_path / "vault" / "blue-prince"),
                last_researched_at=datetime.now().isoformat(),
            )
        )

        asyncio.run(service.reflect(model="unused", recent_context="Recent shared context"))
        asyncio.run(service.reflect(model="unused", recent_context="Recent shared context"))

        items = self.agenda.list_items(limit=10)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].kind, "research_review")

    def test_surfaced_items_are_not_repeated_every_cycle(self):
        agenda_item = AmbientAgendaItem(
            agenda_id="agenda-1",
            title="Follow up on GRPO notes",
            kind="stale_commitment",
            source_type="memory_fact",
            source_ref="fact-1",
            priority_score=0.9,
            status="surfaced",
            created_at=(datetime.now() - timedelta(days=1)).isoformat(),
            updated_at=datetime.now().isoformat(),
            due_at=None,
            last_considered_at=datetime.now().isoformat(),
            backing_topic_id=None,
            backing_memory_ids=["fact-1"],
            surface_message="You committed to sending GRPO notes and there is still no completion signal.",
        )
        self.agenda.create_item(agenda_item)
        llm = FakeLLMService(
            json.dumps(
                {
                    "actions": [
                        {
                            "action_type": "SURFACE_ITEM",
                            "payload": {
                                "agenda_id": "agenda-1",
                                "message": "Repeat this reminder.",
                            },
                        }
                    ]
                }
            )
        )
        service = AmbientReflectionService(
            llm_service=llm,
            memory=self.memory,
            agenda=self.agenda,
            notifications=self.notifications,
            task_queue=self.task_queue,
            topic_queue=self.topics,
        )

        actions = asyncio.run(service.reflect(model="unused", recent_context="ctx"))

        self.assertEqual([action.action_type for action in actions], ["NOTHING"])
        self.assertEqual(len(self.notifications.sent), 0)

    def test_reflection_output_parser_accepts_only_closed_action_set(self):
        service = AmbientReflectionService(
            llm_service=FakeLLMService("{}"),
            memory=self.memory,
            agenda=self.agenda,
            notifications=self.notifications,
            task_queue=self.task_queue,
            topic_queue=self.topics,
        )

        actions = service.parse_actions(
            json.dumps(
                {
                    "actions": [
                        {"action_type": "SURFACE_ITEM", "payload": {"agenda_id": "1"}},
                        {"action_type": "HACK_THE_PLANET", "payload": {}},
                        {"action_type": "QUEUE_COMPLEX_TASK", "payload": {"title": "A", "description": "B"}},
                        {"action_type": "CREATE_SIMPLE_TASK", "payload": {"title": "C", "description": "D"}},
                    ]
                }
            )
        )

        self.assertEqual(
            [action.action_type for action in actions],
            ["SURFACE_ITEM", "QUEUE_COMPLEX_TASK"],
        )


if __name__ == "__main__":
    unittest.main()
