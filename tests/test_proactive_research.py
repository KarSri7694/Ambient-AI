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

from application.services.activity_ledger_service import ActivityLedgerService
from application.services.proactive_research_service import ProactiveResearchService
from application.services.proactive_topic_detection_service import ProactiveTopicDetectionService
from application.services.research_vault_service import ResearchVaultService
from core.models import TranscriptClassificationResult
from infrastructure.adapter.SQLiteActivityLedgerAdapter import SQLiteActivityLedgerAdapter
from infrastructure.adapter.SQLiteProactiveTopicQueueAdapter import SQLiteProactiveTopicQueueAdapter


class FakeNotificationStore:
    def __init__(self):
        self.messages = []

    def add_notification(self, message: str, source: str = "system") -> None:
        self.messages.append((message, source))


class FakeLLMService:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.calls = []

    async def run_interaction(self, **kwargs):
        self.calls.append(kwargs)
        return self.response_text


class ProactiveResearchTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.queue = SQLiteProactiveTopicQueueAdapter(str(self.temp_path / "topics.db"))
        self.vault = ResearchVaultService(str(self.temp_path / "vault"))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_detection_queues_salient_topic(self):
        detector = ProactiveTopicDetectionService(queue=self.queue, vault=self.vault)
        classification = TranscriptClassificationResult(
            label="TASK_COMPLEX",
            speaker_label="USER",
            summary="Please send me the resources to learn the new game Clair Obscur.",
            confidence=0.8,
            reason="matched_complex_task_tokens",
            suggested_action="Research Clair Obscur and learning resources.",
        )

        candidate = detector.maybe_queue_topic(
            classification=classification,
            transcript_text="transcript content",
            source_ref="sample.txt",
        )

        self.assertIsNotNone(candidate)
        self.assertIn("clair-obscur", candidate.normalized_topic)
        pending = self.queue.get_pending_topics()
        self.assertEqual(len(pending), 1)

    def test_repeat_topic_updates_existing_candidate(self):
        detector = ProactiveTopicDetectionService(queue=self.queue, vault=self.vault)
        classification = TranscriptClassificationResult(
            label="PREFERENCE",
            speaker_label="USER",
            summary="I like the game Hades 2.",
            confidence=0.75,
            reason="matched_preference_tokens",
            memory_content="I like the game Hades 2.",
        )

        first = detector.maybe_queue_topic(classification, "text 1", "ref1.txt")
        second = detector.maybe_queue_topic(classification, "text 2", "ref2.txt")

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(first.topic_id, second.topic_id)
        self.assertEqual(len(self.queue.get_pending_topics()), 1)

    def test_research_service_creates_vault_package_and_notification(self):
        detector = ProactiveTopicDetectionService(queue=self.queue, vault=self.vault)
        classification = TranscriptClassificationResult(
            label="FACT",
            speaker_label="USER",
            summary="Someone mentioned the game Blue Prince.",
            confidence=0.7,
            reason="long_statement_fallback",
            memory_content="Blue Prince is a game worth checking out.",
        )
        candidate = detector.maybe_queue_topic(classification, "text", "source.txt")
        self.assertIsNotNone(candidate)

        llm_service = FakeLLMService(
            json.dumps(
                {
                    "summary": "Blue Prince is a puzzle exploration game.",
                    "notes": "Interesting because it mixes mystery and exploration.",
                    "links": [{"title": "Official Site", "url": "https://example.com/blue-prince"}],
                }
            )
        )
        notifications = FakeNotificationStore()
        service = ProactiveResearchService(
            llm_service=llm_service,
            topic_queue=self.queue,
            vault=self.vault,
            notifications=notifications,
            activity_ledger=ActivityLedgerService(
                SQLiteActivityLedgerAdapter(db_path=str(self.temp_path / "activity_ledger.db"))
            ),
        )

        worked = asyncio.run(service.process_next_topic(system_prompt="research prompt", model="unused"))

        self.assertTrue(worked)
        pending = self.queue.get_pending_topics()
        self.assertEqual(len(pending), 0)
        artifact_path = self.vault.get_existing_artifact_path(candidate.normalized_topic)
        self.assertIsNotNone(artifact_path)
        summary_path = Path(artifact_path) / "summary.md"
        self.assertTrue(summary_path.exists())
        self.assertIn("Blue Prince", summary_path.read_text(encoding="utf-8"))
        self.assertEqual(len(notifications.messages), 1)
        self.assertEqual(
            llm_service.calls[0]["allowed_tool_names"],
            {"google_search", "powershell_terminal", "queue_night_task"},
        )
        runs = service.activity_ledger.ledger.list_runs(source_kind="proactive_research")
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].status, "completed")
        detail = service.activity_ledger.ledger.get_run_detail(runs[0].run_id)
        self.assertIsNotNone(detail)
        self.assertEqual(len(detail.artifacts), 3)


if __name__ == "__main__":
    unittest.main()
