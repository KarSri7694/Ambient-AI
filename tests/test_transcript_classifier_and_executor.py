import asyncio
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from application.services.simple_task_execution_service import SimpleTaskExecutionService
from application.services.transcript_classification_service import TranscriptClassificationService
from core.models import TranscriptClassificationResult, TranscriptParticipant


class FakeLLMService:
    def __init__(self):
        self.calls = []

    async def run_interaction(self, **kwargs):
        self.calls.append(kwargs)
        return "executor-finished"


class TranscriptClassifierAndExecutorTests(unittest.TestCase):
    def test_classifier_returns_preference(self):
        classifier = TranscriptClassificationService(llm_provider=None)
        participants = {
            "USER": TranscriptParticipant(
                speaker_label="USER",
                speaker_id="speaker-1",
                display_name="USER",
                confidence=1.0,
            )
        }
        transcript = "[0.0000 - 1.5000] -> USER: I like black coffee very much\n"

        result = asyncio.run(
            classifier.classify(
                transcript_text=transcript,
                participants=participants,
                system_prompt="unused",
                model="unused",
            )
        )

        self.assertEqual(result.label, "PREFERENCE")
        self.assertEqual(result.speaker_label, "USER")
        self.assertIn("black coffee", result.memory_content)

    def test_classifier_returns_reminder(self):
        classifier = TranscriptClassificationService(llm_provider=None)
        participants = {
            "USER": TranscriptParticipant(
                speaker_label="USER",
                speaker_id="speaker-1",
                display_name="USER",
                confidence=1.0,
            )
        }
        transcript = "[0.0000 - 1.5000] -> USER: tomorrow remind me to call mom\n"

        result = asyncio.run(
            classifier.classify(
                transcript_text=transcript,
                participants=participants,
                system_prompt="unused",
                model="unused",
            )
        )

        self.assertEqual(result.label, "REMINDER")
        self.assertIn("call mom", result.summary.lower())

    def test_classifier_returns_complex_task(self):
        classifier = TranscriptClassificationService(llm_provider=None)
        participants = {
            "SPEAKER 01": TranscriptParticipant(
                speaker_label="SPEAKER 01",
                speaker_id="speaker-2",
                display_name="Friend",
                confidence=1.0,
            )
        }
        transcript = "[0.0000 - 1.5000] -> SPEAKER 01: please send me the resources and notes to learn GRPO\n"

        result = asyncio.run(
            classifier.classify(
                transcript_text=transcript,
                participants=participants,
                system_prompt="unused",
                model="unused",
            )
        )

        self.assertEqual(result.label, "TASK_COMPLEX")
        self.assertIn("GRPO", result.summary)

    def test_simple_executor_uses_restricted_tools(self):
        llm_service = FakeLLMService()
        executor = SimpleTaskExecutionService(llm_service=llm_service)
        classification = TranscriptClassificationResult(
            label="REMINDER",
            speaker_label="USER",
            summary="Tomorrow remind me to call mom.",
            confidence=0.8,
            reason="matched_reminder_tokens",
            suggested_action="Create a reminder to call mom tomorrow.",
        )

        result = asyncio.run(
            executor.execute(
                classification=classification,
                transcript_text="[0.0 - 1.0] -> USER: tomorrow remind me to call mom",
                system_prompt="executor prompt",
                model="unused",
            )
        )

        self.assertEqual(result, "executor-finished")
        self.assertEqual(len(llm_service.calls), 1)
        call = llm_service.calls[0]
        self.assertEqual(
            call["allowed_tool_names"],
            {"add_task", "schedule_meeting", "queue_night_task"},
        )
        self.assertEqual(call["system_prompt"], "executor prompt")


if __name__ == "__main__":
    unittest.main()
