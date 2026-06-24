import asyncio
from dataclasses import replace
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from application.services.memory_context_builder import MemoryContextBuilder
from application.services.open_loop_service import OpenLoopService
from application.services.session_tracker_service import SessionTrackerService
from application.services.transcript_evidence_service import TranscriptEvidenceService
from application.services.user_profile_service import UserProfileService
from core.models import TranscriptParticipant, TranscriptTurn
from infrastructure.adapter.SQLiteMemoryAdapter import SQLiteMemoryAdapter


class AudioAmbientStateTests(unittest.TestCase):
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
        self.user = self.memory.upsert_speaker("USER", "USER", is_user=True)
        self.participants = {
            "USER": TranscriptParticipant(
                speaker_label="USER",
                speaker_id=self.user.speaker_id,
                display_name="USER",
                confidence=1.0,
            )
        }
        self.evidence_service = TranscriptEvidenceService()
        self.session_tracker = SessionTrackerService(self.memory)
        self.loop_service = OpenLoopService(self.memory)
        self.profile_service = UserProfileService(self.memory)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_session_loop_and_profile_flow(self):
        first_turns = [
            TranscriptTurn(
                speaker_label="USER",
                text="Tomorrow remind me to call mom about the Amazon order.",
                start_time=0.0,
                end_time=2.0,
            )
        ]
        first_evidence = asyncio.run(
            self.evidence_service.extract(
                turns=first_turns,
                participants=self.participants,
                source_ref="transcript_1.txt",
            )
        )
        first_session = asyncio.run(self.session_tracker.attach_to_session(first_evidence))
        first_evidence = [replace(item, session_id=first_session.session_id) for item in first_evidence]
        for item in first_evidence:
            self.memory.append_evidence(item)
        loops = asyncio.run(self.loop_service.process(session=first_session, evidence_items=first_evidence))
        facets = asyncio.run(
            self.profile_service.update_from_evidence(
                user_speaker=self.profile_service.infer_user_speaker(),
                evidence_items=first_evidence,
            )
        )

        self.assertEqual(len(loops), 1)
        self.assertEqual(loops[0].status, "open")
        self.assertEqual(len(facets), 1)
        self.assertEqual(facets[0].category, "current_obligations")

        second_turns = [
            TranscriptTurn(
                speaker_label="USER",
                text="I already called mom, that is done.",
                start_time=3.0,
                end_time=5.0,
            )
        ]
        second_evidence = asyncio.run(
            self.evidence_service.extract(
                turns=second_turns,
                participants=self.participants,
                source_ref="transcript_2.txt",
            )
        )
        second_session = asyncio.run(self.session_tracker.attach_to_session(second_evidence))
        second_evidence = [replace(item, session_id=second_session.session_id) for item in second_evidence]
        resolved = asyncio.run(self.loop_service.process(session=second_session, evidence_items=second_evidence))

        open_loops = self.memory.list_open_loops(statuses=["resolved"], limit=5)
        self.assertEqual(first_session.session_id, second_session.session_id)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(len(open_loops), 1)
        self.assertEqual(open_loops[0].status, "resolved")

    def test_context_builder_includes_session_and_loop_digests(self):
        turns = [
            TranscriptTurn(
                speaker_label="USER",
                text="I will finish the email draft tonight.",
                start_time=0.0,
                end_time=2.0,
            )
        ]
        evidence = asyncio.run(
            self.evidence_service.extract(
                turns=turns,
                participants=self.participants,
                source_ref="transcript_3.txt",
            )
        )
        session = asyncio.run(self.session_tracker.attach_to_session(evidence))
        evidence = [replace(item, session_id=session.session_id) for item in evidence]
        asyncio.run(self.loop_service.process(session=session, evidence_items=evidence))
        self.session_tracker.refresh_digest()
        self.profile_service.refresh_user_profile(self.user.speaker_id)

        builder = MemoryContextBuilder(self.memory, str(self.prompts_dir))
        prompt = builder.build_prompt(
            base_prompt_filename="AGENT.md",
            skills_summary="Skill summary",
            participants=list(self.participants.values()),
        )

        self.assertIn("Session Continuity", prompt)
        self.assertIn("Open Loops", prompt)
        self.assertIn("finish the email draft tonight", prompt)


if __name__ == "__main__":
    unittest.main()
