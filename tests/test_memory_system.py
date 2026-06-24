import asyncio
import sys
import tempfile
import unittest
import uuid
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from application.services.memory_consolidation_service import MemoryConsolidationService
from application.services.memory_context_builder import MemoryContextBuilder
from application.services.memory_extraction_service import MemoryExtractionService
from application.services.speaker_resolution_service import SpeakerResolutionService
from core.models import MemoryEvent
from infrastructure.adapter.SQLiteMemoryAdapter import SQLiteMemoryAdapter


class MemorySystemTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.prompts_dir = self.temp_path / "prompts"
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        (self.prompts_dir / "AGENT.md").write_text("Agent prompt", encoding="utf-8")
        (self.prompts_dir / "USER.md").write_text("User prompt", encoding="utf-8")

        self.memory = SQLiteMemoryAdapter(
            db_path=str(self.temp_path / "memory.db"),
            memory_root=str(self.temp_path / "memory"),
        )
        self.memory.save_session_digest("# Session Digest\n\n- Existing session\n")
        self.memory.save_open_loop_digest("# Open Loops\n\n- Existing loop\n")
        self.memory.save_visual_digest("# Passive Visual Context\n\n- Existing visual context\n")
        self.memory.save_user_info("# USER_INFO\n\n- Existing durable user fact\n")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_context_builder_includes_profiles_and_recent_context(self):
        speaker = self.memory.upsert_speaker("Alice", "Alice")
        self.memory.save_recent_context("# Recent Shared Context\n\n- Shared context item\n")
        self.memory.append_event(
            MemoryEvent(
                event_id=uuid.uuid4().hex,
                speaker_id=speaker.speaker_id,
                source_type="transcript",
                source_ref="sample.txt",
                event_kind="preference",
                content="Alice likes black coffee.",
                confidence=0.8,
                status="candidate",
                created_at=datetime.now().isoformat(),
            )
        )

        builder = MemoryContextBuilder(self.memory, str(self.prompts_dir))
        participants = [
            type("Participant", (), {
                "speaker_label": "Alice",
                "speaker_id": speaker.speaker_id,
                "display_name": "Alice",
            })()
        ]
        prompt = builder.build_prompt(
            base_prompt_filename="AGENT.md",
            skills_summary="Skill summary",
            participants=participants,
        )

        self.assertIn("Agent prompt", prompt)
        self.assertIn("User prompt", prompt)
        self.assertIn("Existing session", prompt)
        self.assertIn("Existing loop", prompt)
        self.assertIn("Existing visual context", prompt)
        self.assertIn("Existing durable user fact", prompt)
        self.assertIn("Shared context item", prompt)
        self.assertIn("Alice likes black coffee.", prompt)
        self.assertIn("Skill summary", prompt)

    def test_speaker_resolution_reuses_named_speaker(self):
        existing = self.memory.upsert_speaker("Kartikeya Srivastava", "USER")
        resolver = SpeakerResolutionService(self.memory)

        resolved = resolver.resolve_labels(
            ["Kartikeya Srivastava - [98.000%]", "SPEAKER_00 - [31.000%]"],
            source_ref="transcript_001",
        )

        self.assertEqual(
            resolved["Kartikeya Srivastava - [98.000%]"].speaker_id,
            existing.speaker_id,
        )
        self.assertEqual(
            resolved["SPEAKER_00 - [31.000%]"].display_name,
            "SPEAKER_00",
        )
        self.assertFalse(resolved["SPEAKER_00 - [31.000%]"].durable)

    def test_memory_extraction_heuristics_create_candidate_events(self):
        speaker = self.memory.upsert_speaker("USER", "USER")
        extractor = MemoryExtractionService(llm_provider=None)
        participants = {
            "USER": type("Participant", (), {
                "speaker_label": "USER",
                "speaker_id": speaker.speaker_id,
                "display_name": "USER",
            })()
        }
        transcript = "[0.0000 - 1.5000] -> USER: I will send the GRPO notes tomorrow\n"

        events = asyncio.run(
            extractor.extract_events(
                transcript_text=transcript,
                participants=participants,
                model="unused",
                source_ref="transcript.txt",
            )
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_kind, "commitment")
        self.assertEqual(events[0].status, "candidate")

    def test_consolidation_dedupes_and_updates_profile(self):
        speaker = self.memory.upsert_speaker("Bob", "Bob")
        now = datetime.now().isoformat()
        for _ in range(2):
            self.memory.append_event(
                MemoryEvent(
                    event_id=uuid.uuid4().hex,
                    speaker_id=speaker.speaker_id,
                    source_type="transcript",
                    source_ref="sample.txt",
                    event_kind="project",
                    content="Bob is building an ambient AI memory system.",
                    confidence=0.9,
                    status="candidate",
                    created_at=now,
                )
            )

        consolidator = MemoryConsolidationService(self.memory)
        processed_count = consolidator.consolidate()

        facts = self.memory.get_facts(speaker.speaker_id)
        profile = self.memory.get_speaker_profile(speaker.speaker_id)
        recent_context = self.memory.get_recent_context()
        consolidated_events = self.memory.get_recent_events(
            speaker_ids=[speaker.speaker_id],
            status="consolidated",
            limit=10,
        )

        self.assertEqual(processed_count, 2)
        self.assertEqual(len(facts), 1)
        self.assertIn("ambient AI memory system", profile)
        self.assertIn("Bob: Bob is building an ambient AI memory system.", recent_context)
        self.assertEqual(len(consolidated_events), 2)


if __name__ == "__main__":
    unittest.main()
