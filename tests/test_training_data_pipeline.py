import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from application.services.training_data_service import TrainingDataService
from core.models import InteractionLogEntry
from infrastructure.adapter.SQLiteInteractionLogAdapter import SQLiteInteractionLogAdapter
from infrastructure.adapter.SQLiteTrainingDataAdapter import SQLiteTrainingDataAdapter


class TrainingDataPipelineTests(unittest.TestCase):
    def test_sync_review_and_export_llm_and_asr_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            user_data = root / "USER_DATA"
            uploads = user_data / "uploads"
            transcripts = user_data / "transcriptions"
            cleaned = user_data / "cleaned_audio"
            uploads.mkdir(parents=True)
            transcripts.mkdir(parents=True)
            cleaned.mkdir(parents=True)

            training_root = root / "TRAINING_DATA"
            interaction_db = user_data / "database" / "interaction_logs.db"
            training_db = training_root / "database" / "training_data.db"

            interaction_store = SQLiteInteractionLogAdapter(str(interaction_db))
            training_store = SQLiteTrainingDataAdapter(str(training_db))
            service = TrainingDataService(
                store=training_store,
                interaction_store=interaction_store,
                training_root=str(training_root),
                user_data_dir=str(user_data),
                uploads_dir=str(uploads),
                transcripts_dir=str(transcripts),
                cleaned_audio_dir=str(cleaned),
            )

            interaction_store.insert(
                InteractionLogEntry(
                    interaction_id="interaction-1",
                    interaction_run_id="run-1",
                    created_at="2026-06-28T10:00:00",
                    completed_at="2026-06-28T10:00:05",
                    source="passive_observer",
                    model="model-a",
                    messages_json=json.dumps([{"role": "user", "content": "look at this screenshot"}]),
                    tools_json=json.dumps([{"type": "function", "function": {"name": "search"}}]),
                    image_path=str(root / "screen.png"),
                    response_text="original output",
                    reasoning_text="original reasoning",
                    tool_calls_json=json.dumps([{"id": "call-1", "function": {"name": "search", "arguments": "{}"}}]),
                    error_text=None,
                    duration_ms=1234,
                    metadata_json=json.dumps({"source_id": "abc"}),
                    report_json=json.dumps({"title": "Observation"}),
                )
            )

            upload_audio = uploads / "audio_record_001.wav"
            upload_audio.write_bytes(b"wav")
            cleaned_audio = cleaned / "audio_record_001_final.wav"
            cleaned_audio.write_bytes(b"wav")
            transcript = transcripts / "transcript_28062026_100010.txt"
            transcript.write_text("[0.0 - 1.0] -> USER: hello there", encoding="utf-8")

            llm_sync = service.sync_llm_records()
            asr_sync = service.sync_asr_records()

            self.assertEqual(llm_sync["synced"], 1)
            self.assertEqual(asr_sync["synced"], 1)

            llm_record = training_store.list_llm_records(limit=10)[0]
            asr_record = training_store.list_asr_records(limit=10)[0]

            training_store.upsert_llm_review(
                record_id=llm_record.record_id,
                reviewer="tester",
                status="approved",
                corrected_response_text="corrected output",
                corrected_reasoning_text="corrected reasoning",
                corrected_messages_json=None,
                notes="good record",
                created_at="2026-06-28T11:00:00",
                updated_at="2026-06-28T11:00:00",
            )
            training_store.upsert_asr_review(
                record_id=asr_record.record_id,
                reviewer="tester",
                status="approved",
                corrected_transcript_text="hello there corrected",
                notes="fixed punctuation",
                created_at="2026-06-28T11:00:00",
                updated_at="2026-06-28T11:00:00",
            )

            llm_export = service.export_llm_dataset()
            asr_export = service.export_asr_dataset()

            self.assertEqual(llm_export["record_count"], 1)
            self.assertEqual(asr_export["record_count"], 1)

            llm_lines = Path(llm_export["output_path"]).read_text(encoding="utf-8").splitlines()
            asr_lines = Path(asr_export["output_path"]).read_text(encoding="utf-8").splitlines()
            self.assertEqual(json.loads(llm_lines[0])["assistant_response"], "corrected output")
            self.assertEqual(json.loads(asr_lines[0])["transcript"], "hello there corrected")
            self.assertEqual(training_store.list_exports(limit=10)[0].dataset_kind, "asr")


if __name__ == "__main__":
    unittest.main()
