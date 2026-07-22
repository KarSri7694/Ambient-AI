import logging
import json
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from infrastructure.runtime_log_server import RuntimeLogBuffer, RuntimeLogBufferHandler, create_runtime_log_app
from application.services.training_data_service import TrainingDataService
from core.models import InteractionLogEntry
from infrastructure.adapter.SQLiteInteractionLogAdapter import SQLiteInteractionLogAdapter
from infrastructure.adapter.SQLiteTrainingDataAdapter import SQLiteTrainingDataAdapter


class RuntimeLogServerTests(unittest.TestCase):
    def test_api_returns_recent_buffered_logs(self):
        buffer = RuntimeLogBuffer(max_entries=10)
        handler = RuntimeLogBufferHandler(buffer)
        handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))

        logger = logging.getLogger("runtime-log-test")
        logger.setLevel(logging.INFO)
        logger.handlers = []
        logger.propagate = False
        logger.addHandler(handler)

        logger.info("first")
        logger.warning("second")

        app = create_runtime_log_app(buffer)
        client = TestClient(app)
        response = client.get("/api/logs")
        payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["count"], 2)
        self.assertEqual(payload["entries"][0]["message"], "INFO:first")
        self.assertEqual(payload["entries"][1]["message"], "WARNING:second")

    def test_api_filters_by_after_id(self):
        buffer = RuntimeLogBuffer(max_entries=10)
        fake_record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello",
            args=(),
            exc_info=None,
        )
        first = buffer.append(fake_record, "one")
        second = buffer.append(fake_record, "two")

        app = create_runtime_log_app(buffer)
        client = TestClient(app)
        response = client.get(f"/api/logs?after_id={first['id']}")
        payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual([entry["message"] for entry in payload["entries"]], ["two"])
        self.assertEqual(payload["latest_id"], second["id"])

    def test_dashboard_routes_and_static_assets_are_served(self):
        buffer = RuntimeLogBuffer(max_entries=10)
        app = create_runtime_log_app(buffer)
        client = TestClient(app)

        for route in ["/", "/chat", "/reports", "/logs", "/benchmarks", "/training"]:
            response = client.get(route)
            self.assertEqual(response.status_code, 200)
            self.assertIn("Ambient Agent", response.text)
            self.assertIn("/runtime-ui/styles.css", response.text)
            self.assertIn("/runtime-ui/dashboard.js", response.text)

        css_response = client.get("/runtime-ui/styles.css")
        js_response = client.get("/runtime-ui/dashboard.js")

        self.assertEqual(css_response.status_code, 200)
        self.assertIn(".topbar", css_response.text)
        self.assertEqual(js_response.status_code, 200)
        self.assertIn("function renderReports", js_response.text)

    def test_training_media_and_record_endpoints_work(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            user_data = root / "USER_DATA"
            uploads = user_data / "uploads"
            transcripts = user_data / "transcriptions"
            cleaned = user_data / "cleaned_audio"
            uploads.mkdir(parents=True)
            transcripts.mkdir(parents=True)
            cleaned.mkdir(parents=True)

            interaction_store = SQLiteInteractionLogAdapter(str(user_data / "database" / "interaction_logs.db"))
            training_store = SQLiteTrainingDataAdapter(str(root / "TRAINING_DATA" / "database" / "training.db"))
            service = TrainingDataService(
                store=training_store,
                interaction_store=interaction_store,
                training_root=str(root / "TRAINING_DATA"),
                user_data_dir=str(user_data),
                uploads_dir=str(uploads),
                transcripts_dir=str(transcripts),
                cleaned_audio_dir=str(cleaned),
            )

            image_path = user_data / "shot.png"
            image_path.write_bytes(b"png")
            interaction_store.insert(
                InteractionLogEntry(
                    interaction_id="interaction-1",
                    interaction_run_id="run-1",
                    created_at="2026-06-28T10:00:00",
                    completed_at="2026-06-28T10:00:01",
                    source="passive_observer",
                    model="model-a",
                    messages_json=json.dumps([{"role": "user", "content": "describe"}]),
                    image_path=str(image_path),
                    response_text="output",
                )
            )
            service.sync_llm_records()

            transcript_path = transcripts / "transcript_28062026_100010.txt"
            transcript_path.write_text("hello", encoding="utf-8")
            audio_path = cleaned / "audio_record_001_final.wav"
            audio_path.write_bytes(b"wav")
            service.sync_asr_records()

            buffer = RuntimeLogBuffer(max_entries=10)
            app = create_runtime_log_app(
                buffer,
                report_store=interaction_store,
                training_store=training_store,
                training_service=service,
                media_roots=[str(user_data), str(root / "TRAINING_DATA")],
            )
            client = TestClient(app)

            llm_records = client.get("/api/training/llm").json()["records"]
            asr_records = client.get("/api/training/asr").json()["records"]

            self.assertEqual(len(llm_records), 1)
            self.assertEqual(len(asr_records), 1)

            llm_detail = client.get(f"/api/training/llm/{llm_records[0]['record_id']}")
            media = client.get("/api/training/media", params={"path": str(image_path)})
            filtered_llm = client.get(
                "/api/training/llm",
                params={"review_status": "pending", "source": "passive_observer", "model": "model-a"},
            )
            filtered_asr = client.get("/api/training/asr", params={"review_status": "pending"})

            self.assertEqual(llm_detail.status_code, 200)
            self.assertEqual(llm_detail.json()["record"]["messages"][0]["content"], "describe")
            self.assertEqual(media.status_code, 200)
            self.assertEqual(media.content, b"png")
            self.assertEqual(filtered_llm.status_code, 200)
            self.assertEqual(filtered_llm.json()["count"], 1)
            self.assertEqual(filtered_asr.status_code, 200)
            self.assertEqual(filtered_asr.json()["count"], 1)


if __name__ == "__main__":
    unittest.main()
