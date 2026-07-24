import logging
import json
import re
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
from infrastructure.plain_capture_store import PlainCaptureStore


class _AuditStore:
    def __init__(self):
        self.events = []

    def audit(self, actor, action, entity_id, metadata):
        self.events.append((actor, action, entity_id, metadata))


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

        for route in ["/", "/chat", "/interactions", "/reports", "/logs", "/benchmarks", "/training"]:
            response = client.get(route)
            self.assertEqual(response.status_code, 200)
            self.assertIn("Ambient Agent", response.text)
            self.assertIn("/runtime-ui/assets/", response.text)

        asset_paths = re.findall(r'(?:src|href)="(/runtime-ui/assets/[^"]+)"', response.text)
        self.assertGreaterEqual(len(asset_paths), 2)
        for asset_path in asset_paths:
            self.assertEqual(client.get(asset_path).status_code, 200)

    def test_interaction_store_filters_sorts_and_counts_deterministically(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteInteractionLogAdapter(str(Path(tmpdir) / "interaction_logs.db"))
            for interaction_id, created_at in [
                ("b", "2026-07-23T09:00:00"),
                ("a", "2026-07-23T09:00:00"),
                ("c", "2026-07-24T10:00:00"),
            ]:
                store.insert(
                    InteractionLogEntry(
                        interaction_id=interaction_id,
                        interaction_run_id=f"run-{interaction_id}",
                        created_at=created_at,
                        completed_at=created_at,
                        source="test",
                        model="model-a",
                        messages_json=json.dumps([{"role": "user", "content": interaction_id}]),
                        response_text=f"response-{interaction_id}",
                    )
                )

            newest = store.list_entries(limit=10, date_from="2026-07-23", date_to="2026-07-23")
            oldest = store.list_entries(limit=10, sort_order="asc")

            self.assertEqual([row.interaction_id for row in newest], ["b", "a"])
            self.assertEqual([row.interaction_id for row in oldest], ["a", "b", "c"])
            self.assertEqual(store.count_entries(date_from="2026-07-23", date_to="2026-07-23"), 2)

    def test_interaction_api_normalizes_pairs_and_paginates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = SQLiteInteractionLogAdapter(str(root / "interaction_logs.db"))
            store.insert(
                InteractionLogEntry(
                    interaction_id="interaction-1",
                    interaction_run_id="run-1",
                    created_at="2026-07-24T10:00:00",
                    completed_at="2026-07-24T10:00:01",
                    source="direct_chat",
                    model="model-a",
                    messages_json=json.dumps(
                        [
                            {"role": "system", "content": "system context"},
                            {"role": "user", "content": "hello"},
                        ]
                    ),
                    response_text="world",
                    duration_ms=1000,
                )
            )
            store.insert(
                InteractionLogEntry(
                    interaction_id="interaction-2",
                    interaction_run_id="run-2",
                    created_at="2026-07-25T10:00:00",
                    completed_at="2026-07-25T10:00:01",
                    source="test",
                    model="model-b",
                    messages_json="not-json",
                    error_text="failed",
                )
            )
            client = TestClient(create_runtime_log_app(RuntimeLogBuffer(), report_store=store))

            response = client.get(
                "/api/interactions",
                params={"date_from": "2026-07-24", "date_to": "2026-07-24", "sort": "oldest"},
            )
            payload = response.json()

            self.assertEqual(response.status_code, 200)
            self.assertEqual(payload["pagination"]["total"], 1)
            self.assertEqual(payload["items"][0]["input"]["request"]["content"], "hello")
            self.assertEqual(payload["items"][0]["input"]["context_messages"][0]["role"], "system")
            self.assertEqual(payload["items"][0]["response_text"], "world")
            self.assertEqual(client.get("/api/interactions", params={"date_from": "2026-07-26", "date_to": "2026-07-24"}).status_code, 422)

    def test_protected_interaction_input_and_capture_image_are_audited(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            captures = PlainCaptureStore(str(root / "captures"))
            protected_ref = captures.store_bytes(
                json.dumps([{"role": "user", "content": "private prompt"}]).encode("utf-8"),
                original_name="messages.json",
                kind="llm_messages",
                mime_type="application/json",
            )
            image_ref = captures.store_bytes(
                b"fake-png",
                original_name="screen.png",
                kind="screenshot",
                mime_type="image/png",
            )
            store = SQLiteInteractionLogAdapter(str(root / "interaction_logs.db"))
            store.insert(
                InteractionLogEntry(
                    interaction_id="protected-1",
                    interaction_run_id="run-1",
                    created_at="2026-07-24T10:00:00",
                    completed_at="2026-07-24T10:00:01",
                    source="passive_observer",
                    model="model-a",
                    messages_json=json.dumps({"protected_payload_ref": protected_ref}),
                    image_path=image_ref,
                    response_text="description",
                )
            )
            audit = _AuditStore()
            client = TestClient(
                create_runtime_log_app(
                    RuntimeLogBuffer(),
                    report_store=store,
                    capture_store=captures,
                    autonomy_store=audit,
                )
            )

            listed = client.get("/api/interactions").json()["items"][0]
            revealed = client.get("/api/interactions/protected-1/input")
            image = client.get("/api/interactions/protected-1/image")

            self.assertTrue(listed["input"]["protected"])
            self.assertIsNone(listed["input"]["request"])
            self.assertEqual(revealed.json()["input"]["request"]["content"], "private prompt")
            self.assertEqual(image.status_code, 200)
            self.assertEqual(image.headers["content-type"], "image/png")
            self.assertEqual(image.content, b"fake-png")
            self.assertEqual([event[1] for event in audit.events], ["interaction.input_viewed", "interaction.image_viewed"])

    def test_interaction_image_only_serves_database_paths_inside_media_roots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            media_root = root / "media"
            media_root.mkdir()
            allowed = media_root / "allowed.png"
            allowed.write_bytes(b"allowed")
            outside = root / "outside.png"
            outside.write_bytes(b"outside")
            store = SQLiteInteractionLogAdapter(str(root / "interaction_logs.db"))
            for interaction_id, path in [("allowed", allowed), ("outside", outside)]:
                store.insert(
                    InteractionLogEntry(
                        interaction_id=interaction_id,
                        interaction_run_id=f"run-{interaction_id}",
                        created_at="2026-07-24T10:00:00",
                        completed_at="2026-07-24T10:00:01",
                        source="test",
                        model="model-a",
                        messages_json=json.dumps([{"role": "user", "content": "describe"}]),
                        image_path=str(path),
                        response_text="done",
                    )
                )
            client = TestClient(
                create_runtime_log_app(
                    RuntimeLogBuffer(),
                    report_store=store,
                    media_roots=[str(media_root)],
                )
            )

            self.assertEqual(client.get("/api/interactions/allowed/image").content, b"allowed")
            self.assertEqual(client.get("/api/interactions/outside/image").status_code, 404)

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
