import asyncio
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from infrastructure.runtime_log_server import RuntimeLogBuffer, create_runtime_log_app
from real_world_testing.case_loader import load_suite
from real_world_testing.lab import RealWorldLab
from real_world_testing.store import SQLiteRealWorldTestStore
from real_world_testing.runtime_lock import RuntimeOwnershipLock


class FakeExecutor:
    workspaces = []

    def __init__(self, *, workspace, emit, should_cancel, **kwargs):
        self.workspace = Path(workspace)
        self.emit = emit
        self.should_cancel = should_cancel
        self.workspaces.append(self.workspace)

    async def execute(self, scenario, *, playback_speed):
        self.emit("input", "fixture_released", {"scenario_id": scenario.scenario_id})
        self.emit("model", "model_request", {"messages": [{"role": "user", "content": "fixture"}]})
        self.emit("agent", "tool_finished", {"tool_name": "fake", "arguments": {}, "result": "ok"})
        await asyncio.sleep(0)
        return {"transcript_text": "hello" if scenario.modality == "audio_sequence" else None,
                "final_response": "done", "summary": {"playback_speed": playback_speed}}


class RealWorldLabTests(unittest.TestCase):
    def _fixture(self, root: Path):
        media = root / "shot.png"
        media.write_bytes(b"png")
        suite_dir = root / "suites"
        suite_dir.mkdir()
        (suite_dir / "suite.json").write_text(json.dumps({
            "schema_version": 1,
            "suite_id": "suite_one",
            "title": "Suite one",
            "scenarios": [{
                "scenario_id": "screen_flow",
                "title": "Screen flow",
                "modality": "image_sequence",
                "events": [
                    {"offset_seconds": 0, "media_path": "../shot.png"},
                    {"offset_seconds": 4, "media_path": "../shot.png"},
                ],
            }],
        }), encoding="utf-8")
        config = root / "config.ini"
        config.write_text("""
[runtime]
default_model = model-a
[models]
passive_observer_model = model-a
full_passive_observer_model = model-a
passive_followup_model = model-a
followup_execution_model = model-a
transcript_processing_model = model-a
reporter_model = model-a
browser_agent_model = model-a
[autonomy]
mode = active
""", encoding="utf-8")
        (root / "models_preset.ini").write_text("version = 1.0\n[preset-a]\nmodel = fake.gguf\n", encoding="utf-8")
        return suite_dir, config

    def test_manifest_resolves_media_and_rejects_non_monotonic_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suites, _ = self._fixture(root)
            loaded = load_suite(suites / "suite.json")
            self.assertEqual(loaded.scenarios[0].events[0].media_path, str((root / "shot.png").resolve()))
            payload = json.loads((suites / "suite.json").read_text(encoding="utf-8"))
            payload["scenarios"][0]["events"][1]["offset_seconds"] = -1
            (suites / "bad.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "monotonic"):
                load_suite(suites / "bad.json")

    def test_store_persists_trace_and_stage_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteRealWorldTestStore(Path(tmp) / "lab.db")
            run_id = store.create_run(suite_id="suite", scenario_ids=["one"], playback_speed=2,
                                      model_roles={"vision": "model"}, config={})
            scenario = type("Scenario", (), {
                "scenario_id": "one", "title": "One", "modality": "image_sequence",
                "rubric_notes": "", "events": [],
            })()
            result_id = store.create_result(run_id, scenario)
            event = store.append_event(run_id=run_id, result_id=result_id, stage="agent",
                                       event_type="tool_finished", payload={"result": "ok"})
            review = store.upsert_review(result_id, {"reviewer": "tester", "overall_score": 5,
                                                     "perception_score": 4, "notes": "good"})
            self.assertEqual(store.list_events(run_id)[0]["sequence"], event.sequence)
            self.assertEqual(review["overall_score"], 5)
            with self.assertRaisesRegex(ValueError, "between 1 and 5"):
                store.upsert_review(result_id, {"overall_score": 6})

    def test_model_presets_allow_llama_global_header_and_runtime_lock_is_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suites, config = self._fixture(root)
            lab = RealWorldLab(project_root=root, config_path=config, data_root=root / "data",
                               suites_root=suites, executor_factory=FakeExecutor)
            self.assertIn("preset-a", lab.models()["presets"])
            first = RuntimeOwnershipLock(root / "runtime.lock", "first")
            second = RuntimeOwnershipLock(root / "runtime.lock", "second")
            first.acquire()
            try:
                with self.assertRaisesRegex(RuntimeError, "already owns"):
                    second.acquire()
            finally:
                first.release()

    def test_mock_run_is_isolated_traced_and_available_through_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suites, config = self._fixture(root)
            lab = RealWorldLab(project_root=root, config_path=config, data_root=root / "data",
                               suites_root=suites, executor_factory=FakeExecutor)
            client = TestClient(create_runtime_log_app(RuntimeLogBuffer(), real_world_lab=lab))
            denied = client.post("/api/real-world/runs", json={
                "suite_id": "suite_one", "live_tools_confirmation": "no",
            })
            self.assertEqual(denied.status_code, 400)
            started = client.post("/api/real-world/runs", json={
                "suite_id": "suite_one", "playback_speed": 2,
                "live_tools_confirmation": "RUN LIVE TOOLS",
            })
            self.assertEqual(started.status_code, 200)
            self.assertTrue(lab.wait_for_idle(2))
            run_id = started.json()["run"]["run_id"]
            for _ in range(100):
                run = client.get(f"/api/real-world/runs/{run_id}").json()["run"]
                if run["status"] not in {"queued", "running"}:
                    break
                time.sleep(0.01)
            self.assertEqual(run["status"], "completed")
            self.assertEqual(run["results"][0]["final_response"], "done")
            events = client.get(f"/api/real-world/runs/{run_id}/trace").json()["events"]
            self.assertEqual([event["event_type"] for event in events],
                             ["accelerator_detected", "fixture_released", "model_request", "tool_finished"])
            exported = client.get(f"/api/real-world/runs/{run_id}/export.json")
            self.assertEqual(exported.status_code, 200)
            self.assertEqual(exported.json()["run"]["run_id"], run_id)
            exported_csv = client.get(f"/api/real-world/runs/{run_id}/export.csv")
            self.assertEqual(exported_csv.status_code, 200)
            self.assertIn("accelerator_detected", exported_csv.text)
            self.assertTrue(FakeExecutor.workspaces[-1].is_dir())

    def test_raw_upload_is_scoped_and_can_form_inline_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suites, config = self._fixture(root)
            lab = RealWorldLab(project_root=root, config_path=config, data_root=root / "data",
                               suites_root=suites, executor_factory=FakeExecutor)
            client = TestClient(create_runtime_log_app(RuntimeLogBuffer(), real_world_lab=lab))
            uploaded = client.post("/api/real-world/uploads?filename=screen.png&kind=image", content=b"image")
            self.assertEqual(uploaded.status_code, 200)
            media_id = uploaded.json()["media"]["media_id"]
            self.assertEqual(client.get(f"/api/real-world/media/{media_id}").content, b"image")
            started = client.post("/api/real-world/runs", json={
                "playback_speed": 10, "live_tools_confirmation": "RUN LIVE TOOLS",
                "inline_scenario": {"modality": "image_sequence", "events": [
                    {"media_id": media_id, "offset_seconds": 0}
                ]},
            })
            self.assertEqual(started.status_code, 200)
            self.assertTrue(lab.wait_for_idle(2))


if __name__ == "__main__":
    unittest.main()
