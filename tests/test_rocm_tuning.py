import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from application.services.resource_governor_service import ResourceGovernorService
from core.models import InferenceRequest, ResourceSnapshot
from infrastructure.runtime_log_server import RuntimeLogBuffer, create_runtime_log_app
from rocm_tuning.candidate import generate_candidates, render_winning_preset
from rocm_tuning.service import RocmTuningConfig, RocmTuningService
from rocm_tuning.store import SQLiteRocmTuningStore


def test_candidate_generation_and_winning_preset_render():
    candidates = generate_candidates(
        models=[{"name": "model-a", "model": "a.gguf", "mmproj": "a-mmproj.gguf"}],
        contexts=[8192],
        gpu_layers=[99],
        flash_attention=[True, False],
        kv_cache=[("q8_0", "q8_0")],
    )

    assert len(candidates) == 2
    text = render_winning_preset(candidates[0])
    assert "[model-a]" in text
    assert "ngl=99" in text
    assert "fa=on" in text
    assert "ctk=q8_0" in text
    assert "mmproj=a-mmproj.gguf" in text


def test_rocm_tuning_store_persists_latest_profile_and_exports():
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteRocmTuningStore(Path(tmp) / "tuning.db")
        candidate = generate_candidates(
            models=[{"name": "model-a", "model": "a.gguf"}],
            contexts=[8192],
            gpu_layers=[99],
            flash_attention=[True],
            kv_cache=[("q8_0", "q8_0")],
        )[0]
        run_id = store.create_run(
            accelerator={"backend": "amd_rocm", "gpu_name": "Radeon", "gcn_architecture": "gfx1100", "runtime_version": "7.2.1"},
            config={"candidate_count": 1},
        )
        store.insert_result(
            run_id=run_id,
            candidate=candidate,
            status="completed",
            summary={"ttft_seconds_median": 0.4, "chars_per_second_median": 120, "vram_delta_mb_max": 2048},
            measurements=[],
            load_ms=100,
        )
        store.upsert_profile(
            run_id=run_id,
            candidate=candidate,
            summary={"ttft_seconds_median": 0.4, "chars_per_second_median": 120, "vram_delta_mb_max": 2048},
            preset_path=Path(tmp) / "preset.ini",
            accelerator={"gpu_name": "Radeon", "gcn_architecture": "gfx1100"},
        )
        store.finish_run(run_id, status="completed")

        latest = store.latest_profile(model_name="model-a", gpu_architecture="gfx1100")
        assert latest["model_name"] == "model-a"
        assert latest["summary"]["vram_delta_mb_max"] == 2048
        assert "model-a" in store.export_csv()
        assert store.export_json()["profiles"][0]["model_name"] == "model-a"


def test_rocm_tuning_service_selects_fastest_stable_candidate():
    config = RocmTuningConfig(data_root=Path("."), llama_server_path="llama.exe")
    service = RocmTuningService(config=config, store=SQLiteRocmTuningStore(":memory:"))
    first, second = generate_candidates(
        models=[{"name": "model-a", "model": "a.gguf"}],
        contexts=[8192, 16384],
        gpu_layers=[99],
        flash_attention=[True],
        kv_cache=[("q8_0", "q8_0")],
    )

    winner = service._select_winner([
        (first, {"ttft_seconds_median": 0.8, "chars_per_second_median": 200}),
        (second, {"ttft_seconds_median": 0.4, "chars_per_second_median": 100}),
    ])

    assert winner[0] == second


class _Monitor:
    def __init__(self, free_vram_mb):
        self.free_vram_mb = free_vram_mb

    def snapshot(self, *, user_idle=False, force=False):
        return ResourceSnapshot(
            captured_at="now",
            total_ram_mb=16000,
            available_ram_mb=8000,
            available_ram_percent=50,
            total_vram_mb=12000,
            free_vram_mb=self.free_vram_mb,
            gpu_telemetry_available=True,
            gpu_backend="amd_rocm",
        )


def test_resource_governor_uses_rocm_tuned_vram_delta():
    governor = ResourceGovernorService(monitor=_Monitor(free_vram_mb=2200), critical_vram_mb=512)
    governor.set_rocm_profile_provider(lambda model: {
        "summary": {"vram_delta_mb_max": 2048},
        "candidate": {"model_name": model},
    })
    decision = governor.evaluate(InferenceRequest(
        workload="vision",
        model_name="model-a",
        background=False,
        user_active=True,
    ))

    assert decision.allowed is False
    assert "tuned ROCm profile" in decision.reason


def test_runtime_api_exposes_rocm_tuning_status_and_exports():
    with tempfile.TemporaryDirectory() as tmp:
        service = RocmTuningService(
            config=RocmTuningConfig(data_root=Path(tmp), llama_server_path="llama.exe"),
            store=SQLiteRocmTuningStore(Path(tmp) / "tuning.db"),
        )
        client = TestClient(create_runtime_log_app(RuntimeLogBuffer(), rocm_tuning_service=service))

        assert client.get("/api/rocm-tuning/status").status_code == 200
        assert client.get("/api/rocm-tuning/runs").json()["available"] is True
        assert client.get("/api/rocm-tuning/latest").json()["profile"] is None
        assert client.get("/api/rocm-tuning/export.csv").status_code == 200
