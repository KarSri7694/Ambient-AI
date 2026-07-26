from __future__ import annotations

import json
import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from infrastructure.accelerator import detect_accelerator
from rocm_tuning.candidate import TuningCandidate, generate_candidates, render_winning_preset, write_candidate_preset
from rocm_tuning.measure import chat_once, load_model, summarize_measurements, unload_model, vram_snapshot
from rocm_tuning.runner import LlamaServerRunner
from rocm_tuning.store import SQLiteRocmTuningStore


@dataclass(frozen=True)
class RocmTuningConfig:
    data_root: Path
    llama_server_path: str
    tuning_port: int = 8091
    api_key: str = "testkey"
    models: list[dict[str, object]] | None = None
    contexts: list[int] | None = None
    gpu_layers: list[int] | None = None
    flash_attention: list[bool] | None = None
    kv_cache: list[tuple[str, str]] | None = None
    warmups: int = 1
    measured_runs: int = 10
    prompt: str = "Summarize how ambient AI should behave safely and usefully in five concise bullets."
    min_free_vram_mb: int = 512


class RocmTuningService:
    def __init__(self, *, config: RocmTuningConfig, store: SQLiteRocmTuningStore | None = None):
        self.config = config
        self.data_root = Path(config.data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.store = store or SQLiteRocmTuningStore(self.data_root / "tuning.db")
        self.candidates_dir = self.data_root / "candidates"
        self.exports_dir = self.data_root / "exports"
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)

    def status(self) -> dict[str, Any]:
        accelerator = detect_accelerator().to_dict()
        return {
            "available": True,
            "accelerator": accelerator,
            "latest_profile": self.store.latest_profile(gpu_architecture=accelerator.get("gcn_architecture")),
            "recent_runs": self.store.list_runs(limit=5),
        }

    def run(self) -> dict[str, Any]:
        accelerator = detect_accelerator("amd_rocm").to_dict()
        candidates = self._candidates()
        run_id = self.store.create_run(
            accelerator=accelerator,
            config={**self._config_dict(), "candidate_count": len(candidates)},
        )
        passed: list[tuple[TuningCandidate, dict[str, Any]]] = []
        try:
            for candidate in candidates:
                try:
                    summary = self._run_candidate(candidate)
                    status = "completed"
                    error_text = None
                    if self._free_vram_too_low(summary):
                        status = "failed"
                        error_text = f"free VRAM after run below {self.config.min_free_vram_mb} MB"
                    measurements = summary.pop("measurements")
                    self.store.insert_result(
                        run_id=run_id,
                        candidate=candidate,
                        status=status,
                        summary=summary,
                        measurements=measurements,
                        load_ms=summary.get("load_ms"),
                        error_text=error_text,
                    )
                    if status == "completed":
                        passed.append((candidate, summary))
                except Exception as exc:
                    self.store.insert_result(
                        run_id=run_id,
                        candidate=candidate,
                        status="failed",
                        summary={},
                        measurements=[],
                        load_ms=None,
                        error_text=str(exc),
                    )
            winner = self._select_winner(passed)
            if winner is None:
                self.store.finish_run(run_id, status="failed", error_text="no successful ROCm tuning candidate")
            else:
                candidate, summary = winner
                preset_path = self.data_root / "tuned_models_preset.ini"
                preset_path.write_text(render_winning_preset(candidate), encoding="utf-8")
                self.store.upsert_profile(
                    run_id=run_id,
                    candidate=candidate,
                    summary=summary,
                    preset_path=preset_path,
                    accelerator=accelerator,
                )
                self.store.finish_run(run_id, status="completed")
            self._write_exports()
            return self.store.get_run(run_id) or {"run_id": run_id}
        except Exception as exc:
            self.store.finish_run(run_id, status="failed", error_text=str(exc))
            raise

    def _run_candidate(self, candidate: TuningCandidate) -> dict[str, Any]:
        preset_path = write_candidate_preset(candidate, self.candidates_dir / f"{candidate.candidate_id}.ini")
        runner = LlamaServerRunner(
            llama_server_path=self.config.llama_server_path,
            port=self.config.tuning_port,
            api_key=self.config.api_key,
        )
        server = runner.start(models_preset=preset_path, model=candidate.preset_name)
        try:
            before_load = vram_snapshot()
            load_ms = load_model(server.base_url, candidate.preset_name)
            rows = [
                chat_once(
                    server.base_url,
                    self.config.api_key,
                    candidate.preset_name,
                    self.config.prompt,
                    index=index,
                    warmup=index < self.config.warmups,
                )
                for index in range(self.config.warmups + self.config.measured_runs)
            ]
            summary = summarize_measurements(rows)
            summary["load_ms"] = load_ms
            summary["free_vram_before_load_mb"] = before_load.free_mb
            summary["measurements"] = [row.to_dict() for row in rows]
            return summary
        finally:
            unload_model(server.base_url, candidate.preset_name)
            server.stop()

    def _candidates(self) -> list[TuningCandidate]:
        return generate_candidates(
            models=self.config.models or [],
            contexts=self.config.contexts or [8192, 16384],
            gpu_layers=self.config.gpu_layers or [99, 80, 60],
            flash_attention=self.config.flash_attention or [True, False],
            kv_cache=self.config.kv_cache or [("q8_0", "q8_0"), ("q4_0", "q4_0")],
        )

    def _select_winner(self, passed: list[tuple[TuningCandidate, dict[str, Any]]]) -> tuple[TuningCandidate, dict[str, Any]] | None:
        if not passed:
            return None
        return sorted(
            passed,
            key=lambda item: (
                item[1].get("ttft_seconds_median") if item[1].get("ttft_seconds_median") is not None else 999999,
                -(item[1].get("chars_per_second_median") or 0),
            ),
        )[0]

    def _free_vram_too_low(self, summary: dict[str, Any]) -> bool:
        value = summary.get("free_vram_after_mb_min")
        return value is not None and int(value) < self.config.min_free_vram_mb

    def _write_exports(self) -> None:
        payload = self.store.export_json()
        (self.exports_dir / "rocm_tuning.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (self.exports_dir / "rocm_tuning.csv").write_text(self.store.export_csv(), encoding="utf-8")

    def _config_dict(self) -> dict[str, Any]:
        return {
            "llama_server_path": self.config.llama_server_path,
            "tuning_port": self.config.tuning_port,
            "models": self.config.models or [],
            "contexts": self.config.contexts or [],
            "gpu_layers": self.config.gpu_layers or [],
            "flash_attention": self.config.flash_attention or [],
            "kv_cache": self.config.kv_cache or [],
            "warmups": self.config.warmups,
            "measured_runs": self.config.measured_runs,
            "min_free_vram_mb": self.config.min_free_vram_mb,
        }


def config_from_json_file(path: str | Path, *, data_root: str | Path | None = None) -> RocmTuningConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    kv_cache = [tuple(item) for item in payload.get("kv_cache", [])]
    return RocmTuningConfig(
        data_root=Path(data_root or payload.get("data_root") or ".ambient_data/rocm"),
        llama_server_path=str(payload.get("llama_server_path") or ""),
        tuning_port=int(payload.get("tuning_port", 8091)),
        api_key=str(payload.get("api_key", "testkey")),
        models=list(payload.get("models") or []),
        contexts=[int(item) for item in payload.get("contexts", [8192, 16384])],
        gpu_layers=[int(item) for item in payload.get("gpu_layers", [99, 80, 60])],
        flash_attention=[bool(item) for item in payload.get("flash_attention", [True, False])],
        kv_cache=kv_cache or [("q8_0", "q8_0"), ("q4_0", "q4_0")],
        warmups=int(payload.get("warmups", 1)),
        measured_runs=int(payload.get("measured_runs", 10)),
        prompt=str(payload.get("prompt") or RocmTuningConfig(data_root=Path("."), llama_server_path="").prompt),
        min_free_vram_mb=int(payload.get("min_free_vram_mb", 512)),
    )


def config_from_ini(path: str | Path, *, project_root: str | Path | None = None) -> RocmTuningConfig:
    root = Path(project_root or Path(path).resolve().parent)
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")

    def _json(option: str, fallback: Any) -> Any:
        raw = parser.get("rocm_tuning", option, fallback=json.dumps(fallback))
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return fallback

    data_root = Path(parser.get("rocm_tuning", "data_root", fallback=".ambient_data/rocm"))
    if not data_root.is_absolute():
        data_root = root / data_root
    kv_cache = [tuple(item) for item in _json("kv_cache_json", [["q8_0", "q8_0"], ["q4_0", "q4_0"]])]
    return RocmTuningConfig(
        data_root=data_root,
        llama_server_path=parser.get("rocm_tuning", "llama_server_path", fallback="").strip(),
        tuning_port=parser.getint("rocm_tuning", "tuning_port", fallback=8091),
        api_key=parser.get("rocm_tuning", "api_key", fallback="testkey"),
        models=list(_json("models_json", [])),
        contexts=[int(item) for item in _json("contexts_json", [8192, 16384, 32768])],
        gpu_layers=[int(item) for item in _json("gpu_layers_json", [99, 80, 60])],
        flash_attention=[bool(item) for item in _json("flash_attention_json", [True, False])],
        kv_cache=kv_cache,
        warmups=parser.getint("rocm_tuning", "warmups", fallback=1),
        measured_runs=parser.getint("rocm_tuning", "measured_runs", fallback=10),
        prompt=parser.get("rocm_tuning", "prompt", fallback=RocmTuningConfig(data_root=Path("."), llama_server_path="").prompt),
        min_free_vram_mb=parser.getint("rocm_tuning", "min_free_vram_mb", fallback=512),
    )
