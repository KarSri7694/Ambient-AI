from __future__ import annotations

import csv
import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class RocmTuningRun:
    run_id: str
    created_at: str
    completed_at: str | None
    status: str
    gpu_backend: str | None
    gpu_name: str | None
    gpu_architecture: str | None
    runtime_version: str | None
    config_json: str
    error_text: str | None = None


class SQLiteRocmTuningStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS rocm_tuning_runs (
                    run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, completed_at TEXT,
                    status TEXT NOT NULL, gpu_backend TEXT, gpu_name TEXT, gpu_architecture TEXT,
                    runtime_version TEXT, config_json TEXT NOT NULL, error_text TEXT
                );
                CREATE TABLE IF NOT EXISTS rocm_tuning_results (
                    result_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, created_at TEXT NOT NULL,
                    status TEXT NOT NULL, model_name TEXT NOT NULL, preset_name TEXT NOT NULL,
                    candidate_json TEXT NOT NULL, summary_json TEXT, measurements_json TEXT,
                    load_ms REAL, median_ttft_seconds REAL, median_chars_per_second REAL,
                    vram_delta_mb REAL, free_vram_after_mb INTEGER, error_text TEXT,
                    FOREIGN KEY(run_id) REFERENCES rocm_tuning_runs(run_id)
                );
                CREATE TABLE IF NOT EXISTS rocm_tuned_profiles (
                    profile_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, run_id TEXT NOT NULL,
                    model_name TEXT NOT NULL, gpu_architecture TEXT, gpu_name TEXT,
                    preset_path TEXT NOT NULL, candidate_json TEXT NOT NULL, summary_json TEXT NOT NULL,
                    UNIQUE(model_name, gpu_architecture)
                );
                CREATE INDEX IF NOT EXISTS idx_rocm_tuning_runs_created ON rocm_tuning_runs(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_rocm_tuning_results_run ON rocm_tuning_results(run_id);
                """
            )

    def create_run(self, *, accelerator: dict[str, Any], config: dict[str, Any]) -> str:
        run_id = uuid.uuid4().hex
        with self._connection() as conn:
            conn.execute(
                "INSERT INTO rocm_tuning_runs VALUES (?, ?, NULL, 'running', ?, ?, ?, ?, ?, NULL)",
                (
                    run_id,
                    datetime.now().isoformat(),
                    accelerator.get("backend"),
                    accelerator.get("gpu_name"),
                    accelerator.get("gcn_architecture"),
                    accelerator.get("runtime_version"),
                    json.dumps(config, ensure_ascii=False),
                ),
            )
        return run_id

    def finish_run(self, run_id: str, *, status: str, error_text: str | None = None) -> None:
        with self._connection() as conn:
            conn.execute(
                "UPDATE rocm_tuning_runs SET status=?, completed_at=?, error_text=COALESCE(?, error_text) WHERE run_id=?",
                (status, datetime.now().isoformat(), error_text, run_id),
            )

    def insert_result(
        self,
        *,
        run_id: str,
        candidate: Any,
        status: str,
        summary: dict[str, Any] | None,
        measurements: list[dict[str, Any]] | None,
        load_ms: float | None,
        error_text: str | None = None,
    ) -> str:
        result_id = uuid.uuid4().hex
        candidate_payload = candidate.to_dict() if hasattr(candidate, "to_dict") else dict(candidate)
        summary = summary or {}
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO rocm_tuning_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result_id,
                    run_id,
                    datetime.now().isoformat(),
                    status,
                    candidate_payload["model_name"],
                    candidate_payload.get("preset_name") or f"{candidate_payload['model_name']}_{candidate_payload['candidate_id']}",
                    json.dumps(candidate_payload, ensure_ascii=False),
                    json.dumps(summary, ensure_ascii=False),
                    json.dumps(measurements or [], ensure_ascii=False),
                    load_ms,
                    summary.get("ttft_seconds_median"),
                    summary.get("chars_per_second_median"),
                    summary.get("vram_delta_mb_max"),
                    summary.get("free_vram_after_mb_min"),
                    error_text,
                ),
            )
        return result_id

    def upsert_profile(
        self,
        *,
        run_id: str,
        candidate: Any,
        summary: dict[str, Any],
        preset_path: str | Path,
        accelerator: dict[str, Any],
    ) -> None:
        candidate_payload = candidate.to_dict() if hasattr(candidate, "to_dict") else dict(candidate)
        profile_id = uuid.uuid4().hex
        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO rocm_tuned_profiles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    profile_id,
                    datetime.now().isoformat(),
                    run_id,
                    candidate_payload["model_name"],
                    accelerator.get("gcn_architecture"),
                    accelerator.get("gpu_name"),
                    str(preset_path),
                    json.dumps(candidate_payload, ensure_ascii=False),
                    json.dumps(summary, ensure_ascii=False),
                ),
            )

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute("SELECT * FROM rocm_tuning_runs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._row_dict(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connection() as conn:
            run = conn.execute("SELECT * FROM rocm_tuning_runs WHERE run_id=?", (run_id,)).fetchone()
            if run is None:
                return None
            results = conn.execute("SELECT * FROM rocm_tuning_results WHERE run_id=? ORDER BY created_at", (run_id,)).fetchall()
        payload = self._row_dict(run)
        payload["results"] = [self._result_dict(row) for row in results]
        return payload

    def latest_profile(self, *, model_name: str | None = None, gpu_architecture: str | None = None) -> dict[str, Any] | None:
        query = "SELECT * FROM rocm_tuned_profiles WHERE 1=1"
        params: list[Any] = []
        if model_name:
            query += " AND model_name=?"
            params.append(model_name)
        if gpu_architecture:
            query += " AND gpu_architecture=?"
            params.append(gpu_architecture)
        query += " ORDER BY created_at DESC LIMIT 1"
        with self._connection() as conn:
            row = conn.execute(query, tuple(params)).fetchone()
        return self._profile_dict(row) if row else None

    def export_json(self) -> dict[str, Any]:
        runs = self.list_runs(limit=200)
        profiles = []
        with self._connection() as conn:
            rows = conn.execute("SELECT * FROM rocm_tuned_profiles ORDER BY created_at DESC").fetchall()
        profiles = [self._profile_dict(row) for row in rows]
        return {"runs": runs, "profiles": profiles}

    def export_csv(self) -> str:
        with self._connection() as conn:
            rows = conn.execute("SELECT * FROM rocm_tuning_results ORDER BY created_at DESC").fetchall()
        import io

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=["run_id", "status", "model_name", "preset_name", "load_ms", "median_ttft_seconds", "median_chars_per_second", "vram_delta_mb", "free_vram_after_mb", "error_text"])
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in writer.fieldnames})
        return output.getvalue()

    def _row_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["config"] = json.loads(item.pop("config_json") or "{}")
        return item

    def _result_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["candidate"] = json.loads(item.pop("candidate_json") or "{}")
        item["summary"] = json.loads(item.pop("summary_json") or "{}")
        item["measurements"] = json.loads(item.pop("measurements_json") or "[]")
        return item

    def _profile_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["candidate"] = json.loads(item.pop("candidate_json") or "{}")
        item["summary"] = json.loads(item.pop("summary_json") or "{}")
        return item
