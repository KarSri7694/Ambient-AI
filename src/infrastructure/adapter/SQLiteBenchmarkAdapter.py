import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

from core.models import BenchmarkManualReview, BenchmarkResult, BenchmarkRun


class SQLiteBenchmarkAdapter:
    """SQLite-backed store for benchmark runs, results, and manual reviews."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _managed_connection(self):
        conn = self._connection()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._managed_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS benchmark_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    service_name TEXT NOT NULL,
                    model_names_json TEXT NOT NULL,
                    case_ids_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    notes TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS benchmark_results (
                    result_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    service_name TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    case_title TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    screenshot_path TEXT,
                    transcript_path TEXT,
                    response_text TEXT,
                    structured_output_json TEXT,
                    error_text TEXT,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    prefill_seconds REAL,
                    generation_seconds REAL,
                    prefill_tokens_per_second REAL,
                    generation_tokens_per_second REAL,
                    token_count_method TEXT,
                    auto_score REAL,
                    auto_score_details_json TEXT,
                    metadata_json TEXT,
                    status TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES benchmark_runs (run_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS benchmark_manual_reviews (
                    review_id TEXT PRIMARY KEY,
                    result_id TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    reviewer TEXT NOT NULL,
                    score REAL,
                    notes TEXT,
                    FOREIGN KEY (result_id) REFERENCES benchmark_results (result_id)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_benchmark_runs_created_at
                ON benchmark_runs (created_at DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_benchmark_results_run_id
                ON benchmark_results (run_id, created_at DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_benchmark_results_service_model
                ON benchmark_results (service_name, model_name, created_at DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_benchmark_results_case_id
                ON benchmark_results (case_id, created_at DESC)
                """
            )

    def insert_run(self, run: BenchmarkRun) -> None:
        with self._managed_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO benchmark_runs (
                    run_id, created_at, completed_at, service_name,
                    model_names_json, case_ids_json, status, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.created_at,
                    run.completed_at,
                    run.service_name,
                    run.model_names_json,
                    run.case_ids_json,
                    run.status,
                    run.notes,
                ),
            )

    def update_run_status(self, run_id: str, *, status: str, completed_at: Optional[str]) -> None:
        with self._managed_connection() as conn:
            conn.execute(
                """
                UPDATE benchmark_runs
                SET status = ?, completed_at = ?
                WHERE run_id = ?
                """,
                (status, completed_at, run_id),
            )

    def insert_result(self, result: BenchmarkResult) -> None:
        with self._managed_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO benchmark_results (
                    result_id, run_id, created_at, completed_at, service_name, case_id, case_title,
                    model_name, screenshot_path, transcript_path, response_text, structured_output_json,
                    error_text, prompt_tokens, completion_tokens, total_tokens, prefill_seconds,
                    generation_seconds, prefill_tokens_per_second, generation_tokens_per_second,
                    token_count_method, auto_score, auto_score_details_json, metadata_json, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.result_id,
                    result.run_id,
                    result.created_at,
                    result.completed_at,
                    result.service_name,
                    result.case_id,
                    result.case_title,
                    result.model_name,
                    result.screenshot_path,
                    result.transcript_path,
                    result.response_text,
                    result.structured_output_json,
                    result.error_text,
                    result.prompt_tokens,
                    result.completion_tokens,
                    result.total_tokens,
                    result.prefill_seconds,
                    result.generation_seconds,
                    result.prefill_tokens_per_second,
                    result.generation_tokens_per_second,
                    result.token_count_method,
                    result.auto_score,
                    result.auto_score_details_json,
                    result.metadata_json,
                    result.status,
                ),
            )

    def list_runs(self, limit: int = 50, service_name: Optional[str] = None) -> List[BenchmarkRun]:
        query = "SELECT * FROM benchmark_runs"
        params: List[object] = []
        if service_name:
            query += " WHERE service_name = ?"
            params.append(service_name)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._managed_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._run_from_row(row) for row in rows]

    def list_results(
        self,
        *,
        limit: int = 200,
        run_id: Optional[str] = None,
        service_name: Optional[str] = None,
        model_name: Optional[str] = None,
        case_id: Optional[str] = None,
    ) -> List[BenchmarkResult]:
        query = "SELECT * FROM benchmark_results WHERE 1 = 1"
        params: List[object] = []
        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)
        if service_name:
            query += " AND service_name = ?"
            params.append(service_name)
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
        if case_id:
            query += " AND case_id = ?"
            params.append(case_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._managed_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._result_from_row(row) for row in rows]

    def get_result(self, result_id: str) -> Optional[BenchmarkResult]:
        with self._managed_connection() as conn:
            row = conn.execute(
                "SELECT * FROM benchmark_results WHERE result_id = ?",
                (result_id,),
            ).fetchone()
        return self._result_from_row(row) if row is not None else None

    def upsert_manual_review(
        self,
        *,
        result_id: str,
        reviewer: str,
        score: Optional[float],
        notes: Optional[str],
        created_at: str,
        updated_at: str,
    ) -> BenchmarkManualReview:
        with self._managed_connection() as conn:
            row = conn.execute(
                "SELECT review_id, created_at FROM benchmark_manual_reviews WHERE result_id = ?",
                (result_id,),
            ).fetchone()
            review_id = row["review_id"] if row is not None else uuid.uuid4().hex
            original_created_at = row["created_at"] if row is not None else created_at
            conn.execute(
                """
                INSERT OR REPLACE INTO benchmark_manual_reviews (
                    review_id, result_id, created_at, updated_at, reviewer, score, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    review_id,
                    result_id,
                    original_created_at,
                    updated_at,
                    reviewer,
                    score,
                    notes,
                ),
            )
        return BenchmarkManualReview(
            review_id=review_id,
            result_id=result_id,
            created_at=original_created_at,
            updated_at=updated_at,
            reviewer=reviewer,
            score=score,
            notes=notes,
        )

    def get_manual_review(self, result_id: str) -> Optional[BenchmarkManualReview]:
        with self._managed_connection() as conn:
            row = conn.execute(
                "SELECT * FROM benchmark_manual_reviews WHERE result_id = ?",
                (result_id,),
            ).fetchone()
        return self._review_from_row(row) if row is not None else None

    def _run_from_row(self, row: sqlite3.Row) -> BenchmarkRun:
        return BenchmarkRun(
            run_id=row["run_id"],
            created_at=row["created_at"],
            completed_at=row["completed_at"],
            service_name=row["service_name"],
            model_names_json=row["model_names_json"],
            case_ids_json=row["case_ids_json"],
            status=row["status"],
            notes=row["notes"],
        )

    def _result_from_row(self, row: sqlite3.Row) -> BenchmarkResult:
        return BenchmarkResult(
            result_id=row["result_id"],
            run_id=row["run_id"],
            created_at=row["created_at"],
            completed_at=row["completed_at"],
            service_name=row["service_name"],
            case_id=row["case_id"],
            case_title=row["case_title"],
            model_name=row["model_name"],
            screenshot_path=row["screenshot_path"],
            transcript_path=row["transcript_path"],
            response_text=row["response_text"],
            structured_output_json=row["structured_output_json"],
            error_text=row["error_text"],
            prompt_tokens=row["prompt_tokens"],
            completion_tokens=row["completion_tokens"],
            total_tokens=row["total_tokens"],
            prefill_seconds=row["prefill_seconds"],
            generation_seconds=row["generation_seconds"],
            prefill_tokens_per_second=row["prefill_tokens_per_second"],
            generation_tokens_per_second=row["generation_tokens_per_second"],
            token_count_method=row["token_count_method"],
            auto_score=row["auto_score"],
            auto_score_details_json=row["auto_score_details_json"],
            metadata_json=row["metadata_json"],
            status=row["status"],
        )

    def _review_from_row(self, row: sqlite3.Row) -> BenchmarkManualReview:
        return BenchmarkManualReview(
            review_id=row["review_id"],
            result_id=row["result_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            reviewer=row["reviewer"],
            score=row["score"],
            notes=row["notes"],
        )
