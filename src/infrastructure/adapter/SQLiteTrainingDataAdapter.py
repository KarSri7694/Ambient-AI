import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

from core.models import (
    TrainingASRRecord,
    TrainingASRReview,
    TrainingDatasetExport,
    TrainingLLMRecord,
    TrainingLLMReview,
)


class SQLiteTrainingDataAdapter:
    """SQLite-backed review and export store for LLM and ASR training data."""

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
                CREATE TABLE IF NOT EXISTS training_llm_records (
                    record_id TEXT PRIMARY KEY,
                    interaction_id TEXT NOT NULL UNIQUE,
                    interaction_run_id TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    source TEXT NOT NULL,
                    model TEXT NOT NULL,
                    messages_json TEXT NOT NULL,
                    tools_json TEXT,
                    image_path TEXT,
                    response_text TEXT,
                    reasoning_text TEXT,
                    tool_calls_json TEXT,
                    error_text TEXT,
                    duration_ms INTEGER,
                    metadata_json TEXT,
                    report_json TEXT,
                    review_status TEXT NOT NULL DEFAULT 'pending',
                    review_updated_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS training_llm_reviews (
                    review_id TEXT PRIMARY KEY,
                    record_id TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    reviewer TEXT NOT NULL,
                    status TEXT NOT NULL,
                    corrected_response_text TEXT,
                    corrected_reasoning_text TEXT,
                    corrected_messages_json TEXT,
                    notes TEXT,
                    FOREIGN KEY (record_id) REFERENCES training_llm_records (record_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS training_asr_records (
                    record_id TEXT PRIMARY KEY,
                    transcript_path TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    transcript_text TEXT NOT NULL,
                    upload_audio_path TEXT,
                    cleaned_audio_path TEXT,
                    metadata_json TEXT,
                    review_status TEXT NOT NULL DEFAULT 'pending',
                    review_updated_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS training_asr_reviews (
                    review_id TEXT PRIMARY KEY,
                    record_id TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    reviewer TEXT NOT NULL,
                    status TEXT NOT NULL,
                    corrected_transcript_text TEXT,
                    notes TEXT,
                    FOREIGN KEY (record_id) REFERENCES training_asr_records (record_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS training_dataset_exports (
                    export_id TEXT PRIMARY KEY,
                    dataset_kind TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    output_path TEXT NOT NULL,
                    record_count INTEGER NOT NULL,
                    metadata_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_training_llm_records_created_at
                ON training_llm_records (created_at DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_training_llm_records_review_status
                ON training_llm_records (review_status, created_at DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_training_asr_records_created_at
                ON training_asr_records (created_at DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_training_asr_records_review_status
                ON training_asr_records (review_status, created_at DESC)
                """
            )

    def upsert_llm_record(self, record: TrainingLLMRecord) -> None:
        with self._managed_connection() as conn:
            existing = conn.execute(
                "SELECT record_id, review_status, review_updated_at FROM training_llm_records WHERE interaction_id = ?",
                (record.interaction_id,),
            ).fetchone()
            persisted_record_id = existing["record_id"] if existing is not None else record.record_id
            review_status = existing["review_status"] if existing is not None else record.review_status
            review_updated_at = existing["review_updated_at"] if existing is not None else record.review_updated_at
            conn.execute(
                """
                INSERT OR REPLACE INTO training_llm_records (
                    record_id, interaction_id, interaction_run_id, created_at, completed_at, source,
                    model, messages_json, tools_json, image_path, response_text, reasoning_text,
                    tool_calls_json, error_text, duration_ms, metadata_json, report_json,
                    review_status, review_updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    persisted_record_id,
                    record.interaction_id,
                    record.interaction_run_id,
                    record.created_at,
                    record.completed_at,
                    record.source,
                    record.model,
                    record.messages_json,
                    record.tools_json,
                    record.image_path,
                    record.response_text,
                    record.reasoning_text,
                    record.tool_calls_json,
                    record.error_text,
                    record.duration_ms,
                    record.metadata_json,
                    record.report_json,
                    review_status,
                    review_updated_at,
                ),
            )

    def list_llm_records(
        self,
        *,
        limit: int = 100,
        source: Optional[str] = None,
        model: Optional[str] = None,
        review_status: Optional[str] = None,
    ) -> List[TrainingLLMRecord]:
        query = "SELECT * FROM training_llm_records WHERE 1 = 1"
        params: List[object] = []
        if source:
            query += " AND source = ?"
            params.append(source)
        if model:
            query += " AND model = ?"
            params.append(model)
        if review_status:
            query += " AND review_status = ?"
            params.append(review_status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._managed_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._llm_record_from_row(row) for row in rows]

    def get_llm_record(self, record_id: str) -> Optional[TrainingLLMRecord]:
        with self._managed_connection() as conn:
            row = conn.execute(
                "SELECT * FROM training_llm_records WHERE record_id = ?",
                (record_id,),
            ).fetchone()
        return self._llm_record_from_row(row) if row is not None else None

    def upsert_llm_review(
        self,
        *,
        record_id: str,
        reviewer: str,
        status: str,
        corrected_response_text: Optional[str],
        corrected_reasoning_text: Optional[str],
        corrected_messages_json: Optional[str],
        notes: Optional[str],
        created_at: str,
        updated_at: str,
    ) -> TrainingLLMReview:
        with self._managed_connection() as conn:
            row = conn.execute(
                "SELECT review_id, created_at FROM training_llm_reviews WHERE record_id = ?",
                (record_id,),
            ).fetchone()
            review_id = row["review_id"] if row is not None else uuid.uuid4().hex
            original_created_at = row["created_at"] if row is not None else created_at
            conn.execute(
                """
                INSERT OR REPLACE INTO training_llm_reviews (
                    review_id, record_id, created_at, updated_at, reviewer, status,
                    corrected_response_text, corrected_reasoning_text, corrected_messages_json, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    review_id,
                    record_id,
                    original_created_at,
                    updated_at,
                    reviewer,
                    status,
                    corrected_response_text,
                    corrected_reasoning_text,
                    corrected_messages_json,
                    notes,
                ),
            )
            conn.execute(
                """
                UPDATE training_llm_records
                SET review_status = ?, review_updated_at = ?
                WHERE record_id = ?
                """,
                (status, updated_at, record_id),
            )
        return TrainingLLMReview(
            review_id=review_id,
            record_id=record_id,
            created_at=original_created_at,
            updated_at=updated_at,
            reviewer=reviewer,
            status=status,
            corrected_response_text=corrected_response_text,
            corrected_reasoning_text=corrected_reasoning_text,
            corrected_messages_json=corrected_messages_json,
            notes=notes,
        )

    def get_llm_review(self, record_id: str) -> Optional[TrainingLLMReview]:
        with self._managed_connection() as conn:
            row = conn.execute(
                "SELECT * FROM training_llm_reviews WHERE record_id = ?",
                (record_id,),
            ).fetchone()
        return self._llm_review_from_row(row) if row is not None else None

    def list_llm_records_for_export(self, *, statuses: List[str]) -> List[TrainingLLMRecord]:
        placeholders = ", ".join("?" for _ in statuses)
        query = (
            f"SELECT * FROM training_llm_records WHERE review_status IN ({placeholders}) "
            "ORDER BY created_at ASC"
        )
        with self._managed_connection() as conn:
            rows = conn.execute(query, list(statuses)).fetchall()
        return [self._llm_record_from_row(row) for row in rows]

    def upsert_asr_record(self, record: TrainingASRRecord) -> None:
        with self._managed_connection() as conn:
            existing = conn.execute(
                "SELECT record_id, review_status, review_updated_at FROM training_asr_records WHERE transcript_path = ?",
                (record.transcript_path,),
            ).fetchone()
            persisted_record_id = existing["record_id"] if existing is not None else record.record_id
            review_status = existing["review_status"] if existing is not None else record.review_status
            review_updated_at = existing["review_updated_at"] if existing is not None else record.review_updated_at
            conn.execute(
                """
                INSERT OR REPLACE INTO training_asr_records (
                    record_id, transcript_path, created_at, transcript_text, upload_audio_path,
                    cleaned_audio_path, metadata_json, review_status, review_updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    persisted_record_id,
                    record.transcript_path,
                    record.created_at,
                    record.transcript_text,
                    record.upload_audio_path,
                    record.cleaned_audio_path,
                    record.metadata_json,
                    review_status,
                    review_updated_at,
                ),
            )

    def list_asr_records(self, *, limit: int = 100, review_status: Optional[str] = None) -> List[TrainingASRRecord]:
        query = "SELECT * FROM training_asr_records WHERE 1 = 1"
        params: List[object] = []
        if review_status:
            query += " AND review_status = ?"
            params.append(review_status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._managed_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._asr_record_from_row(row) for row in rows]

    def get_asr_record(self, record_id: str) -> Optional[TrainingASRRecord]:
        with self._managed_connection() as conn:
            row = conn.execute(
                "SELECT * FROM training_asr_records WHERE record_id = ?",
                (record_id,),
            ).fetchone()
        return self._asr_record_from_row(row) if row is not None else None

    def upsert_asr_review(
        self,
        *,
        record_id: str,
        reviewer: str,
        status: str,
        corrected_transcript_text: Optional[str],
        notes: Optional[str],
        created_at: str,
        updated_at: str,
    ) -> TrainingASRReview:
        with self._managed_connection() as conn:
            row = conn.execute(
                "SELECT review_id, created_at FROM training_asr_reviews WHERE record_id = ?",
                (record_id,),
            ).fetchone()
            review_id = row["review_id"] if row is not None else uuid.uuid4().hex
            original_created_at = row["created_at"] if row is not None else created_at
            conn.execute(
                """
                INSERT OR REPLACE INTO training_asr_reviews (
                    review_id, record_id, created_at, updated_at, reviewer, status,
                    corrected_transcript_text, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    review_id,
                    record_id,
                    original_created_at,
                    updated_at,
                    reviewer,
                    status,
                    corrected_transcript_text,
                    notes,
                ),
            )
            conn.execute(
                """
                UPDATE training_asr_records
                SET review_status = ?, review_updated_at = ?
                WHERE record_id = ?
                """,
                (status, updated_at, record_id),
            )
        return TrainingASRReview(
            review_id=review_id,
            record_id=record_id,
            created_at=original_created_at,
            updated_at=updated_at,
            reviewer=reviewer,
            status=status,
            corrected_transcript_text=corrected_transcript_text,
            notes=notes,
        )

    def get_asr_review(self, record_id: str) -> Optional[TrainingASRReview]:
        with self._managed_connection() as conn:
            row = conn.execute(
                "SELECT * FROM training_asr_reviews WHERE record_id = ?",
                (record_id,),
            ).fetchone()
        return self._asr_review_from_row(row) if row is not None else None

    def list_asr_records_for_export(self, *, statuses: List[str]) -> List[TrainingASRRecord]:
        placeholders = ", ".join("?" for _ in statuses)
        query = (
            f"SELECT * FROM training_asr_records WHERE review_status IN ({placeholders}) "
            "ORDER BY created_at ASC"
        )
        with self._managed_connection() as conn:
            rows = conn.execute(query, list(statuses)).fetchall()
        return [self._asr_record_from_row(row) for row in rows]

    def insert_export(self, export: TrainingDatasetExport) -> None:
        with self._managed_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO training_dataset_exports (
                    export_id, dataset_kind, created_at, output_path, record_count, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    export.export_id,
                    export.dataset_kind,
                    export.created_at,
                    export.output_path,
                    export.record_count,
                    export.metadata_json,
                ),
            )

    def list_exports(self, *, dataset_kind: Optional[str] = None, limit: int = 50) -> List[TrainingDatasetExport]:
        query = "SELECT * FROM training_dataset_exports"
        params: List[object] = []
        if dataset_kind:
            query += " WHERE dataset_kind = ?"
            params.append(dataset_kind)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._managed_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._export_from_row(row) for row in rows]

    def _llm_record_from_row(self, row: sqlite3.Row) -> TrainingLLMRecord:
        return TrainingLLMRecord(
            record_id=row["record_id"],
            interaction_id=row["interaction_id"],
            interaction_run_id=row["interaction_run_id"],
            created_at=row["created_at"],
            completed_at=row["completed_at"],
            source=row["source"],
            model=row["model"],
            messages_json=row["messages_json"],
            tools_json=row["tools_json"],
            image_path=row["image_path"],
            response_text=row["response_text"],
            reasoning_text=row["reasoning_text"],
            tool_calls_json=row["tool_calls_json"],
            error_text=row["error_text"],
            duration_ms=row["duration_ms"],
            metadata_json=row["metadata_json"],
            report_json=row["report_json"],
            review_status=row["review_status"],
            review_updated_at=row["review_updated_at"],
        )

    def _llm_review_from_row(self, row: sqlite3.Row) -> TrainingLLMReview:
        return TrainingLLMReview(
            review_id=row["review_id"],
            record_id=row["record_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            reviewer=row["reviewer"],
            status=row["status"],
            corrected_response_text=row["corrected_response_text"],
            corrected_reasoning_text=row["corrected_reasoning_text"],
            corrected_messages_json=row["corrected_messages_json"],
            notes=row["notes"],
        )

    def _asr_record_from_row(self, row: sqlite3.Row) -> TrainingASRRecord:
        return TrainingASRRecord(
            record_id=row["record_id"],
            transcript_path=row["transcript_path"],
            created_at=row["created_at"],
            transcript_text=row["transcript_text"],
            upload_audio_path=row["upload_audio_path"],
            cleaned_audio_path=row["cleaned_audio_path"],
            metadata_json=row["metadata_json"],
            review_status=row["review_status"],
            review_updated_at=row["review_updated_at"],
        )

    def _asr_review_from_row(self, row: sqlite3.Row) -> TrainingASRReview:
        return TrainingASRReview(
            review_id=row["review_id"],
            record_id=row["record_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            reviewer=row["reviewer"],
            status=row["status"],
            corrected_transcript_text=row["corrected_transcript_text"],
            notes=row["notes"],
        )

    def _export_from_row(self, row: sqlite3.Row) -> TrainingDatasetExport:
        return TrainingDatasetExport(
            export_id=row["export_id"],
            dataset_kind=row["dataset_kind"],
            created_at=row["created_at"],
            output_path=row["output_path"],
            record_count=row["record_count"],
            metadata_json=row["metadata_json"],
        )
