import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.models import (
    InteractionLogEntry,
    TrainingASRRecord,
    TrainingDatasetExport,
    TrainingLLMRecord,
)
from infrastructure.adapter.SQLiteInteractionLogAdapter import SQLiteInteractionLogAdapter
from infrastructure.adapter.SQLiteTrainingDataAdapter import SQLiteTrainingDataAdapter


class TrainingDataService:
    """Syncs raw runtime data into reviewable training tables and exports curated datasets."""

    def __init__(
        self,
        *,
        store: SQLiteTrainingDataAdapter,
        interaction_store: SQLiteInteractionLogAdapter,
        training_root: str,
        user_data_dir: str,
        uploads_dir: str,
        transcripts_dir: str,
        cleaned_audio_dir: str,
    ):
        self.store = store
        self.interaction_store = interaction_store
        self.training_root = Path(training_root)
        self.user_data_dir = Path(user_data_dir)
        self.uploads_dir = Path(uploads_dir)
        self.transcripts_dir = Path(transcripts_dir)
        self.cleaned_audio_dir = Path(cleaned_audio_dir)

    def sync_llm_records(self, *, batch_size: int = 500, max_batches: int = 20) -> Dict[str, int]:
        synced = 0
        offset = 0
        for _ in range(max_batches):
            rows = self.interaction_store.list_entries(limit=batch_size, offset=offset)
            if not rows:
                break
            for row in rows:
                self.store.upsert_llm_record(self._build_llm_record(row))
                synced += 1
            offset += len(rows)
            if len(rows) < batch_size:
                break
        return {"synced": synced}

    def sync_asr_records(self) -> Dict[str, int]:
        if not self.transcripts_dir.exists():
            return {"synced": 0}
        synced = 0
        upload_candidates = self._build_media_index(self.uploads_dir, suffixes={".wav", ".mp3", ".m4a", ".ogg", ".webm"})
        cleaned_candidates = self._build_media_index(self.cleaned_audio_dir, suffixes={".wav", ".mp3", ".m4a", ".ogg", ".webm"})
        for transcript_path in sorted(self.transcripts_dir.glob("*.txt")):
            text = transcript_path.read_text(encoding="utf-8").strip()
            created_at = datetime.fromtimestamp(transcript_path.stat().st_mtime).isoformat()
            upload_audio = self._match_audio_for_transcript(transcript_path, upload_candidates, suffix_strip="")
            cleaned_audio = self._match_audio_for_transcript(transcript_path, cleaned_candidates, suffix_strip="_final")
            metadata = {
                "transcript_name": transcript_path.name,
                "upload_audio_exists": bool(upload_audio),
                "cleaned_audio_exists": bool(cleaned_audio),
            }
            self.store.upsert_asr_record(
                TrainingASRRecord(
                    record_id=uuid.uuid4().hex,
                    transcript_path=str(transcript_path.resolve()),
                    created_at=created_at,
                    transcript_text=text,
                    upload_audio_path=upload_audio,
                    cleaned_audio_path=cleaned_audio,
                    metadata_json=json.dumps(metadata, ensure_ascii=False, indent=2),
                )
            )
            synced += 1
        return {"synced": synced}

    def export_llm_dataset(
        self,
        *,
        statuses: Optional[List[str]] = None,
        output_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        selected_statuses = statuses or ["approved"]
        records = self.store.list_llm_records_for_export(statuses=selected_statuses)
        export_dir = self.training_root / "llm" / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = export_dir / (output_name or f"llm_sft_{timestamp}.jsonl")
        lines: List[str] = []
        for record in records:
            review = self.store.get_llm_review(record.record_id)
            payload = {
                "record_id": record.record_id,
                "interaction_id": record.interaction_id,
                "interaction_run_id": record.interaction_run_id,
                "created_at": record.created_at,
                "source": record.source,
                "model": record.model,
                "messages": self._safe_json_load(review.corrected_messages_json if review else None, fallback=self._safe_json_load(record.messages_json, fallback=[])),
                "assistant_response": (review.corrected_response_text if review and review.corrected_response_text is not None else record.response_text) or "",
                "reasoning_text": (review.corrected_reasoning_text if review and review.corrected_reasoning_text is not None else record.reasoning_text),
                "tools": self._safe_json_load(record.tools_json, fallback=None),
                "tool_calls": self._safe_json_load(record.tool_calls_json, fallback=None),
                "image_path": record.image_path,
                "error_text": record.error_text,
                "metadata": self._safe_json_load(record.metadata_json, fallback={}),
                "report": self._safe_json_load(record.report_json, fallback=None),
                "review": {
                    "status": review.status if review else record.review_status,
                    "reviewer": review.reviewer if review else None,
                    "notes": review.notes if review else None,
                },
            }
            lines.append(json.dumps(payload, ensure_ascii=False))
        output_path.write_text("\n".join(lines), encoding="utf-8")
        export = TrainingDatasetExport(
            export_id=uuid.uuid4().hex,
            dataset_kind="llm",
            created_at=datetime.now().isoformat(),
            output_path=str(output_path.resolve()),
            record_count=len(records),
            metadata_json=json.dumps({"statuses": selected_statuses}, ensure_ascii=False),
        )
        self.store.insert_export(export)
        return {"output_path": export.output_path, "record_count": export.record_count}

    def export_asr_dataset(
        self,
        *,
        statuses: Optional[List[str]] = None,
        output_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        selected_statuses = statuses or ["approved"]
        records = self.store.list_asr_records_for_export(statuses=selected_statuses)
        export_dir = self.training_root / "asr" / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = export_dir / (output_name or f"asr_manifest_{timestamp}.jsonl")
        lines: List[str] = []
        for record in records:
            review = self.store.get_asr_review(record.record_id)
            preferred_audio = record.cleaned_audio_path or record.upload_audio_path
            payload = {
                "record_id": record.record_id,
                "created_at": record.created_at,
                "audio_path": preferred_audio,
                "upload_audio_path": record.upload_audio_path,
                "cleaned_audio_path": record.cleaned_audio_path,
                "transcript": (review.corrected_transcript_text if review and review.corrected_transcript_text is not None else record.transcript_text) or "",
                "raw_transcript_path": record.transcript_path,
                "metadata": self._safe_json_load(record.metadata_json, fallback={}),
                "review": {
                    "status": review.status if review else record.review_status,
                    "reviewer": review.reviewer if review else None,
                    "notes": review.notes if review else None,
                },
            }
            lines.append(json.dumps(payload, ensure_ascii=False))
        output_path.write_text("\n".join(lines), encoding="utf-8")
        export = TrainingDatasetExport(
            export_id=uuid.uuid4().hex,
            dataset_kind="asr",
            created_at=datetime.now().isoformat(),
            output_path=str(output_path.resolve()),
            record_count=len(records),
            metadata_json=json.dumps({"statuses": selected_statuses}, ensure_ascii=False),
        )
        self.store.insert_export(export)
        return {"output_path": export.output_path, "record_count": export.record_count}

    def _build_llm_record(self, row: InteractionLogEntry) -> TrainingLLMRecord:
        return TrainingLLMRecord(
            record_id=uuid.uuid4().hex,
            interaction_id=row.interaction_id,
            interaction_run_id=row.interaction_run_id,
            created_at=row.created_at,
            completed_at=row.completed_at,
            source=row.source,
            model=row.model,
            messages_json=row.messages_json,
            tools_json=row.tools_json,
            image_path=row.image_path,
            response_text=row.response_text,
            reasoning_text=row.reasoning_text,
            tool_calls_json=row.tool_calls_json,
            error_text=row.error_text,
            duration_ms=row.duration_ms,
            metadata_json=row.metadata_json,
            report_json=row.report_json,
        )

    def _build_media_index(self, root: Path, *, suffixes: set[str]) -> List[Path]:
        if not root.exists():
            return []
        return [path for path in root.iterdir() if path.is_file() and path.suffix.lower() in suffixes]

    def _match_audio_for_transcript(
        self,
        transcript_path: Path,
        candidates: List[Path],
        *,
        suffix_strip: str,
    ) -> Optional[str]:
        if not candidates:
            return None
        transcript_mtime = transcript_path.stat().st_mtime
        best_path: Optional[Path] = None
        best_score: Optional[float] = None
        transcript_stem = transcript_path.stem.lower()
        for candidate in candidates:
            stem = candidate.stem.lower()
            normalized = stem[:-len(suffix_strip)] if suffix_strip and stem.endswith(suffix_strip) else stem
            score = abs(candidate.stat().st_mtime - transcript_mtime)
            if normalized in transcript_stem or transcript_stem in normalized:
                score -= 300.0
            if best_score is None or score < best_score:
                best_score = score
                best_path = candidate
        return str(best_path.resolve()) if best_path is not None else None

    def _safe_json_load(self, value: Optional[str], *, fallback: Any) -> Any:
        if value is None or not str(value).strip():
            return fallback
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return fallback
