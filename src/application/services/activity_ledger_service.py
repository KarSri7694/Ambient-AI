from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from application.ports.activity_ledger_port import ActivityLedgerPort
from application.services.interaction_trace import interaction_trace
from core.models import ActivityArtifact, ActivityRun, ActivityStep


@dataclass(frozen=True)
class ActivityContext:
    run: ActivityRun
    step: Optional[ActivityStep] = None


class ActivityLedgerService:
    """Owns the run and step lifecycle used by higher-level workflows."""

    def __init__(self, ledger: ActivityLedgerPort):
        self.ledger = ledger

    def start_run(
        self,
        *,
        source_kind: str,
        trigger_kind: str,
        title: str,
        status: str = "running",
        summary: str = "",
        output_text: str = "",
        model: str = "",
        session_id: str | None = None,
        parent_run_id: str | None = None,
        priority: str = "medium",
        metadata: dict | None = None,
        tags: list[str] | None = None,
    ) -> ActivityRun:
        return self.ledger.create_run(
            source_kind=source_kind,
            trigger_kind=trigger_kind,
            title=title,
            status=status,
            summary=summary,
            output_text=output_text,
            model=model,
            session_id=session_id,
            parent_run_id=parent_run_id,
            priority=priority,
            metadata=metadata,
            tags=tags,
        )

    def complete_run(
        self,
        run_id: str,
        *,
        summary: str = "",
        output_text: str = "",
        metadata: dict | None = None,
    ) -> None:
        self.ledger.update_run_status(
            run_id,
            status="completed",
            summary=summary,
            output_text=output_text,
            completed_at=datetime.now().isoformat(),
            metadata=metadata,
        )

    def fail_run(
        self,
        run_id: str,
        *,
        error_text: str,
        summary: str = "",
        output_text: str = "",
        metadata: dict | None = None,
    ) -> None:
        self.ledger.update_run_status(
            run_id,
            status="failed",
            summary=summary,
            output_text=output_text,
            completed_at=datetime.now().isoformat(),
            error_text=error_text,
            metadata=metadata,
        )

    def queue_run(
        self,
        *,
        source_kind: str,
        trigger_kind: str,
        title: str,
        summary: str = "",
        output_text: str = "",
        model: str = "",
        session_id: str | None = None,
        parent_run_id: str | None = None,
        priority: str = "medium",
        metadata: dict | None = None,
        tags: list[str] | None = None,
    ) -> ActivityRun:
        return self.start_run(
            source_kind=source_kind,
            trigger_kind=trigger_kind,
            title=title,
            status="queued",
            summary=summary,
            output_text=output_text,
            model=model,
            session_id=session_id,
            parent_run_id=parent_run_id,
            priority=priority,
            metadata=metadata,
            tags=tags,
        )

    def start_step(
        self,
        run_id: str,
        *,
        step_kind: str,
        title: str,
        input_ref: str | None = None,
        metadata: dict | None = None,
    ) -> ActivityStep:
        return self.ledger.append_step(
            run_id=run_id,
            step_kind=step_kind,
            title=title,
            status="running",
            input_ref=input_ref,
            metadata=metadata,
        )

    def complete_step(
        self,
        step_id: str,
        *,
        output_ref: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        self.ledger.complete_step(
            step_id,
            status="completed",
            output_ref=output_ref,
            metadata=metadata,
        )

    def fail_step(
        self,
        step_id: str,
        *,
        error_text: str,
        output_ref: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        self.ledger.complete_step(
            step_id,
            status="failed",
            output_ref=output_ref,
            error_text=error_text,
            metadata=metadata,
        )

    def attach_artifact(
        self,
        *,
        run_id: str,
        artifact_kind: str,
        title: str,
        step_id: str | None = None,
        path: str | None = None,
        mime_type: str | None = None,
        text_preview: str | None = None,
        metadata: dict | None = None,
    ) -> ActivityArtifact:
        return self.ledger.attach_artifact(
            run_id=run_id,
            step_id=step_id,
            artifact_kind=artifact_kind,
            title=title,
            path=path,
            mime_type=mime_type,
            text_preview=text_preview,
            metadata=metadata,
        )

    def attach_file_artifact(
        self,
        *,
        run_id: str,
        artifact_kind: str,
        title: str,
        path: str,
        step_id: str | None = None,
        mime_type: str | None = None,
        preview_chars: int = 300,
        metadata: dict | None = None,
    ) -> ActivityArtifact:
        preview = None
        file_path = Path(path)
        if file_path.exists() and file_path.is_file():
            try:
                preview = file_path.read_text(encoding="utf-8")[:preview_chars]
            except (OSError, UnicodeDecodeError):
                preview = None
        return self.attach_artifact(
            run_id=run_id,
            artifact_kind=artifact_kind,
            title=title,
            step_id=step_id,
            path=path,
            mime_type=mime_type,
            text_preview=preview,
            metadata=metadata,
        )

    def link_entity(
        self,
        *,
        run_id: str,
        entity_type: str,
        entity_id: str,
        relation: str,
        metadata: dict | None = None,
    ) -> None:
        self.ledger.link_entity(
            run_id=run_id,
            entity_type=entity_type,
            entity_id=entity_id,
            relation=relation,
            metadata=metadata,
        )

    @contextmanager
    def interaction_scope(
        self,
        run_id: str,
        *,
        step_id: str | None = None,
        metadata: dict | None = None,
        source: str,
    ) -> Iterator[ActivityContext]:
        payload = {"run_id": run_id}
        if step_id:
            payload["step_id"] = step_id
        if metadata:
            payload.update(metadata)
        with interaction_trace(source, payload):
            yield ActivityContext(run=self.ledger.get_run_detail(run_id).run, step=None)
