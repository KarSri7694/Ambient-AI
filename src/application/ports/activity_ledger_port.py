from abc import ABC, abstractmethod
from typing import Optional

from core.models import ActivityArtifact, ActivityLink, ActivityRun, ActivityRunDetail, ActivityStep


class ActivityLedgerPort(ABC):
    """Port for the user-facing activity ledger."""

    @abstractmethod
    def create_run(
        self,
        *,
        source_kind: str,
        trigger_kind: str,
        title: str,
        status: str = "queued",
        summary: str = "",
        output_text: str = "",
        model: str = "",
        session_id: str | None = None,
        parent_run_id: str | None = None,
        priority: str = "medium",
        error_text: str | None = None,
        metadata: dict | None = None,
        tags: list[str] | None = None,
    ) -> ActivityRun:
        pass

    @abstractmethod
    def update_run_status(
        self,
        run_id: str,
        *,
        status: str,
        summary: str | None = None,
        output_text: str | None = None,
        completed_at: str | None = None,
        error_text: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        pass

    @abstractmethod
    def append_step(
        self,
        *,
        run_id: str,
        step_kind: str,
        title: str,
        status: str = "running",
        input_ref: str | None = None,
        output_ref: str | None = None,
        error_text: str | None = None,
        metadata: dict | None = None,
    ) -> ActivityStep:
        pass

    @abstractmethod
    def complete_step(
        self,
        step_id: str,
        *,
        status: str,
        completed_at: str | None = None,
        output_ref: str | None = None,
        error_text: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def link_entity(
        self,
        *,
        run_id: str,
        entity_type: str,
        entity_id: str,
        relation: str,
        metadata: dict | None = None,
    ) -> ActivityLink:
        pass

    @abstractmethod
    def add_tags(self, run_id: str, tags: list[str]) -> None:
        pass

    @abstractmethod
    def list_runs(
        self,
        *,
        status: str | None = None,
        source_kind: str | None = None,
        trigger_kind: str | None = None,
        tag: str | None = None,
        query: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 50,
    ) -> list[ActivityRun]:
        pass

    @abstractmethod
    def get_run_detail(self, run_id: str) -> Optional[ActivityRunDetail]:
        pass

    @abstractmethod
    def search_runs(self, query: str, limit: int = 50) -> list[ActivityRun]:
        pass
