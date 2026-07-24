from abc import ABC, abstractmethod
from typing import Any, Optional

from core.models import AmbientEvent, ApprovalGrant, OpportunityCandidate, ProactiveInboxItem


class AutonomyStorePort(ABC):
    """Durable control-plane storage for events, policies, approvals, and inbox items."""

    @abstractmethod
    def enqueue_event(self, event: AmbientEvent) -> AmbientEvent:
        pass

    @abstractmethod
    def claim_next_event(
        self,
        *,
        lease_seconds: int = 180,
        event_types: Optional[list[str]] = None,
    ) -> Optional[AmbientEvent]:
        pass

    @abstractmethod
    def complete_event(self, event_id: str, *, status: str = "processed", error_text: str | None = None) -> None:
        pass

    @abstractmethod
    def retry_event(self, event_id: str, *, error_text: str, delay_seconds: int = 30, max_attempts: int = 3) -> None:
        pass

    @abstractmethod
    def defer_event(self, event_id: str, *, reason: str, delay_seconds: int = 30) -> None:
        """Release a lease without counting resource unavailability as an attempt."""
        pass

    @abstractmethod
    def upsert_opportunity(self, candidate: OpportunityCandidate) -> OpportunityCandidate:
        pass

    @abstractmethod
    def add_inbox_item(self, item: ProactiveInboxItem) -> ProactiveInboxItem:
        pass

    @abstractmethod
    def list_inbox_items(self, *, limit: int = 50, status: str | None = None) -> list[ProactiveInboxItem]:
        pass

    @abstractmethod
    def record_feedback(self, inbox_id: str, feedback: str) -> bool:
        pass

    @abstractmethod
    def get_policy(self, capability: str) -> Optional[dict[str, Any]]:
        pass

    @abstractmethod
    def set_policy(self, capability: str, decision: str, constraints: dict[str, Any] | None = None) -> dict[str, Any]:
        pass

    @abstractmethod
    def create_approval(self, approval: ApprovalGrant) -> ApprovalGrant:
        pass

    @abstractmethod
    def find_valid_approval(self, capability: str, action_fingerprint: str) -> Optional[ApprovalGrant]:
        pass
