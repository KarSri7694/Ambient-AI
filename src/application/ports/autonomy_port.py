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
        exclude_event_types: Optional[list[str]] = None,
    ) -> Optional[AmbientEvent]:
        pass

    def claim_next_events(
        self,
        *,
        lease_seconds: int = 180,
        event_types: Optional[list[str]] = None,
        exclude_event_types: Optional[list[str]] = None,
        limit: int = 1,
    ) -> list[AmbientEvent]:
        events: list[AmbientEvent] = []
        for _ in range(max(1, int(limit))):
            kwargs = {"lease_seconds": lease_seconds, "event_types": event_types}
            if exclude_event_types:
                kwargs["exclude_event_types"] = exclude_event_types
            event = self.claim_next_event(**kwargs)
            if event is None:
                break
            events.append(event)
        return events

    @abstractmethod
    def complete_event(self, event_id: str, *, status: str = "processed", error_text: str | None = None) -> None:
        """Mark an event terminal. Status may include processed, ignored, dead_letter, or interrupted."""
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

    def list_feedback_signals(self, *, query_text: str = "", limit: int = 8) -> list[dict[str, Any]]:
        """Return bounded, durable user feedback for personalization.

        Implementations may use the query to select relevant signals. Feedback is
        preference evidence only; it never grants a capability or replaces source
        evidence.
        """
        return []

    def list_feedback_for_inbox(self, inbox_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Return the local audit trail for a displayed proactive result."""
        return []

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
