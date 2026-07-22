from abc import ABC, abstractmethod
from typing import Any, List, Optional
from core.models import NightTask


class TaskQueuePort(ABC):
    """Port for night-time task queue operations."""

    @abstractmethod
    def get_pending_tasks(self) -> List[NightTask]:
        """Retrieve pending idle/night tasks that have no exact run time."""
        pass

    @abstractmethod
    def get_due_tasks(self, now_utc: str) -> List[NightTask]:
        """Retrieve exact-time tasks due at or before the supplied UTC time."""
        pass

    @abstractmethod
    def get_all_pending_tasks(self) -> List[NightTask]:
        """Retrieve both idle and exact-time pending tasks for display."""
        pass

    @abstractmethod
    def add_task(
        self,
        description: str,
        priority: str = "medium",
        metadata: Optional[dict[str, Any]] = None,
        run_at_utc: Optional[str] = None,
    ) -> str:
        """Add a new task to the queue. Returns confirmation message."""
        pass

    @abstractmethod
    def mark_task_complete(self, task_id: int, status: str = "completed") -> None:
        """Mark a queued task as completed."""
        pass

    @abstractmethod
    def claim_task(self, task_id: int) -> bool:
        """Atomically mark a pending task as running."""
        pass

    @abstractmethod
    def cancel_task(self, task_id: int) -> bool:
        """Cancel a task that has not started."""
        pass
