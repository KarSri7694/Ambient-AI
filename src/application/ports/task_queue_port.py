from abc import ABC, abstractmethod
from typing import List
from core.models import NightTask


class TaskQueuePort(ABC):
    """Port for night-time task queue operations."""

    @abstractmethod
    def get_pending_tasks(self) -> List[NightTask]:
        """Retrieve all pending tasks from the queue."""
        pass

    @abstractmethod
    def add_task(self, description: str, priority: str = "medium") -> str:
        """Add a new task to the queue. Returns confirmation message."""
        pass

    @abstractmethod
    def mark_task_complete(self, task_id: int, status: str = "completed") -> None:
        """Mark a queued task as completed."""
        pass
