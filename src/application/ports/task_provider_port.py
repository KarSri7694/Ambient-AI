from abc import ABC, abstractmethod
from typing import List, Dict


class TaskProviderPort(ABC):
    """Port for external task providers (e.g. Todoist, Jira, etc.)."""

    @abstractmethod
    def get_tasks(self) -> List[Dict]:
        """Retrieve tasks from the external provider."""
        pass

    @abstractmethod
    def complete_task(self, task_id: str) -> None:
        """Mark a task as complete in the external provider."""
        pass
