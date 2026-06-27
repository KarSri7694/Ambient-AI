from typing import List, Dict

from application.ports.task_provider_port import TaskProviderPort
from utils.todoist_helper import TodoistHelper


class TodoistTaskAdapter(TaskProviderPort):
    """Adapter wrapping TodoistHelper behind TaskProviderPort."""

    def __init__(self):
        self._helper = TodoistHelper()

    def get_tasks(self) -> List[Dict]:
        return self._helper.get_tasks()

    def complete_task(self, task_id: str) -> None:
        self._helper.complete_task(task_id)
