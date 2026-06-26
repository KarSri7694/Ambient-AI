from typing import Any, List, Optional

import night_mode
from application.ports.task_queue_port import TaskQueuePort
from core.models import NightTask


class SQLiteTaskQueueAdapter(TaskQueuePort):
    """Adapter wrapping night_mode task-queue functions behind TaskQueuePort."""

    def get_pending_tasks(self) -> List[NightTask]:
        rows = night_mode.get_pending_tasks()
        return [
            NightTask(
                id=row["id"],
                description=row["description"],
                priority=row.get("priority", "medium"),
                status=row.get("status", "pending"),
                created_at=row.get("created_at"),
                metadata_json=row.get("meta_data"),
            )
            for row in rows
        ]

    def add_task(
        self,
        description: str,
        priority: str = "medium",
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        return night_mode.add_task(description, priority, metadata=metadata)

    def mark_task_complete(self, task_id: int, status: str = "completed") -> None:
        night_mode.mark_task_complete(task_id, status)
