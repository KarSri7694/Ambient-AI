from typing import Any, List, Optional

import night_mode
from application.ports.task_queue_port import TaskQueuePort
from core.models import NightTask


class SQLiteTaskQueueAdapter(TaskQueuePort):
    """Adapter wrapping night_mode task-queue functions behind TaskQueuePort."""

    def get_pending_tasks(self) -> List[NightTask]:
        return self._to_tasks(night_mode.get_pending_tasks())

    def get_due_tasks(self, now_utc: str) -> List[NightTask]:
        return self._to_tasks(night_mode.get_due_tasks(now_utc))

    def get_all_pending_tasks(self) -> List[NightTask]:
        return self._to_tasks(night_mode.get_all_pending_tasks())

    def _to_tasks(self, rows) -> List[NightTask]:
        return [
            NightTask(
                id=row["id"],
                description=row["description"],
                priority=row.get("priority", "medium"),
                status=row.get("status", "pending"),
                created_at=row.get("created_at"),
                metadata_json=row.get("meta_data"),
                run_at_utc=row.get("run_at_utc"),
                claimed_at=row.get("claimed_at"),
                completed_at=row.get("completed_at"),
            )
            for row in rows
        ]

    def add_task(
        self,
        description: str,
        priority: str = "medium",
        metadata: Optional[dict[str, Any]] = None,
        run_at_utc: Optional[str] = None,
    ) -> str:
        return night_mode.add_task(
            description,
            priority,
            metadata=metadata,
            run_at_utc=run_at_utc,
        )

    def mark_task_complete(self, task_id: int, status: str = "completed") -> None:
        night_mode.mark_task_complete(task_id, status)

    def claim_task(self, task_id: int) -> bool:
        return night_mode.claim_task(task_id)

    def cancel_task(self, task_id: int) -> bool:
        return night_mode.cancel_task(task_id)

    def recover_interrupted(self) -> int:
        return night_mode.recover_interrupted_tasks()
