from datetime import datetime, timezone
from typing import Any, Optional

from application.ports.task_queue_port import TaskQueuePort


class ScheduledTaskService:
    """Validate and persist exact-time tasks in the shared task queue."""

    VALID_PRIORITIES = {"low", "medium", "high"}

    def __init__(self, task_queue: TaskQueuePort):
        self.task_queue = task_queue

    @staticmethod
    def normalize_run_at(run_at: str) -> str:
        raw = str(run_at or "").strip()
        if not raw:
            raise ValueError("run_at is required and must be an ISO 8601 date-time.")
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise ValueError("run_at must be a valid ISO 8601 date-time.") from exc
        if parsed.tzinfo is None:
            parsed = parsed.astimezone()
        return parsed.astimezone(timezone.utc).isoformat()

    def schedule(
        self,
        *,
        task: str,
        run_at: str,
        priority: str = "medium",
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        normalized_task = str(task or "").strip()
        if not normalized_task:
            raise ValueError("task is required.")
        normalized_priority = str(priority or "medium").strip().lower()
        if normalized_priority not in self.VALID_PRIORITIES:
            raise ValueError("priority must be low, medium, or high.")
        run_at_utc = self.normalize_run_at(run_at)
        if datetime.fromisoformat(run_at_utc) <= datetime.now(timezone.utc):
            raise ValueError("run_at must be in the future; execute overdue work directly.")

        task_metadata = dict(metadata or {})
        task_metadata.update(
            {
                "task_kind": "scheduled_exact_time",
                "source": task_metadata.get("source", "scheduled_task_tool"),
                "run_at_utc": run_at_utc,
            }
        )
        result = self.task_queue.add_task(
            normalized_task,
            normalized_priority,
            metadata=task_metadata,
            run_at_utc=run_at_utc,
        )
        return {
            "status": "scheduled",
            "task": normalized_task,
            "run_at_utc": run_at_utc,
            "priority": normalized_priority,
            "message": result,
        }
