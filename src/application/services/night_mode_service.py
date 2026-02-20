import asyncio
import logging
from typing import Optional

from application.ports.task_queue_port import TaskQueuePort
from application.ports.task_provider_port import TaskProviderPort
from application.ports.notification_port import NotificationPort
from application.services.llm_interaction_service import LLMInteractionService


class NightModeService:
    """
    Orchestrates autonomous night-time task processing.

    Processes queued night tasks, external provider tasks, and
    monitors notifications — all through injected port abstractions.
    """

    DEFAULT_NIGHT_PROMPT = (
        "You are an autonomous agent working through a list of night-time tasks. "
        "You have access to various tools to help you accomplish these tasks. "
        "DO NOT use the `queue_night_task` tool here."
    )

    SLEEP_INTERVAL = 30  # seconds between notification checks
    MAX_IDLE_CYCLES = 3  # exit after this many consecutive empty notification checks

    def __init__(
        self,
        task_queue: TaskQueuePort,
        task_provider: TaskProviderPort,
        notification_port: NotificationPort,
        llm_service: LLMInteractionService,
        model: str = "",
        username: str = "",
    ):
        self.task_queue = task_queue
        self.task_provider = task_provider
        self.notifications = notification_port
        self.llm_service = llm_service
        self.model = model
        self.logger = logging.getLogger(self.__class__.__name__)

        if username:
            self.night_prompt = (
                f"You are an autonomous agent working through a list of night-time tasks "
                f"queued by {username}. You have access to various tools to help you accomplish "
                f"these tasks. DO NOT use the `queue_night_task` tool here."
            )
        else:
            self.night_prompt = self.DEFAULT_NIGHT_PROMPT

    def _format_notifications(self) -> str:
        """Fetch and format unread notifications into a string."""
        notifications = self.notifications.get_unread_notifications()
        if not notifications:
            return "No new notifications."
        lines = ["New notifications:"]
        for note in notifications:
            lines.append(f"- {note.message} (Source: {note.source})")
        return "\n".join(lines)

    async def run_night_loop(self) -> None:
        """
        Main night-mode loop:
        1. Process queued night tasks
        2. Process external provider tasks (Todoist)
        3. Monitor for new notifications
        4. Exit after MAX_IDLE_CYCLES with no new notifications
        """
        self.logger.info("Late night execution mode started.")
        idle_count = 0

        while True:
            notifications_str = self._format_notifications()

            # ── Process queued night tasks ────────────────────────
            pending = self.task_queue.get_pending_tasks()
            if not pending:
                self.logger.info("No pending tasks found.")
            else:
                for task in pending:
                    self.logger.info(f"Processing night task ID {task.id}: {task.description}")
                    await self.llm_service.run_interaction(
                        user_input=f"{task.description}\n{notifications_str}",
                        system_prompt=self.night_prompt,
                        model=self.model,
                    )
                    self.task_queue.mark_task_complete(task.id)

            # ── Process external provider tasks ───────────────────
            try:
                external_tasks = self.task_provider.get_tasks()
                for ext_task in external_tasks:
                    task_desc = f"Task: {ext_task['content']}, ID: {ext_task['id']}"
                    self.logger.info(f"Processing external task ID {ext_task['id']}: {ext_task['content']}")
                    await self.llm_service.run_interaction(
                        user_input=f"{task_desc}\n{notifications_str}",
                        system_prompt=self.night_prompt,
                        model=self.model,
                    )
                    self.task_provider.complete_task(ext_task["id"])
            except Exception as e:
                self.logger.error(f"Error processing external tasks: {e}")

            # ── Monitor notifications ─────────────────────────────
            notifications_str = self._format_notifications()
            if notifications_str == "No new notifications.":
                idle_count += 1
            else:
                idle_count = 0
                await self.llm_service.run_interaction(
                    user_input=notifications_str,
                    system_prompt=self.night_prompt,
                    model=self.model,
                )

            if idle_count >= self.MAX_IDLE_CYCLES:
                self.logger.info(
                    f"No new notifications for {self.MAX_IDLE_CYCLES} consecutive checks. "
                    "Exiting night mode."
                )
                break

            self.logger.info(f"Sleeping for {self.SLEEP_INTERVAL}s before next check...")
            await asyncio.sleep(self.SLEEP_INTERVAL)
