import asyncio
import logging

from application.ports.notification_port import NotificationPort
from application.ports.task_provider_port import TaskProviderPort
from application.ports.task_queue_port import TaskQueuePort
from application.services.ambient_reflection_service import AmbientReflectionService
from application.services.llm_interaction_service import LLMInteractionService
from application.services.memory_consolidation_service import MemoryConsolidationService
from application.services.proactive_research_service import ProactiveResearchService


class NightModeService:
    """Orchestrate night tasks, notifications, and memory consolidation."""

    DEFAULT_NIGHT_PROMPT = (
        "You are an autonomous agent working through a list of night-time tasks. "
        "You have access to various tools to help you accomplish these tasks. "
        "DO NOT use the `queue_night_task` tool here."
    )
    DEFAULT_PROACTIVE_RESEARCH_PROMPT = (
        "You are doing bounded proactive research on a topic that surfaced ambiently. "
        "Produce a private research package and do not take outbound actions."
    )

    SLEEP_INTERVAL = 30
    MAX_IDLE_CYCLES = 3

    def __init__(
        self,
        task_queue: TaskQueuePort,
        task_provider: TaskProviderPort,
        notification_port: NotificationPort,
        llm_service: LLMInteractionService,
        memory_consolidator: MemoryConsolidationService | None = None,
        proactive_research_service: ProactiveResearchService | None = None,
        ambient_reflection_service: AmbientReflectionService | None = None,
        model: str = "",
        username: str = "",
    ):
        self.task_queue = task_queue
        self.task_provider = task_provider
        self.notifications = notification_port
        self.llm_service = llm_service
        self.memory_consolidator = memory_consolidator
        self.proactive_research_service = proactive_research_service
        self.ambient_reflection_service = ambient_reflection_service
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
        notifications = self.notifications.get_unread_notifications()
        if not notifications:
            return "No new notifications."
        lines = ["New notifications:"]
        for note in notifications:
            lines.append(f"- {note.message} (Source: {note.source})")
        return "\n".join(lines)

    async def _process_external_tasks(self, notifications_str: str) -> int:
        processed_external_tasks = 0
        try:
            external_tasks = self.task_provider.get_tasks()
            for ext_task in external_tasks:
                task_desc = f"Task: {ext_task['content']}, ID: {ext_task['id']}"
                self.logger.info(
                    "Processing external task ID %s: %s",
                    ext_task["id"],
                    ext_task["content"],
                )
                await self.llm_service.run_interaction(
                    user_input=f"{task_desc}\n{notifications_str}",
                    system_prompt=self.night_prompt,
                    model=self.model,
                )
                self.task_provider.complete_task(ext_task["id"])
                processed_external_tasks += 1
        except Exception as exc:
            self.logger.error("Error processing external tasks: %s", exc)
        return processed_external_tasks

    async def run_external_task_cycle(self) -> int:
        """Process only explicit external tasks, such as the Ambient AI Tasks Todoist project."""
        notifications_str = self._format_notifications()
        return await self._process_external_tasks(notifications_str)

    async def _process_internal_tasks(self, notifications_str: str) -> int:
        processed_internal_tasks = 0
        pending = self.task_queue.get_pending_tasks()
        if not pending:
            self.logger.info("No pending tasks found.")
        else:
            for task in pending:
                self.logger.info("Processing night task ID %s: %s", task.id, task.description)
                await self.llm_service.run_interaction(
                    user_input=f"{task.description}\n{notifications_str}",
                    system_prompt=self.night_prompt,
                    model=self.model,
                )
                self.task_queue.mark_task_complete(task.id)
                processed_internal_tasks += 1
        return processed_internal_tasks

    async def run_night_cycle(self) -> dict:
        """Run one bounded night-mode cycle without sleeping or looping forever."""
        notifications_str = self._format_notifications()
        processed_external_tasks = await self._process_external_tasks(notifications_str)
        processed_internal_tasks = await self._process_internal_tasks(notifications_str)
        processed_research_topics = 0
        reflection_actions = 0
        surfaced_notifications = 0

        if self.memory_consolidator is not None:
            try:
                consolidated = self.memory_consolidator.consolidate()
                if consolidated:
                    self.logger.info("Consolidated %s pending memory events.", consolidated)
            except Exception as exc:
                self.logger.error("Error consolidating memory: %s", exc)

        if self.proactive_research_service is not None:
            try:
                processed_research_topics = await self.proactive_research_service.process_topics(
                    system_prompt=self.DEFAULT_PROACTIVE_RESEARCH_PROMPT,
                    model=self.model,
                    max_topics=2,
                )
                if processed_research_topics:
                    self.logger.info("Processed %s proactive research topics.", processed_research_topics)
            except Exception as exc:
                self.logger.error("Error running proactive research: %s", exc)

        if self.ambient_reflection_service is not None and self.memory_consolidator is not None:
            try:
                reflection_results = await self.ambient_reflection_service.reflect(
                    model=self.model,
                    recent_context=self.memory_consolidator.memory.get_recent_context(),
                )
                reflection_actions = len(reflection_results)
                if reflection_results:
                    self.logger.info(
                        "Ambient reflection produced %s action(s): %s",
                        len(reflection_results),
                        ", ".join(action.action_type for action in reflection_results),
                    )
            except Exception as exc:
                self.logger.error("Error running ambient reflection: %s", exc)

        notifications_str = self._format_notifications()
        if notifications_str != "No new notifications.":
            surfaced_notifications = len(notifications_str.splitlines()) - 1
            await self.llm_service.run_interaction(
                user_input=notifications_str,
                system_prompt=self.night_prompt,
                model=self.model,
            )

        return {
            "processed_internal_tasks": processed_internal_tasks,
            "processed_external_tasks": processed_external_tasks,
            "processed_research_topics": processed_research_topics,
            "reflection_actions": reflection_actions,
            "surfaced_notifications": surfaced_notifications,
        }

    async def run_night_loop(self) -> None:
        self.logger.info("Late night execution mode started.")
        idle_count = 0

        while True:
            cycle_result = await self.run_night_cycle()
            notifications_str = self._format_notifications()
            if notifications_str == "No new notifications.":
                idle_count += 1
            else:
                idle_count = 0
            if any(
                cycle_result[key] > 0
                for key in (
                    "processed_internal_tasks",
                    "processed_external_tasks",
                    "processed_research_topics",
                    "reflection_actions",
                    "surfaced_notifications",
                )
            ):
                idle_count = 0

            if idle_count >= self.MAX_IDLE_CYCLES:
                self.logger.info(
                    "No new notifications for %s consecutive checks. Exiting night mode.",
                    self.MAX_IDLE_CYCLES,
                )
                break

            self.logger.info("Sleeping for %ss before next check...", self.SLEEP_INTERVAL)
            await asyncio.sleep(self.SLEEP_INTERVAL)
