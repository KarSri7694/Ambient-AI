import asyncio
import json
import logging

from application.ports.notification_port import NotificationPort
from application.ports.task_provider_port import TaskProviderPort
from application.ports.task_queue_port import TaskQueuePort
from application.services.activity_ledger_service import ActivityLedgerService
from application.services.interaction_trace import interaction_trace
from application.services.ambient_reflection_service import AmbientReflectionService
from application.services.llm_interaction_service import LLMInteractionService
from application.services.memory_consolidation_service import MemoryConsolidationService
from application.services.proactive_research_service import ProactiveResearchService


class NightModeService:
    """Orchestrate night tasks, notifications, and memory consolidation."""

    DEFAULT_NIGHT_PROMPT = (
        "You are an autonomous agent working through a list of night-time tasks. "
        "You have access to various tools to help you accomplish these tasks. "
        "DO NOT use the `queue_night_task` tool here. "
        "Treat the user message as an execution brief, not a question. "
        "When the task references a screenshot, file path, task ID, URL, or other concrete artifact, use that artifact directly. "
        "If notifications say 'No new notifications.', ignore that line and continue the task. "
        "Do the work first; do not ask what to do unless the task is genuinely missing the required artifact."
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
        activity_ledger: ActivityLedgerService | None = None,
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
        self.activity_ledger = activity_ledger
        self.model = model
        self.logger = logging.getLogger(self.__class__.__name__)

        if username:
            self.night_prompt = (
                f"You are an autonomous agent working through a list of night-time tasks "
                f"queued by {username}. You have access to various tools to help you accomplish "
                f"these tasks. DO NOT use the `queue_night_task` tool here. "
                f"Treat the user message as an execution brief, not a question. "
                f"When the task references a screenshot, file path, task ID, URL, or other concrete artifact, use that artifact directly. "
                f"If notifications say 'No new notifications.', ignore that line and continue the task. "
                f"Do the work first; do not ask what to do unless the task is genuinely missing the required artifact."
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

    def _build_execution_input(self, task_desc: str, notifications_str: str) -> str:
        return "\n".join(
            [
                "Execute this night task now.",
                "",
                "Task brief:",
                task_desc,
                "",
                "Execution rules:",
                "- Use the task brief as the source of truth.",
                "- If the brief contains a screenshot path or file path, inspect that artifact directly.",
                "- If the brief references prior terminal output, rely only on the concrete details copied into the brief.",
                "- Ignore the notifications section when it says there are no new notifications.",
                "",
                "Notifications:",
                notifications_str,
            ]
        )

    def _task_metadata(self, task) -> dict:
        raw = getattr(task, "metadata_json", None)
        if not raw:
            return {}
        if isinstance(raw, dict):
            return raw
        try:
            parsed = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _build_passive_observer_execution_input(self, task, metadata: dict, notifications_str: str) -> str:
        lines = [
            "Execute this passive-observer night task now.",
            "",
            "Queued task:",
            task.description,
            "",
            "Resolved passive observation context:",
            f"- Observation ID: {metadata.get('source_observation_id', '')}",
            f"- Captured at: {metadata.get('source_created_at', '')}",
            f"- App: {metadata.get('app_name', '')}",
            f"- Page hint: {metadata.get('page_hint', '')}",
            f"- Summary: {metadata.get('summary', '')}",
            f"- Detailed description: {metadata.get('detailed_description', '')}",
            f"- Inferred user activity: {metadata.get('inferred_user_activity', '')}",
            f"- Open loops: {', '.join(metadata.get('open_loops', [])[:4]) if isinstance(metadata.get('open_loops'), list) else ''}",
            f"- Possible next task: {metadata.get('possible_next_task', '')}",
            "",
            "Execution rules:",
            "- Treat this as durable deferred work only if it still appears unresolved now.",
            "- If the context looks completed, expired, or no longer actionable, return a brief skip result instead of acting.",
            "- Ignore the notifications section when it says there are no new notifications.",
            "",
            "Notifications:",
            notifications_str,
        ]
        return "\n".join(lines)

    async def _process_external_tasks(self, notifications_str: str, parent_run_id: str | None = None) -> int:
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
                run = None
                llm_step = None
                todoist_step = None
                try:
                    if self.activity_ledger is not None:
                        run = self.activity_ledger.start_run(
                            source_kind="ambient_ai_task",
                            trigger_kind="explicit_user_task",
                            title=ext_task["content"],
                            summary=task_desc,
                            model=self.model,
                            parent_run_id=parent_run_id,
                            metadata={"external_task_id": ext_task["id"]},
                            tags=["todoist", "ambient_ai_task"],
                        )
                        self.activity_ledger.link_entity(
                            run_id=run.run_id,
                            entity_type="todoist_task",
                            entity_id=str(ext_task["id"]),
                            relation="executes",
                        )
                        llm_step = self.activity_ledger.start_step(
                            run.run_id,
                            step_kind="llm_interaction",
                            title="Execute external task",
                            input_ref=ext_task["content"],
                        )
                    with interaction_trace(
                        "night_mode_external_task",
                        {"run_id": run.run_id, "step_id": llm_step.step_id} if run and llm_step else None,
                    ):
                        result = await self.llm_service.run_interaction(
                            user_input=self._build_execution_input(task_desc, notifications_str),
                            system_prompt=self.night_prompt,
                            model=self.model,
                        )
                    result_text = str(result)
                    if self.activity_ledger is not None and run is not None:
                        if llm_step is not None:
                            self.activity_ledger.complete_step(llm_step.step_id, output_ref="assistant_result")
                        self.activity_ledger.attach_artifact(
                            run_id=run.run_id,
                            step_id=llm_step.step_id if llm_step else None,
                            artifact_kind="tool_output",
                            title="Ambient AI task result",
                            text_preview=result_text[:500],
                        )
                        todoist_step = self.activity_ledger.start_step(
                            run.run_id,
                            step_kind="task_provider_completion",
                            title="Complete Todoist task",
                            input_ref=str(ext_task["id"]),
                        )
                    self.task_provider.complete_task(ext_task["id"])
                    if self.activity_ledger is not None and run is not None:
                        if todoist_step is not None:
                            self.activity_ledger.complete_step(todoist_step.step_id, output_ref="todoist_completed")
                        self.activity_ledger.complete_run(
                            run.run_id,
                            summary=task_desc,
                            output_text=result_text,
                        )
                    processed_external_tasks += 1
                except Exception as exc:
                    if self.activity_ledger is not None and run is not None:
                        if llm_step is not None:
                            self.activity_ledger.fail_step(llm_step.step_id, error_text=str(exc))
                        if todoist_step is not None:
                            self.activity_ledger.fail_step(todoist_step.step_id, error_text=str(exc))
                        self.activity_ledger.fail_run(
                            run.run_id,
                            error_text=str(exc),
                            summary=task_desc,
                        )
                    raise
        except Exception as exc:
            self.logger.error("Error processing external tasks: %s", exc)
        return processed_external_tasks

    async def run_external_task_cycle(self) -> int:
        """Process only explicit external tasks, such as the Ambient AI Tasks Todoist project."""
        notifications_str = self._format_notifications()
        return await self._process_external_tasks(notifications_str)

    async def _process_internal_tasks(self, notifications_str: str, parent_run_id: str | None = None) -> int:
        processed_internal_tasks = 0
        pending = self.task_queue.get_pending_tasks()
        if not pending:
            self.logger.info("No pending tasks found.")
        else:
            for task in pending:
                self.logger.info("Processing night task ID %s: %s", task.id, task.description)
                run = None
                llm_step = None
                queue_step = None
                metadata = self._task_metadata(task)
                try:
                    if self.activity_ledger is not None:
                        run = self.activity_ledger.start_run(
                            source_kind="night_mode",
                            trigger_kind="night_cycle",
                            title=task.description.splitlines()[0][:120],
                            summary=task.description,
                            model=self.model,
                            parent_run_id=parent_run_id,
                            metadata={"night_task_id": task.id},
                            tags=["night_mode", "queued_task"],
                        )
                        self.activity_ledger.link_entity(
                            run_id=run.run_id,
                            entity_type="night_task",
                            entity_id=str(task.id),
                            relation="executes",
                        )
                    execution_input = (
                        self._build_passive_observer_execution_input(task, metadata, notifications_str)
                        if metadata.get("task_kind") == "passive_observer_followup"
                        else self._build_execution_input(task.description, notifications_str)
                    )
                    if self.activity_ledger is not None and run is not None:
                        llm_step = self.activity_ledger.start_step(
                            run.run_id,
                            step_kind="llm_interaction",
                            title="Execute queued night task",
                            input_ref=task.description,
                        )
                    with interaction_trace(
                        "night_mode_internal_task",
                        {"run_id": run.run_id, "step_id": llm_step.step_id} if run and llm_step else None,
                    ):
                        result = await self.llm_service.run_interaction(
                            user_input=execution_input,
                            system_prompt=self.night_prompt,
                            model=self.model,
                        )
                    result_text = str(result)
                    if self.activity_ledger is not None and run is not None:
                        if llm_step is not None:
                            self.activity_ledger.complete_step(llm_step.step_id, output_ref="assistant_result")
                        queue_step = self.activity_ledger.start_step(
                            run.run_id,
                            step_kind="task_queue_completion",
                            title="Mark queued task complete",
                            input_ref=str(task.id),
                        )
                    self.task_queue.mark_task_complete(task.id)
                    if self.activity_ledger is not None and run is not None:
                        if queue_step is not None:
                            self.activity_ledger.complete_step(queue_step.step_id, output_ref="task_completed")
                        self.activity_ledger.complete_run(
                            run.run_id,
                            summary=task.description.splitlines()[0][:160],
                            output_text=result_text,
                        )
                    processed_internal_tasks += 1
                except Exception as exc:
                    if self.activity_ledger is not None and run is not None:
                        if llm_step is not None:
                            self.activity_ledger.fail_step(llm_step.step_id, error_text=str(exc))
                        if queue_step is not None:
                            self.activity_ledger.fail_step(queue_step.step_id, error_text=str(exc))
                        self.activity_ledger.fail_run(run.run_id, error_text=str(exc), summary=task.description[:200])
                    raise
        return processed_internal_tasks

    async def run_night_cycle(self) -> dict:
        """Run one bounded night-mode cycle without sleeping or looping forever."""
        notifications_str = self._format_notifications()
        cycle_run = None
        if self.activity_ledger is not None:
            cycle_run = self.activity_ledger.start_run(
                source_kind="night_mode",
                trigger_kind="night_cycle",
                title="Night mode cycle",
                summary="Bounded night-mode orchestration cycle.",
                model=self.model,
                tags=["night_mode", "cycle"],
            )
        processed_external_tasks = await self._process_external_tasks(
            notifications_str,
            parent_run_id=cycle_run.run_id if cycle_run else None,
        )
        processed_internal_tasks = await self._process_internal_tasks(
            notifications_str,
            parent_run_id=cycle_run.run_id if cycle_run else None,
        )
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

        if cycle_run is not None:
            self.activity_ledger.complete_run(
                cycle_run.run_id,
                summary=(
                    f"Night cycle processed {processed_external_tasks} external task(s), "
                    f"{processed_internal_tasks} internal task(s), "
                    f"{processed_research_topics} research topic(s), "
                    f"and {reflection_actions} reflection action(s)."
                ),
                output_text="Night mode cycle completed.",
                metadata={"surfaced_notifications": surfaced_notifications},
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
