import logging
from datetime import datetime, timedelta

from application.services.interaction_trace import interaction_trace
from application.services.llm_interaction_service import LLMInteractionService
from core.models import TranscriptClassificationResult


class SimpleTaskExecutionService:
    """Execute bounded low-risk tasks with a small allowed tool set."""

    ALLOWED_TOOL_NAMES = {
        "add_task",
        "schedule_meeting",
        "queue_night_task",
    }

    def __init__(self, llm_service: LLMInteractionService, logger: logging.Logger | None = None):
        self.llm_service = llm_service
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    async def execute(
        self,
        classification: TranscriptClassificationResult,
        transcript_text: str,
        system_prompt: str,
        model: str,
    ) -> str:
        if classification.label not in {"REMINDER", "TASK_SIMPLE"}:
            raise ValueError(f"SimpleTaskExecutionService cannot execute label {classification.label}")

        if classification.label == "REMINDER":
            due_at = (datetime.now() + timedelta(hours=5)).isoformat(timespec="seconds")
            user_input = (
                f"Transcript:\n{transcript_text}\n\n"
                f"Classification label: {classification.label}\n"
                f"Speaker: {classification.speaker_label}\n"
                f"Summary: {classification.summary}\n"
                f"Suggested action: {classification.suggested_action or classification.summary}\n"
                f"Create a reminder/task for this. Use add_task unless schedule_meeting is clearly required. "
                f"If you call add_task, use due_datetime={due_at} unless the transcript gives a clearer date/time."
            )
        else:
            user_input = (
                f"Transcript:\n{transcript_text}\n\n"
                f"Classification label: {classification.label}\n"
                f"Speaker: {classification.speaker_label}\n"
                f"Summary: {classification.summary}\n"
                f"Suggested action: {classification.suggested_action or classification.summary}\n"
                "Complete this as a simple task using only the allowed tools. "
                "If it is not safely doable as a simple task, use queue_night_task."
            )

        with interaction_trace("simple_task_execution"):
            return await self.llm_service.run_interaction(
                user_input=user_input,
                system_prompt=system_prompt,
                model=model,
                allowed_tool_names=self.ALLOWED_TOOL_NAMES,
            )
