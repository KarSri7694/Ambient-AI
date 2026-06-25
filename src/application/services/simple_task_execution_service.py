import logging
from datetime import datetime, timedelta

from application.services.activity_ledger_service import ActivityLedgerService
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

    def __init__(
        self,
        llm_service: LLMInteractionService,
        activity_ledger: ActivityLedgerService | None = None,
        logger: logging.Logger | None = None,
    ):
        self.llm_service = llm_service
        self.activity_ledger = activity_ledger
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

        run = None
        llm_step = None
        if self.activity_ledger is not None:
            run = self.activity_ledger.start_run(
                source_kind="simple_execution",
                trigger_kind="transcript",
                title=f"Simple execution: {classification.summary[:80]}",
                summary=classification.summary,
                model=model,
                metadata={
                    "classification_label": classification.label,
                    "speaker_label": classification.speaker_label,
                    "suggested_action": classification.suggested_action,
                },
                tags=["simple_execution", classification.label.lower()],
            )
            llm_step = self.activity_ledger.start_step(
                run.run_id,
                step_kind="llm_interaction",
                title="Execute simple task",
                input_ref=classification.summary,
            )

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

        try:
            with interaction_trace(
                "simple_task_execution",
                {"run_id": run.run_id, "step_id": llm_step.step_id} if run and llm_step else None,
            ):
                result = await self.llm_service.run_interaction(
                    user_input=user_input,
                    system_prompt=system_prompt,
                    model=model,
                    allowed_tool_names=self.ALLOWED_TOOL_NAMES,
                )
            if self.activity_ledger is not None and run is not None:
                if llm_step is not None:
                    self.activity_ledger.complete_step(llm_step.step_id, output_ref="assistant_result")
                self.activity_ledger.attach_artifact(
                    run_id=run.run_id,
                    step_id=llm_step.step_id if llm_step else None,
                    artifact_kind="tool_output",
                    title="Simple execution result",
                    text_preview=result[:500],
                    metadata={"allowed_tool_names": sorted(self.ALLOWED_TOOL_NAMES)},
                )
                self.activity_ledger.complete_run(
                    run.run_id,
                    summary=classification.summary,
                    output_text=result,
                )
            return result
        except Exception as exc:
            if self.activity_ledger is not None and run is not None:
                if llm_step is not None:
                    self.activity_ledger.fail_step(llm_step.step_id, error_text=str(exc))
                self.activity_ledger.fail_run(
                    run.run_id,
                    error_text=str(exc),
                    summary=classification.summary,
                )
            raise
