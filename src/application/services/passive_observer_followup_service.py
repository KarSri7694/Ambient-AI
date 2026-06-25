import json
import logging
from typing import List

from application.ports.LLMProvider import LLMProvider
from application.ports.memory_port import MemoryPort
from application.ports.task_queue_port import TaskQueuePort
from application.services.activity_ledger_service import ActivityLedgerService
from application.services.interaction_trace import interaction_trace
from core.models import VisualObservation


class PassiveObserverFollowupService:
    """Choose one deferred follow-up directly from recent visual observations."""

    FOLLOWUP_PROMPT = """You decide whether the ambient agent should take one deferred follow-up action from recent passive visual observations.

Return JSON only:
{
  "action": "nothing|queue_task",
  "title": "short title",
  "description": "short concrete task description",
  "source_observation_id": "observation id or empty string",
  "confidence": 0.0
}

Rules:
- Prefer nothing unless there is a clear unfinished user activity.
- Never queue a task if the observation suggests the work is already completed.
- Do not repeat a task that is already present in pending_tasks.
- Choose at most one task.
- Keep the task subtle and useful, based on what the user was already doing.
"""

    def __init__(
        self,
        *,
        memory: MemoryPort,
        task_queue: TaskQueuePort,
        llm_provider: LLMProvider,
        activity_ledger: ActivityLedgerService | None = None,
        logger: logging.Logger | None = None,
    ):
        self.memory = memory
        self.task_queue = task_queue
        self.llm = llm_provider
        self.activity_ledger = activity_ledger
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    async def maybe_queue_followup(self, *, model: str) -> dict:
        observations = self.memory.get_recent_visual_observations(limit=5)
        if not observations:
            return {"action": "nothing"}
        pending_tasks = self.task_queue.get_pending_tasks()
        payload = {
            "recent_visual_observations": [self._observation_payload(item) for item in observations],
            "pending_tasks": [task.description for task in pending_tasks[:10]],
            "visual_digest": self.memory.get_visual_digest(),
            "user_info": self.memory.get_user_info(),
        }
        with interaction_trace("passive_observer_followup"):
            completion = await self.llm.chat_completion_stream(
                model=model,
                messages=[
                    {"role": "system", "content": self.FOLLOWUP_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
                ],
                tools=None,
                temperature=0.1,
                top_p=0.9,
            )
        text = await self._consume_stream_text(completion)
        parsed = self._parse_json_object(text)
        if not isinstance(parsed, dict):
            return {"action": "nothing"}
        if str(parsed.get("action", "")).strip().lower() != "queue_task":
            return {"action": "nothing"}
        title = str(parsed.get("title", "")).strip()
        description = str(parsed.get("description", "")).strip()
        if not title or not description:
            return {"action": "nothing"}
        if self._looks_duplicate(title=title, description=description, pending_tasks=payload["pending_tasks"]):
            return {"action": "nothing"}
        self.task_queue.add_task(
            f"[Passive observer] {title}\n{description}",
            priority="low",
        )
        if self.activity_ledger is not None:
            run = self.activity_ledger.queue_run(
                source_kind="passive_observer_followup",
                trigger_kind="ambient_inference",
                title=title,
                summary=description,
                metadata={"source_observation_id": str(parsed.get("source_observation_id", "")).strip()},
                tags=["passive_observer", "followup"],
            )
            source_observation_id = str(parsed.get("source_observation_id", "")).strip()
            if source_observation_id:
                self.activity_ledger.link_entity(
                    run_id=run.run_id,
                    entity_type="visual_observation",
                    entity_id=source_observation_id,
                    relation="derived_from",
                )
        return {
            "action": "queue_task",
            "title": title,
            "description": description,
            "source_observation_id": str(parsed.get("source_observation_id", "")).strip(),
        }

    def _observation_payload(self, item: VisualObservation) -> dict:
        return {
            "observation_id": item.observation_id,
            "created_at": item.created_at,
            "app_name": item.app_name,
            "page_hint": item.page_hint,
            "summary": item.summary,
            "inferred_user_activity": item.inferred_user_activity,
            "previous_activity_status": item.previous_activity_status,
            "completed_items": item.completed_items,
            "open_loops": item.open_loops,
            "possible_next_task": item.possible_next_task,
        }

    def _looks_duplicate(self, *, title: str, description: str, pending_tasks: List[str]) -> bool:
        normalized_title = title.lower()
        normalized_description = description.lower()
        for task in pending_tasks:
            text = task.lower()
            if normalized_title in text or normalized_description in text:
                return True
        return False

    async def _consume_stream_text(self, completion) -> str:
        parts: List[str] = []
        async for chunk in completion:
            delta = chunk.choices[0].delta
            if delta.content:
                parts.append(delta.content)
        return "".join(parts)

    def _parse_json_object(self, response_text: str) -> dict:
        text = response_text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            self.logger.warning("Passive observer follow-up response was not valid JSON.")
            return {}
