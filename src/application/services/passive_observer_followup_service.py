import json
import logging
import re
from datetime import datetime
from typing import Any, List

from application.ports.LLMProvider import LLMProvider
from application.ports.memory_port import MemoryPort
from application.ports.task_queue_port import TaskQueuePort
from application.services.interaction_trace import interaction_trace
from core.models import VisualObservation


class PassiveObserverFollowupService:
    """Choose one deferred follow-up directly from recent visual observations."""

    DURABLE_ARTIFACT_PATTERN = re.compile(
        r"\b(draft|document|doc|spreadsheet|sheet|notebook|repo|code|pull request|pr|issue|ticket|research|comparison|compare|cart|checkout|shortlist|notes?|email|reply|proposal|slide|presentation)\b",
        re.IGNORECASE,
    )
    EPHEMERAL_ACTION_PATTERN = re.compile(
        r"\b(scroll|swipe|accept|decline|dismiss|close|like|watch|play|pause|resume video|join call|video call|call from|reply now|tap|click|open notification|see more posts|feed)\b",
        re.IGNORECASE,
    )
    DURABLE_APP_HINTS = {
        "gmail",
        "google docs",
        "docs",
        "google sheets",
        "sheets",
        "notion",
        "github",
        "gitlab",
        "visual studio code",
        "vscode",
        "amazon",
        "flipkart",
        "excel",
        "powerpoint",
        "word",
        "chrome",
    }
    FOLLOWUP_TTL_SECONDS = 2 * 60 * 60
    UNSENT_OBSERVATION_LIMIT = 20

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
- Prefer nothing unless there is clear durable work that would still matter later.
- Never queue a task if the observation suggests the work is already completed.
- Do not repeat a task that is already present in pending_tasks.
- Choose at most one task.
- Queue only for durable later work such as unfinished drafts, research, multi-step comparison, code changes, or document review.
- Never queue transient UI actions such as scrolling a feed, accepting a call, clicking a button, or dismissing a popup.
- The description must be self-contained and actionable later, without relying on hidden terminal context.
"""

    def __init__(
        self,
        *,
        memory: MemoryPort,
        task_queue: TaskQueuePort,
        llm_provider: LLMProvider,
        activity_ledger: Any = None,
        logger: logging.Logger | None = None,
    ):
        self.memory = memory
        self.task_queue = task_queue
        self.llm = llm_provider
        self.activity_ledger = activity_ledger
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    async def maybe_queue_followup(self, *, model: str) -> dict:
        observations = self.memory.get_recent_unsent_visual_observations(limit=self.UNSENT_OBSERVATION_LIMIT)
        if not observations:
            return {"action": "nothing", "reason": "no unsent observations"}
        source_observation = observations[0]
        triage = self._triage_observation(source_observation, observations)
        if not triage["eligible"]:
            return {"action": "nothing", "reason": triage["reason"]}
        pending_tasks = self.task_queue.get_pending_tasks()
        payload = {
            "recent_visual_observations": [self._observation_payload(item) for item in observations],
            "selected_observation": self._observation_payload(source_observation),
            "pending_tasks": [task.description for task in pending_tasks[:10]],
            "visual_digest": self.memory.get_visual_digest(),
            "user_info": self.memory.get_user_info(),
            "triage": triage,
        }
        sent_at = datetime.now().isoformat()
        self.memory.mark_visual_observations_followup_sent(
            [item.observation_id for item in observations],
            sent_at=sent_at,
        )
        with interaction_trace("passive_observer_followup"):
            completion = await self.llm.chat_completion_stream(
                model=model,
                messages=[
                    {"role": "system", "content": self.FOLLOWUP_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
                ],
                tools=None,
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
        source_observation_id = str(parsed.get("source_observation_id", "")).strip() or source_observation.observation_id
        source_observation = self._resolve_source_observation(
            source_observation_id=source_observation_id,
            observations=observations,
        )
        metadata = self._build_task_metadata(
            observation=source_observation,
            title=title,
            description=description,
            triage=triage,
        )
        self.task_queue.add_task(
            self._build_task_description(
                title=title,
                description=description,
                observation=source_observation,
            ),
            priority="low",
            metadata=metadata,
        )
        if self.activity_ledger is not None:
            run = self.activity_ledger.queue_run(
                source_kind="passive_observer_followup",
                trigger_kind="ambient_inference",
                title=title,
                summary=description,
                metadata={"source_observation_id": source_observation_id},
                tags=["passive_observer", "followup"],
            )
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
            "source_observation_id": source_observation_id,
        }

    def _resolve_source_observation(
        self,
        *,
        source_observation_id: str,
        observations: List[VisualObservation],
    ) -> VisualObservation | None:
        if source_observation_id:
            for observation in observations:
                if observation.observation_id == source_observation_id:
                    return observation
        return observations[0] if observations else None

    def _build_task_description(
        self,
        *,
        title: str,
        description: str,
        observation: VisualObservation | None,
    ) -> str:
        lines = [f"[Passive observer] {title}", description.strip()]
        if observation is None:
            return "\n".join(lines)
        lines.extend(
            [
                "",
                "Passive observation context:",
                f"- Observation ID: {observation.observation_id}",
                f"- Captured at: {observation.created_at}",
            ]
        )
        if observation.app_name:
            lines.append(f"- App: {observation.app_name}")
        if observation.page_hint:
            lines.append(f"- Page hint: {observation.page_hint}")
        if observation.summary:
            lines.append(f"- Summary: {observation.summary}")
        if observation.detailed_description:
            lines.append(f"- Detailed description: {observation.detailed_description}")
        if observation.inferred_user_activity:
            lines.append(f"- Inferred user activity: {observation.inferred_user_activity}")
        if observation.possible_next_task:
            lines.append(f"- Possible next task: {observation.possible_next_task}")
        if observation.open_loops:
            lines.append(f"- Open loops: {', '.join(observation.open_loops[:4])}")
        return "\n".join(lines)

    def _build_task_metadata(
        self,
        *,
        observation: VisualObservation | None,
        title: str,
        description: str,
        triage: dict,
    ) -> dict:
        metadata = {
            "task_kind": "passive_observer_followup",
            "title": title,
            "description": description,
            "durability_class": triage.get("durability_class", "durable"),
            "queue_reason": triage.get("reason", ""),
            "ttl_seconds": self.FOLLOWUP_TTL_SECONDS,
        }
        if observation is None:
            return metadata
        metadata.update(
            {
                "source_observation_id": observation.observation_id,
                "source_created_at": observation.created_at,
                "app_name": observation.app_name,
                "page_hint": observation.page_hint,
                "summary": observation.summary,
                "detailed_description": observation.detailed_description,
                "inferred_user_activity": observation.inferred_user_activity,
                "previous_activity_status": observation.previous_activity_status,
                "completed_items": observation.completed_items,
                "open_loops": observation.open_loops,
                "possible_next_task": observation.possible_next_task,
                "session_id": observation.session_id,
            }
        )
        return metadata

    def _observation_payload(self, item: VisualObservation) -> dict:
        return {
            "observation_id": item.observation_id,
            "created_at": item.created_at,
            "app_name": item.app_name,
            "page_hint": item.page_hint,
            "summary": item.summary,
            "detailed_description": item.detailed_description,
            "inferred_user_activity": item.inferred_user_activity,
            "previous_activity_status": item.previous_activity_status,
            "completed_items": item.completed_items,
            "open_loops": item.open_loops,
            "possible_next_task": item.possible_next_task,
        }

    def _triage_observation(
        self,
        observation: VisualObservation,
        observations: List[VisualObservation],
    ) -> dict:
        text = " ".join(
            part
            for part in [
                observation.summary,
                observation.detailed_description,
                observation.inferred_user_activity,
                observation.possible_next_task or "",
                " ".join(observation.open_loops),
            ]
            if part
        )
        lowered = text.lower()
        if observation.previous_activity_status == "completed" or observation.completed_items:
            return {"eligible": False, "reason": "observation looks completed"}
        if self.EPHEMERAL_ACTION_PATTERN.search(lowered):
            return {"eligible": False, "reason": "ephemeral immediate UI action"}
        if not observation.open_loops and not observation.possible_next_task:
            return {"eligible": False, "reason": "no unresolved durable work"}

        durable_artifact = bool(self.DURABLE_ARTIFACT_PATTERN.search(lowered)) or self._durable_app_context(observation)
        repeated = self._has_repeated_unresolved_intent(observation, observations)

        if not durable_artifact:
            return {"eligible": False, "reason": "no durable artifact or workflow"}
        if observation.previous_activity_status == "left_midway":
            return {
                "eligible": True,
                "reason": "durable artifact left midway",
                "durability_class": "durable",
                "repeated": repeated,
            }
        if repeated:
            return {
                "eligible": True,
                "reason": "durable unresolved work repeated across observations",
                "durability_class": "durable",
                "repeated": True,
            }
        if self._looks_like_draft_or_comparison(observation):
            return {
                "eligible": True,
                "reason": "durable draft or comparison artifact visible",
                "durability_class": "durable",
                "repeated": repeated,
            }
        return {"eligible": False, "reason": "single observation is not durable enough"}

    def _durable_app_context(self, observation: VisualObservation) -> bool:
        values = {
            (observation.app_name or "").strip().lower(),
            (observation.page_hint or "").strip().lower(),
        }
        return any(value in self.DURABLE_APP_HINTS for value in values if value)

    def _looks_like_draft_or_comparison(self, observation: VisualObservation) -> bool:
        text = " ".join(
            part
            for part in [
                observation.summary,
                observation.detailed_description,
                observation.inferred_user_activity,
                observation.possible_next_task or "",
            ]
            if part
        ).lower()
        return any(
            token in text
            for token in (
                "draft",
                "reply",
                "compare",
                "comparison",
                "research",
                "review",
                "cart",
                "code",
                "document",
                "email",
            )
        )

    def _has_repeated_unresolved_intent(
        self,
        observation: VisualObservation,
        observations: List[VisualObservation],
    ) -> bool:
        if not observation.open_loops and not observation.possible_next_task:
            return False
        current_terms = self._intent_terms(observation)
        if not current_terms:
            return False
        matches = 0
        for item in observations[1:]:
            if item.previous_activity_status == "completed":
                continue
            if observation.session_id and item.session_id and item.session_id == observation.session_id:
                matches += 1
                continue
            other_terms = self._intent_terms(item)
            if other_terms and len(current_terms & other_terms) >= 2:
                matches += 1
        return matches >= 1

    def _intent_terms(self, observation: VisualObservation) -> set[str]:
        text = " ".join(
            part
            for part in [
                observation.summary,
                observation.detailed_description,
                observation.inferred_user_activity,
                observation.possible_next_task or "",
                " ".join(observation.open_loops),
            ]
            if part
        ).lower()
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text)
            if len(token) >= 4 and token not in {"that", "with", "from", "this", "user"}
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
