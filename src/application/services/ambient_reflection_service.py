import json
import logging
import uuid
from dataclasses import replace
from datetime import datetime
from typing import Iterable, List

from application.ports.ambient_agenda_port import AmbientAgendaPort
from application.ports.memory_port import MemoryPort
from application.ports.notification_port import NotificationPort
from application.ports.proactive_topic_queue_port import ProactiveTopicQueuePort
from application.ports.task_queue_port import TaskQueuePort
from application.services.agenda_scoring_service import AgendaScoringService, ReflectionCandidate
from application.services.interaction_trace import interaction_trace
from application.services.llm_interaction_service import LLMInteractionService
from core.models import AmbientAgendaItem, AmbientReflectionAction


class AmbientReflectionService:
    """Run a bounded ambient reflection pass and apply small agenda mutations."""

    VALID_ACTIONS = {
        "NOTHING",
        "CREATE_AGENDA_ITEM",
        "UPDATE_AGENDA_ITEM",
        "SURFACE_ITEM",
        "CREATE_SIMPLE_TASK",
        "QUEUE_COMPLEX_TASK",
        "DISMISS_ITEM",
    }
    MAX_SURFACES = 1
    MAX_ACTIONABLE_CREATIONS = 1

    REFLECTION_PROMPT = """You are the ambient reflection layer for a personal agent.

Return JSON only with this shape:
{
  "actions": [
    {"action_type": "ACTION_NAME", "payload": {...}}
  ]
}

Allowed action_type values:
- NOTHING
- CREATE_AGENDA_ITEM
- UPDATE_AGENDA_ITEM
- SURFACE_ITEM
- CREATE_SIMPLE_TASK
- QUEUE_COMPLEX_TASK
- DISMISS_ITEM

Rules:
- Closed set only. Do not invent actions.
- At most one SURFACE_ITEM.
- At most one actionable creation total across CREATE_SIMPLE_TASK and QUEUE_COMPLEX_TASK.
- Prefer subtle surfacing over action.
- Do not execute tools or assume tools exist.
- Use candidate_id from the provided candidates when possible.
- CREATE_AGENDA_ITEM payload should include: candidate_id, title, kind, source_type, source_ref, priority_score.
- UPDATE_AGENDA_ITEM payload should include: agenda_id and any changed fields.
- SURFACE_ITEM payload should include: agenda_id or candidate_id, and message.
- CREATE_SIMPLE_TASK and QUEUE_COMPLEX_TASK payload should include: title and description.
- DISMISS_ITEM payload should include: agenda_id or candidate_id.
"""

    def __init__(
        self,
        llm_service: LLMInteractionService,
        memory: MemoryPort,
        agenda: AmbientAgendaPort,
        notifications: NotificationPort,
        task_queue: TaskQueuePort,
        topic_queue: ProactiveTopicQueuePort,
        scorer: AgendaScoringService | None = None,
        logger: logging.Logger | None = None,
    ):
        self.llm_service = llm_service
        self.memory = memory
        self.agenda = agenda
        self.notifications = notifications
        self.task_queue = task_queue
        self.topic_queue = topic_queue
        self.scorer = scorer or AgendaScoringService(agenda=agenda)
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    async def reflect(self, model: str, recent_context: str) -> List[AmbientReflectionAction]:
        facts = self._all_facts()
        open_items = self.agenda.list_items(statuses=["open", "surfaced"], limit=50)
        topics = self.topic_queue.list_topics(statuses=["pending", "completed"], limit=50)
        pending_tasks = self.task_queue.get_pending_tasks()
        unread_notifications = self.notifications.peek_unread_notifications()

        candidates = self.scorer.build_top_candidates(
            facts=facts,
            topics=topics,
            open_agenda_items=open_items,
            pending_tasks=pending_tasks,
            unread_notifications=unread_notifications,
            limit=8,
        )
        if not candidates:
            return [AmbientReflectionAction(action_type="NOTHING", payload={})]

        parsed_actions = await self._request_actions(
            model=model,
            recent_context=recent_context,
            candidates=candidates,
        )
        return self._apply_actions(parsed_actions, candidates)

    async def _request_actions(
        self,
        *,
        model: str,
        recent_context: str,
        candidates: List[ReflectionCandidate],
    ) -> List[AmbientReflectionAction]:
        user_input = json.dumps(
            {
                "recent_context": recent_context,
                "open_agenda_count": len(self.agenda.list_items(statuses=["open", "surfaced"], limit=200)),
                "candidates": [candidate.__dict__ for candidate in candidates],
            },
            indent=2,
        )
        with interaction_trace("ambient_reflection"):
            response = await self.llm_service.run_interaction(
                user_input=user_input,
                system_prompt=self.REFLECTION_PROMPT,
                model=model,
                allowed_tool_names=set(),
            )
        return self.parse_actions(response)

    def parse_actions(self, response: str) -> List[AmbientReflectionAction]:
        text = response.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            self.logger.warning("Reflection response was not valid JSON.")
            return [AmbientReflectionAction(action_type="NOTHING", payload={})]

        raw_actions = parsed.get("actions", [])
        if not isinstance(raw_actions, list):
            return [AmbientReflectionAction(action_type="NOTHING", payload={})]

        validated: List[AmbientReflectionAction] = []
        surface_count = 0
        creation_count = 0
        for raw_action in raw_actions:
            if not isinstance(raw_action, dict):
                continue
            action_type = str(raw_action.get("action_type", "")).strip()
            if action_type not in self.VALID_ACTIONS:
                continue
            if action_type == "SURFACE_ITEM":
                if surface_count >= self.MAX_SURFACES:
                    continue
                surface_count += 1
            if action_type in {"CREATE_SIMPLE_TASK", "QUEUE_COMPLEX_TASK"}:
                if creation_count >= self.MAX_ACTIONABLE_CREATIONS:
                    continue
                creation_count += 1
            payload = raw_action.get("payload", {})
            if not isinstance(payload, dict):
                payload = {}
            validated.append(AmbientReflectionAction(action_type=action_type, payload=payload))

        return validated or [AmbientReflectionAction(action_type="NOTHING", payload={})]

    def _apply_actions(
        self,
        actions: Iterable[AmbientReflectionAction],
        candidates: List[ReflectionCandidate],
    ) -> List[AmbientReflectionAction]:
        candidate_lookup = {candidate.candidate_id: candidate for candidate in candidates}
        applied: List[AmbientReflectionAction] = []
        now = datetime.now().isoformat()

        for action in actions:
            payload = dict(action.payload)
            if action.action_type == "NOTHING":
                applied.append(action)
                continue

            if action.action_type == "CREATE_AGENDA_ITEM":
                candidate = candidate_lookup.get(str(payload.get("candidate_id", "")).strip())
                if candidate is None:
                    continue
                existing = self.agenda.find_by_source(
                    source_type=candidate.source_type,
                    source_ref=candidate.source_ref,
                    kind=candidate.proposed_kind,
                    statuses=["open", "surfaced", "completed"],
                )
                if existing is not None:
                    continue
                item = AmbientAgendaItem(
                    agenda_id=uuid.uuid4().hex,
                    title=str(payload.get("title") or candidate.title),
                    kind=str(payload.get("kind") or candidate.proposed_kind),
                    source_type=str(payload.get("source_type") or candidate.source_type),
                    source_ref=str(payload.get("source_ref") or candidate.source_ref),
                    priority_score=float(payload.get("priority_score", candidate.score)),
                    status="open",
                    created_at=now,
                    updated_at=now,
                    due_at=payload.get("due_at") or candidate.due_at,
                    last_considered_at=now,
                    backing_topic_id=payload.get("backing_topic_id") or candidate.backing_topic_id,
                    backing_memory_ids=list(payload.get("backing_memory_ids") or candidate.backing_memory_ids),
                    surface_message=str(payload.get("surface_message") or candidate.summary),
                )
                self.agenda.create_item(item)
                applied.append(replace(action, payload={"agenda_id": item.agenda_id, **payload}))
                continue

            if action.action_type == "UPDATE_AGENDA_ITEM":
                agenda_id = str(payload.get("agenda_id", "")).strip()
                item = self.agenda.get_item(agenda_id)
                if item is None:
                    continue
                updated = AmbientAgendaItem(
                    agenda_id=item.agenda_id,
                    title=str(payload.get("title", item.title)),
                    kind=str(payload.get("kind", item.kind)),
                    source_type=item.source_type,
                    source_ref=item.source_ref,
                    priority_score=float(payload.get("priority_score", item.priority_score)),
                    status=str(payload.get("status", item.status)),
                    created_at=item.created_at,
                    updated_at=now,
                    due_at=payload.get("due_at", item.due_at),
                    last_considered_at=now,
                    backing_topic_id=payload.get("backing_topic_id", item.backing_topic_id),
                    backing_memory_ids=list(payload.get("backing_memory_ids", item.backing_memory_ids)),
                    surface_message=payload.get("surface_message", item.surface_message),
                )
                self.agenda.update_item(updated)
                applied.append(action)
                continue

            if action.action_type == "SURFACE_ITEM":
                item = self._resolve_agenda_from_payload(payload, candidate_lookup)
                if item is None:
                    continue
                if item.status == "surfaced":
                    last_seen = self._parse_dt(item.updated_at)
                    if last_seen is not None and (datetime.now() - last_seen).total_seconds() < 86400:
                        continue
                message = str(payload.get("message") or item.surface_message or item.title).strip()
                if not message:
                    continue
                self.notifications.add_notification(message, source="ambient_digest")
                surfaced = AmbientAgendaItem(
                    agenda_id=item.agenda_id,
                    title=item.title,
                    kind=item.kind,
                    source_type=item.source_type,
                    source_ref=item.source_ref,
                    priority_score=item.priority_score,
                    status="surfaced",
                    created_at=item.created_at,
                    updated_at=now,
                    due_at=item.due_at,
                    last_considered_at=now,
                    backing_topic_id=item.backing_topic_id,
                    backing_memory_ids=item.backing_memory_ids,
                    surface_message=message,
                )
                self.agenda.update_item(surfaced)
                applied.append(replace(action, payload={"agenda_id": item.agenda_id, "message": message}))
                continue

            if action.action_type == "CREATE_SIMPLE_TASK":
                title = str(payload.get("title", "")).strip()
                description = str(payload.get("description", "")).strip()
                if not title or not description:
                    continue
                self.task_queue.add_task(f"[Ambient simple follow-through] {title}\n{description}", priority="low")
                applied.append(action)
                continue

            if action.action_type == "QUEUE_COMPLEX_TASK":
                title = str(payload.get("title", "")).strip()
                description = str(payload.get("description", "")).strip()
                if not title or not description:
                    continue
                self.task_queue.add_task(f"[Ambient queued task] {title}\n{description}", priority="medium")
                applied.append(action)
                continue

            if action.action_type == "DISMISS_ITEM":
                item = self._resolve_agenda_from_payload(payload, candidate_lookup)
                if item is None:
                    continue
                dismissed = AmbientAgendaItem(
                    agenda_id=item.agenda_id,
                    title=item.title,
                    kind=item.kind,
                    source_type=item.source_type,
                    source_ref=item.source_ref,
                    priority_score=item.priority_score,
                    status="dismissed",
                    created_at=item.created_at,
                    updated_at=now,
                    due_at=item.due_at,
                    last_considered_at=now,
                    backing_topic_id=item.backing_topic_id,
                    backing_memory_ids=item.backing_memory_ids,
                    surface_message=item.surface_message,
                )
                self.agenda.update_item(dismissed)
                applied.append(replace(action, payload={"agenda_id": item.agenda_id}))

        return applied or [AmbientReflectionAction(action_type="NOTHING", payload={})]

    def _resolve_agenda_from_payload(
        self,
        payload: dict,
        candidate_lookup: dict[str, ReflectionCandidate],
    ) -> AmbientAgendaItem | None:
        agenda_id = str(payload.get("agenda_id", "")).strip()
        if agenda_id:
            return self.agenda.get_item(agenda_id)
        candidate_id = str(payload.get("candidate_id", "")).strip()
        candidate = candidate_lookup.get(candidate_id)
        if candidate is None:
            return None
        existing = self.agenda.find_by_source(
            source_type=candidate.source_type,
            source_ref=candidate.source_ref,
            kind=candidate.proposed_kind,
            statuses=["open", "surfaced", "completed"],
        )
        if existing is not None:
            return existing
        now = datetime.now().isoformat()
        item = AmbientAgendaItem(
            agenda_id=uuid.uuid4().hex,
            title=candidate.title,
            kind=candidate.proposed_kind,
            source_type=candidate.source_type,
            source_ref=candidate.source_ref,
            priority_score=candidate.score,
            status="open",
            created_at=now,
            updated_at=now,
            due_at=candidate.due_at,
            last_considered_at=now,
            backing_topic_id=candidate.backing_topic_id,
            backing_memory_ids=candidate.backing_memory_ids,
            surface_message=candidate.summary,
        )
        self.agenda.create_item(item)
        return item

    def _all_facts(self) -> List:
        facts = []
        for speaker in self.memory.list_speakers():
            facts.extend(self.memory.get_facts(speaker.speaker_id))
        return facts

    def _parse_dt(self, value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
