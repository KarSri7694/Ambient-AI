import json
import logging
import uuid
from dataclasses import replace
from datetime import datetime
from typing import List, Optional

from application.ports.LLMProvider import LLMProvider
from application.ports.memory_port import MemoryPort
from application.services.interaction_trace import interaction_trace
from core.models import ConversationSession, OpenLoop, TranscriptEvidence


class OpenLoopService:
    """Track unresolved intents across transcript sessions."""

    CREATION_SIGNALS = {
        "commitment": "commitment",
        "follow_up_request": "follow_up_request",
        "self_reminder": "self_reminder",
        "pending_decision": "pending_decision",
        "task_in_progress": "task_in_progress",
        "curiosity_to_revisit": "curiosity_to_revisit",
    }

    DECISION_PROMPT = """You update open-loop state for an ambient audio agent.

Return JSON only:
{
  "actions": [
    {
      "action": "new|update|resolve|nothing",
      "evidence_id": "evidence id",
      "loop_id": "existing loop id or empty string",
      "loop_type": "commitment|follow_up_request|self_reminder|pending_decision|task_in_progress|curiosity_to_revisit|social_obligation",
      "title": "short title",
      "confidence": 0.0,
      "urgency": 0.0,
      "due_hint": "optional due hint",
      "next_action_hint": "optional next action hint",
      "resolution_summary": "optional resolution summary"
    }
  ]
}

Rules:
- Use only the provided evidence and loop list.
- Prefer update over creating duplicates.
- Resolve only when the new evidence clearly indicates completion/cancellation.
- Hinglish is expected.
"""

    def __init__(self, memory: MemoryPort, llm_provider: LLMProvider | None = None, logger: logging.Logger | None = None):
        self.memory = memory
        self.llm = llm_provider
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    async def process(self, *, session: ConversationSession, evidence_items: List[TranscriptEvidence], model: str = "") -> List[OpenLoop]:
        affected: List[OpenLoop] = []
        llm_actions = await self._decide_actions(session=session, evidence_items=evidence_items, model=model)
        if llm_actions:
            affected = self._apply_llm_actions(session=session, evidence_items=evidence_items, actions=llm_actions)
            if affected:
                self.refresh_digest()
                return affected
        for item in evidence_items:
            if item.signal_type == "completion":
                resolved = self._resolve_existing(item)
                if resolved is not None:
                    affected.append(resolved)
                continue

            loop_type = self.CREATION_SIGNALS.get(item.signal_type)
            if loop_type is None or item.trust_score < 0.45:
                continue
            existing = self._find_existing(item, session)
            if existing is None:
                new_loop = OpenLoop(
                    loop_id=uuid.uuid4().hex,
                    title=self._title(item),
                    loop_type=loop_type,
                    status="open" if item.trust_score >= 0.65 else "tentative",
                    owner_speaker_id=item.speaker_id,
                    source_session_id=session.session_id,
                    supporting_event_ids=[item.evidence_id],
                    confidence=item.trust_score,
                    urgency=self._urgency(item),
                    due_hint=item.time_hints[0] if item.time_hints else None,
                    next_action_hint=item.action_hints[0] if item.action_hints else None,
                    last_updated_at=item.created_at,
                    resolution_summary=None,
                )
                self.memory.upsert_open_loop(new_loop)
                affected.append(new_loop)
            else:
                updated = OpenLoop(
                    loop_id=existing.loop_id,
                    title=existing.title,
                    loop_type=existing.loop_type,
                    status="open" if max(existing.confidence, item.trust_score) >= 0.6 else existing.status,
                    owner_speaker_id=existing.owner_speaker_id,
                    source_session_id=existing.source_session_id,
                    supporting_event_ids=self._append_unique(existing.supporting_event_ids, item.evidence_id),
                    confidence=max(existing.confidence, item.trust_score),
                    urgency=max(existing.urgency, self._urgency(item)),
                    due_hint=existing.due_hint or (item.time_hints[0] if item.time_hints else None),
                    next_action_hint=existing.next_action_hint or (item.action_hints[0] if item.action_hints else None),
                    last_updated_at=item.created_at,
                    resolution_summary=existing.resolution_summary,
                )
                self.memory.upsert_open_loop(updated)
                affected.append(updated)

        if affected:
            self._refresh_session_links(session, affected)
        self.refresh_digest()
        return affected

    async def _decide_actions(self, *, session: ConversationSession, evidence_items: List[TranscriptEvidence], model: str) -> List[dict]:
        if self.llm is None or not model or not evidence_items:
            return []
        try:
            existing_loops = self.memory.list_open_loops(statuses=["open", "tentative"], limit=20)
            payload = {
                "session": {
                    "session_id": session.session_id,
                    "topic_summary": session.topic_summary,
                    "entity_summary": session.entity_summary,
                },
                "existing_loops": [
                    {
                        "loop_id": loop.loop_id,
                        "title": loop.title,
                        "loop_type": loop.loop_type,
                        "status": loop.status,
                        "owner_speaker_id": loop.owner_speaker_id,
                        "confidence": loop.confidence,
                        "urgency": loop.urgency,
                        "due_hint": loop.due_hint,
                        "next_action_hint": loop.next_action_hint,
                    }
                    for loop in existing_loops
                ],
                "new_evidence": [
                    {
                        "evidence_id": item.evidence_id,
                        "speaker_id": item.speaker_id,
                        "signal_type": item.signal_type,
                        "content": item.content,
                        "normalized_entities": item.normalized_entities,
                        "time_hints": item.time_hints,
                        "action_hints": item.action_hints,
                        "trust_score": item.trust_score,
                    }
                    for item in evidence_items
                ],
            }
            with interaction_trace("open_loop_decision"):
                completion = await self.llm.chat_completion_stream(
                    model=model,
                    messages=[
                        {"role": "system", "content": self.DECISION_PROMPT},
                        {"role": "user", "content": json.dumps(payload, indent=2)},
                    ],
                    tools=None,
                )
            text = await self._consume_stream_text(completion)
            parsed = self._parse_json_object(text)
            actions = parsed.get("actions", [])
            return actions if isinstance(actions, list) else []
        except Exception as exc:
            self.logger.warning("Open-loop LLM decision failed, using heuristic fallback: %s", exc)
            return []

    def _apply_llm_actions(
        self,
        *,
        session: ConversationSession,
        evidence_items: List[TranscriptEvidence],
        actions: List[dict],
    ) -> List[OpenLoop]:
        evidence_by_id = {item.evidence_id: item for item in evidence_items}
        affected: List[OpenLoop] = []
        for action in actions:
            if not isinstance(action, dict):
                continue
            action_name = str(action.get("action", "")).strip().lower()
            item = evidence_by_id.get(str(action.get("evidence_id", "")).strip())
            if item is None or action_name == "nothing":
                continue
            if action_name == "new":
                loop = OpenLoop(
                    loop_id=uuid.uuid4().hex,
                    title=str(action.get("title") or self._title(item)).strip(),
                    loop_type=str(action.get("loop_type") or self.CREATION_SIGNALS.get(item.signal_type, "commitment")).strip(),
                    status="open" if self._num(action.get("confidence"), item.trust_score) >= 0.6 else "tentative",
                    owner_speaker_id=item.speaker_id,
                    source_session_id=session.session_id,
                    supporting_event_ids=[item.evidence_id],
                    confidence=self._num(action.get("confidence"), item.trust_score),
                    urgency=self._num(action.get("urgency"), self._urgency(item)),
                    due_hint=self._opt_text(action.get("due_hint")) or (item.time_hints[0] if item.time_hints else None),
                    next_action_hint=self._opt_text(action.get("next_action_hint")) or (item.action_hints[0] if item.action_hints else None),
                    last_updated_at=item.created_at,
                    resolution_summary=None,
                )
                self.memory.upsert_open_loop(loop)
                affected.append(loop)
                continue
            existing = self.memory.get_open_loop(str(action.get("loop_id", "")).strip())
            if existing is None:
                continue
            if action_name == "resolve":
                resolved = replace(
                    existing,
                    status="resolved",
                    supporting_event_ids=self._append_unique(existing.supporting_event_ids, item.evidence_id),
                    last_updated_at=item.created_at,
                    resolution_summary=self._opt_text(action.get("resolution_summary")) or item.content,
                )
                self.memory.upsert_open_loop(resolved)
                affected.append(resolved)
                continue
            if action_name == "update":
                updated = replace(
                    existing,
                    supporting_event_ids=self._append_unique(existing.supporting_event_ids, item.evidence_id),
                    confidence=max(existing.confidence, self._num(action.get("confidence"), item.trust_score)),
                    urgency=max(existing.urgency, self._num(action.get("urgency"), self._urgency(item))),
                    due_hint=existing.due_hint or self._opt_text(action.get("due_hint")) or (item.time_hints[0] if item.time_hints else None),
                    next_action_hint=existing.next_action_hint or self._opt_text(action.get("next_action_hint")) or (item.action_hints[0] if item.action_hints else None),
                    last_updated_at=item.created_at,
                )
                self.memory.upsert_open_loop(updated)
                affected.append(updated)
        if affected:
            self._refresh_session_links(session, affected)
        return affected

    def _resolve_existing(self, item: TranscriptEvidence) -> Optional[OpenLoop]:
        loops = self.memory.list_open_loops(statuses=["open", "tentative"], limit=20)
        for loop in loops:
            if loop.owner_speaker_id != item.speaker_id:
                continue
            if loop.next_action_hint and loop.next_action_hint in item.content.lower():
                return self._mark_resolved(loop, item)
        for loop in loops:
            if loop.owner_speaker_id == item.speaker_id:
                return self._mark_resolved(loop, item)
        return None

    def _mark_resolved(self, loop: OpenLoop, item: TranscriptEvidence) -> OpenLoop:
        resolved = replace(
            loop,
            status="resolved",
            supporting_event_ids=self._append_unique(loop.supporting_event_ids, item.evidence_id),
            last_updated_at=item.created_at,
            resolution_summary=item.content,
        )
        self.memory.upsert_open_loop(resolved)
        return resolved

    def _find_existing(self, item: TranscriptEvidence, session: ConversationSession) -> Optional[OpenLoop]:
        loops = self.memory.list_open_loops(statuses=["open", "tentative"], limit=20)
        item_entities = set(item.normalized_entities)
        for loop in loops:
            if loop.owner_speaker_id != item.speaker_id:
                continue
            if loop.source_session_id == session.session_id and loop.loop_type == self.CREATION_SIGNALS[item.signal_type]:
                return loop
            if item_entities and any(entity in loop.title.lower() for entity in item_entities):
                return loop
        return None

    def refresh_digest(self, limit: int = 10) -> None:
        loops = self.memory.list_open_loops(statuses=["open", "tentative", "resolved"], limit=limit)
        lines = ["# Open Loops", ""]
        if not loops:
            lines.append("- No tracked open loops.")
        for loop in loops:
            lines.append(f"- [{loop.status}] {loop.title} ({loop.loop_type})")
            if loop.due_hint:
                lines.append(f"  Due hint: {loop.due_hint}")
            if loop.next_action_hint:
                lines.append(f"  Next action: {loop.next_action_hint}")
        self.memory.save_open_loop_digest("\n".join(lines) + "\n")

    def _title(self, item: TranscriptEvidence) -> str:
        content = item.content.strip()
        if len(content) <= 80:
            return content
        return content[:77] + "..."

    def _urgency(self, item: TranscriptEvidence) -> float:
        urgency = 0.45
        if item.time_hints:
            urgency += 0.2
        if item.signal_type in {"self_reminder", "follow_up_request"}:
            urgency += 0.15
        return min(0.95, urgency)

    def _append_unique(self, values: List[str], new_value: str) -> List[str]:
        if new_value in values:
            return list(values)
        return [*values, new_value]

    def _refresh_session_links(self, session: ConversationSession, loops: List[OpenLoop]) -> None:
        loop_ids = {*(session.derived_loop_ids or []), *(loop.loop_id for loop in loops)}
        updated = replace(session, derived_loop_ids=sorted(loop_ids), ended_at=datetime.now().isoformat())
        self.memory.upsert_session(updated)

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
            return {}

    def _num(self, value, fallback: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return fallback

    def _opt_text(self, value) -> Optional[str]:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None
