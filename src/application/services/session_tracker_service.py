import json
import logging
import uuid
from datetime import datetime
from typing import Iterable, List

from application.ports.LLMProvider import LLMProvider
from application.ports.memory_port import MemoryPort
from application.services.interaction_trace import interaction_trace
from core.models import ConversationSession, TranscriptEvidence


class SessionTrackerService:
    """Group transcript evidence into short-lived conversational sessions."""

    DECISION_PROMPT = """You decide whether new transcript evidence belongs to an existing session.

Return JSON only:
{
  "decision": "new|continue",
  "session_id": "existing session id or empty string",
  "continuation_score": 0.0,
  "topic_summary": "short session topic summary",
  "entity_summary": "comma-separated entities",
  "recent_turn_summary": "short latest summary"
}

Rules:
- Prefer continue when the new evidence is clearly about the same people/task/topic, even in Hinglish.
- Prefer new when topic or participants shift materially.
- continuation_score must be between 0 and 1.
"""

    def __init__(self, memory: MemoryPort, llm_provider: LLMProvider | None = None, logger: logging.Logger | None = None):
        self.memory = memory
        self.llm = llm_provider
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    async def attach_to_session(self, evidence_items: List[TranscriptEvidence], model: str = "") -> ConversationSession:
        if not evidence_items:
            raise ValueError("attach_to_session requires at least one evidence item")

        open_sessions = self.memory.list_sessions(statuses=["open"], limit=5)
        best = await self._best_match(open_sessions, evidence_items, model=model)
        if best is None:
            now = evidence_items[-1].created_at
            best = ConversationSession(
                session_id=uuid.uuid4().hex,
                started_at=now,
                ended_at=now,
                participant_ids=sorted({item.speaker_id for item in evidence_items}),
                status="open",
                topic_summary=self._topic_summary(evidence_items),
                entity_summary=", ".join(self._merge_entities(evidence_items)),
                recent_turn_summary=self._recent_summary(evidence_items),
                last_activity_at=now,
                continuation_score=0.0,
                derived_loop_ids=[],
            )
        else:
            merged_participants = sorted({*best.participant_ids, *(item.speaker_id for item in evidence_items)})
            merged_entities = self._merge_entities(evidence_items, existing=best.entity_summary.split(", "))
            best = ConversationSession(
                session_id=best.session_id,
                started_at=best.started_at,
                ended_at=evidence_items[-1].created_at,
                participant_ids=merged_participants,
                status="open",
                topic_summary=self._merge_text(best.topic_summary, self._topic_summary(evidence_items)),
                entity_summary=", ".join(merged_entities),
                recent_turn_summary=self._merge_text(best.recent_turn_summary, self._recent_summary(evidence_items)),
                last_activity_at=evidence_items[-1].created_at,
                continuation_score=self._score(best, evidence_items),
                derived_loop_ids=best.derived_loop_ids,
            )
        self.memory.upsert_session(best)
        return best

    def refresh_digest(self, limit: int = 5) -> None:
        sessions = self.memory.list_sessions(statuses=["open"], limit=limit)
        lines = ["# Session Digest", ""]
        if not sessions:
            lines.append("- No active sessions.")
        for session in sessions:
            lines.append(f"- Session {session.session_id[:8]} | topics: {session.topic_summary or 'n/a'}")
            if session.entity_summary:
                lines.append(f"  Entities: {session.entity_summary}")
            if session.recent_turn_summary:
                lines.append(f"  Recent: {session.recent_turn_summary}")
        self.memory.save_session_digest("\n".join(lines) + "\n")

    async def _best_match(
        self,
        sessions: Iterable[ConversationSession],
        evidence_items: List[TranscriptEvidence],
        model: str = "",
    ) -> ConversationSession | None:
        session_list = list(sessions)
        if self.llm is not None and model and session_list:
            decision = await self._llm_decision(session_list, evidence_items, model)
            if decision.get("decision") == "continue":
                session_id = str(decision.get("session_id", "")).strip()
                for session in session_list:
                    if session.session_id == session_id:
                        return session
        ranked = [(self._score(session, evidence_items), session) for session in session_list]
        ranked = [item for item in ranked if item[0] >= 0.45]
        if not ranked:
            return None
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[0][1]

    async def _llm_decision(
        self,
        sessions: List[ConversationSession],
        evidence_items: List[TranscriptEvidence],
        model: str,
    ) -> dict:
        try:
            payload = {
                "open_sessions": [
                    {
                        "session_id": session.session_id,
                        "participant_ids": session.participant_ids,
                        "topic_summary": session.topic_summary,
                        "entity_summary": session.entity_summary,
                        "recent_turn_summary": session.recent_turn_summary,
                    }
                    for session in sessions
                ],
                "new_evidence": [
                    {
                        "speaker_label": item.speaker_label,
                        "signal_type": item.signal_type,
                        "content": item.content,
                        "normalized_entities": item.normalized_entities,
                    }
                    for item in evidence_items
                ],
            }
            with interaction_trace("session_tracker"):
                completion = await self.llm.chat_completion_stream(
                    model=model,
                    messages=[
                        {"role": "system", "content": self.DECISION_PROMPT},
                        {"role": "user", "content": json.dumps(payload, indent=2)},
                    ],
                    tools=None,
                    temperature=0.1,
                    top_p=0.9,
                )
            text = await self._consume_stream_text(completion)
            parsed = self._parse_json_object(text)
            return parsed if isinstance(parsed, dict) else {}
        except Exception as exc:
            self.logger.warning("Session LLM decision failed, using heuristic fallback: %s", exc)
            return {}

    def _score(self, session: ConversationSession, evidence_items: List[TranscriptEvidence]) -> float:
        participant_overlap = len(set(session.participant_ids) & {item.speaker_id for item in evidence_items})
        entity_overlap = len(set(filter(None, (entity.strip().lower() for entity in session.entity_summary.split(",")))) & set(self._merge_entities(evidence_items)))
        continuity_markers = sum(
            1
            for item in evidence_items
            if any(token in item.content.lower() for token in ("it", "that", "also", "still", "then"))
        )
        score = 0.2 + min(participant_overlap, 2) * 0.2 + min(entity_overlap, 3) * 0.12 + min(continuity_markers, 2) * 0.12
        return min(score, 0.95)

    def _topic_summary(self, evidence_items: List[TranscriptEvidence]) -> str:
        signals = [item.signal_type for item in evidence_items if item.signal_type not in {"context", "relationship"}]
        entities = self._merge_entities(evidence_items)
        parts = []
        if signals:
            parts.append(", ".join(sorted(dict.fromkeys(signals))))
        if entities:
            parts.append(", ".join(entities[:4]))
        return " | ".join(parts) or "general conversation"

    def _recent_summary(self, evidence_items: List[TranscriptEvidence]) -> str:
        snippets = [item.content.strip() for item in evidence_items[-2:] if item.content.strip()]
        return " / ".join(snippets[:2])[:240]

    def _merge_entities(
        self,
        evidence_items: List[TranscriptEvidence],
        existing: Iterable[str] | None = None,
    ) -> List[str]:
        entities = {entity.strip().lower() for entity in (existing or []) if entity.strip()}
        for item in evidence_items:
            entities.update(item.normalized_entities)
        return sorted(entities)

    def _merge_text(self, existing: str, new: str) -> str:
        if not existing:
            return new
        if not new or new in existing:
            return existing
        return f"{existing} | {new}"[:400]

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
