import json
import logging
import re
import uuid
from datetime import datetime
from typing import Dict, List

from application.ports.LLMProvider import LLMProvider
from application.services.interaction_trace import interaction_trace
from core.models import TranscriptEvidence, TranscriptParticipant, TranscriptTurn


class TranscriptEvidenceService:
    """Convert raw transcript turns into bounded structured evidence."""

    ENTITY_PATTERN = re.compile(r"\b(?:[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*)\b")
    TIME_PATTERN = re.compile(
        r"\b(?:today|tomorrow|tonight|next week|this week|monday|tuesday|wednesday|thursday|friday|saturday|sunday|morning|evening)\b",
        re.IGNORECASE,
    )

    EXTRACTION_PROMPT = """You extract structured transcript evidence for an ambient audio agent.

Return JSON only in this exact shape:
{
  "evidence": [
    {
      "speaker_label": "exact speaker label from transcript",
      "signal_type": "commitment|follow_up_request|self_reminder|pending_decision|task_in_progress|curiosity_to_revisit|completion|relationship|preference|context",
      "content": "short paraphrase or exact short content",
      "normalized_entities": ["entity one", "entity two"],
      "time_hints": ["tomorrow"],
      "action_hints": ["call", "send"],
      "trust_score": 0.0
    }
  ]
}

Rules:
- Output at most one evidence item per transcript turn.
- Hinglish is expected. Infer meaning conservatively.
- Use completion only when the speaker indicates a task is done, cancelled, or no longer needed.
- Use trust_score between 0 and 1 based on transcript clarity and certainty.
- Keep normalized_entities short and lowercase.
- If a turn is noisy or meaningless, use signal_type=context with low trust_score.
"""

    def __init__(self, llm_provider: LLMProvider | None = None, logger: logging.Logger | None = None):
        self.llm = llm_provider
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    async def extract(
        self,
        *,
        turns: List[TranscriptTurn],
        participants: Dict[str, TranscriptParticipant],
        source_ref: str,
        session_id: str | None = None,
        model: str = "",
    ) -> List[TranscriptEvidence]:
        created_at = datetime.now().isoformat()
        extracted = self._extract_with_heuristics(turns)
        if self.llm is not None and model:
            llm_items = await self._extract_with_llm(
                turns=turns,
                participants=participants,
                model=model,
            )
            if llm_items:
                extracted = llm_items
        evidence: List[TranscriptEvidence] = []
        for turn, extracted_item in zip(turns, extracted):
            participant = self._resolve_participant(
                turn=turn,
                extracted_item=extracted_item,
                participants=participants,
            )
            if participant is None:
                continue
            evidence.append(
                TranscriptEvidence(
                    evidence_id=uuid.uuid4().hex,
                    source_ref=source_ref,
                    speaker_id=participant.speaker_id,
                    speaker_label=turn.speaker_label,
                    session_id=session_id,
                    signal_type=str(extracted_item.get("signal_type", "context")).strip() or "context",
                    content=str(extracted_item.get("content", turn.text)).strip() or turn.text,
                    normalized_entities=self._clean_string_list(extracted_item.get("normalized_entities", [])),
                    time_hints=self._clean_string_list(extracted_item.get("time_hints", [])),
                    action_hints=self._clean_string_list(extracted_item.get("action_hints", [])),
                    trust_score=self._coerce_score(
                        extracted_item.get("trust_score"),
                        fallback=self._trust_score(turn.text, participant.confidence),
                    ),
                    created_at=created_at,
                )
            )
        if not evidence and turns:
            self.logger.warning(
                "LLM evidence extraction produced zero usable items for %s; falling back to heuristic alignment.",
                source_ref,
            )
            return await self.extract(
                turns=turns,
                participants=participants,
                source_ref=source_ref,
                session_id=session_id,
                model="",
            )
        return evidence

    async def _extract_with_llm(
        self,
        *,
        turns: List[TranscriptTurn],
        participants: Dict[str, TranscriptParticipant],
        model: str,
    ) -> List[dict]:
        try:
            transcript_lines = []
            for turn in turns:
                transcript_lines.append(f"{turn.speaker_label}: {turn.text}")
            participant_lines = [
                f"- {participant.speaker_label} => {participant.display_name}"
                for participant in participants.values()
            ]
            with interaction_trace("transcript_evidence"):
                completion = await self.llm.chat_completion_stream(
                    model=model,
                    messages=[
                        {"role": "system", "content": self.EXTRACTION_PROMPT},
                        {
                            "role": "user",
                            "content": "Participants:\n"
                            + "\n".join(participant_lines)
                            + "\n\nTranscript turns:\n"
                            + "\n".join(transcript_lines),
                        },
                    ],
                    tools=None,
                )
            response_text = await self._consume_stream_text(completion)
            parsed = self._parse_json_object(response_text)
            raw_items = parsed.get("evidence", [])
            if not isinstance(raw_items, list):
                return []
            cleaned: List[dict] = []
            for raw_item in raw_items[: len(turns)]:
                if not isinstance(raw_item, dict):
                    continue
                cleaned.append(raw_item)
            return cleaned
        except Exception as exc:
            self.logger.warning("Evidence extraction LLM call failed, using heuristics: %s", exc)
            return []

    def _signal_type(self, text: str) -> str:
        lowered = text.lower()
        if any(token in lowered for token in ("i sent", "done", "finished", "already booked", "never mind")):
            return "completion"
        if any(token in lowered for token in ("remind me", "i need to", "need to", "don't forget")):
            return "self_reminder"
        if any(token in lowered for token in ("i will", "i'll", "kar dunga", "send it", "will send", "will do")):
            return "commitment"
        if any(token in lowered for token in ("send me", "please send", "let me know", "tell me", "share")):
            return "follow_up_request"
        if any(token in lowered for token in ("decide", "choose between", "not sure", "which one")):
            return "pending_decision"
        if any(token in lowered for token in ("interesting", "heard about", "check out", "look into", "research")):
            return "curiosity_to_revisit"
        if any(token in lowered for token in ("working on", "still writing", "halfway", "continue", "finish")):
            return "task_in_progress"
        if any(token in lowered for token in ("my friend", "my mom", "my dad", "client", "manager", "brother", "sister")):
            return "relationship"
        if any(token in lowered for token in ("like", "love", "prefer", "favorite", "hate")):
            return "preference"
        return "context"

    def _trust_score(self, text: str, speaker_confidence: float) -> float:
        length_bonus = min(len(text.split()) / 12.0, 1.0) * 0.25
        malformed_penalty = 0.2 if len(text.strip()) < 4 else 0.0
        repetition_penalty = 0.15 if len(set(text.lower().split())) <= 2 else 0.0
        base = 0.45 + min(max(speaker_confidence, 0.0), 1.0) * 0.2 + length_bonus
        return max(0.05, min(0.98, base - malformed_penalty - repetition_penalty))

    def _entities(self, text: str) -> List[str]:
        entities = {match.group(0).strip().lower() for match in self.ENTITY_PATTERN.finditer(text)}
        lowered = text.lower()
        for token in ("amazon", "gmail", "youtube", "todoist", "game", "email", "meeting"):
            if token in lowered:
                entities.add(token)
        return sorted(entities)

    def _time_hints(self, text: str) -> List[str]:
        return [match.group(0).lower() for match in self.TIME_PATTERN.finditer(text)]

    def _action_hints(self, text: str) -> List[str]:
        hints = []
        lowered = text.lower()
        for token in ("send", "call", "book", "buy", "research", "write", "finish", "reply", "schedule"):
            if token in lowered:
                hints.append(token)
        return hints

    def _extract_with_heuristics(self, turns: List[TranscriptTurn]) -> List[dict]:
        extracted: List[dict] = []
        for turn in turns:
            extracted.append(
                {
                    "speaker_label": turn.speaker_label,
                    "signal_type": self._signal_type(turn.text),
                    "content": turn.text,
                    "normalized_entities": self._entities(turn.text),
                    "time_hints": self._time_hints(turn.text),
                    "action_hints": self._action_hints(turn.text),
                }
            )
        return extracted

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
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _clean_string_list(self, value) -> List[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip().lower() for item in value if str(item).strip()]

    def _coerce_score(self, value, fallback: float) -> float:
        try:
            return max(0.05, min(0.98, float(value)))
        except (TypeError, ValueError):
            return fallback

    def _resolve_participant(
        self,
        *,
        turn: TranscriptTurn,
        extracted_item: dict,
        participants: Dict[str, TranscriptParticipant],
    ) -> TranscriptParticipant | None:
        direct = participants.get(turn.speaker_label)
        if direct is not None:
            return direct

        llm_label = str(extracted_item.get("speaker_label", "")).strip()
        if llm_label:
            direct = participants.get(llm_label)
            if direct is not None:
                return direct
            normalized_llm = self._normalize_label(llm_label)
            for raw_label, participant in participants.items():
                if self._normalize_label(raw_label) == normalized_llm:
                    return participant
        return None

    def _normalize_label(self, label: str) -> str:
        label = label.strip().lower()
        label = re.sub(r"\s*-\s*\[\d+(?:\.\d+)?%\]\s*$", "", label)
        return " ".join(label.split())
