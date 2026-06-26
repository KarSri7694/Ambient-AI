import json
import logging
import re
import uuid
from datetime import datetime
from typing import Dict, Iterable, List

from application.ports.LLMProvider import LLMProvider
from application.services.interaction_trace import interaction_trace
from core.models import MemoryEvent, TranscriptParticipant, TranscriptTurn


class MemoryExtractionService:
    """Extract candidate memory events from transcript text."""

    TRANSCRIPT_LINE_PATTERN = re.compile(
        r"^\[(?P<start>\d+(?:\.\d+)?)\s*-\s*(?P<end>\d+(?:\.\d+)?)\]\s*->\s*(?P<label>.*?):\s*(?P<text>.*)$"
    )

    EXTRACTION_PROMPT = """You extract long-term memory candidates from transcripts.

Return JSON only.
Output format:
[
  {
    "speaker_label": "exact speaker label from transcript",
    "event_kind": "preference|commitment|relationship|identity|schedule|project|general_fact",
    "content": "one durable fact or commitment in one sentence",
    "confidence": 0.0
  }
]

Rules:
- Extract only durable or reminder-worthy information.
- Skip chatter, filler, and things that make no sense.
- Keep speaker_label exactly as it appears in the transcript.
- Confidence must be between 0 and 1.
"""

    PREFERENCE_PATTERNS = (" like ", " love ", " hate ", " prefer ", " favorite ", "watch ")
    COMMITMENT_PATTERNS = ("i will", "i'll", "sure", "dekhunga", "kar dunga", "laaunga", "send", "remind")
    SCHEDULE_PATTERNS = ("tomorrow", "birthday", "meeting", "today", "next week", "kal ")
    IDENTITY_PATTERNS = ("i am", "i'm", "my name is", "works on", "working on", "studying")

    def __init__(self, llm_provider: LLMProvider | None = None, logger: logging.Logger | None = None):
        self.llm = llm_provider
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def parse_transcript(self, transcript_text: str) -> List[TranscriptTurn]:
        turns: List[TranscriptTurn] = []
        for line in transcript_text.splitlines():
            match = self.TRANSCRIPT_LINE_PATTERN.match(line.strip())
            if not match:
                continue
            turns.append(
                TranscriptTurn(
                    speaker_label=match.group("label").strip(),
                    text=match.group("text").strip(),
                    start_time=float(match.group("start")),
                    end_time=float(match.group("end")),
                )
            )
        return turns

    async def extract_events(
        self,
        transcript_text: str,
        participants: Dict[str, TranscriptParticipant],
        model: str,
        source_ref: str,
    ) -> List[MemoryEvent]:
        turns = self.parse_transcript(transcript_text)
        extracted = []
        if self.llm is not None:
            extracted = await self._extract_with_llm(transcript_text, model)
        if not extracted:
            extracted = self._extract_with_heuristics(turns)

        events: List[MemoryEvent] = []
        created_at = datetime.now().isoformat()
        for item in extracted:
            speaker_label = item.get("speaker_label", "").strip()
            participant = participants.get(speaker_label)
            if participant is None:
                continue
            content = item.get("content", "").strip()
            if not content:
                continue
            events.append(
                MemoryEvent(
                    event_id=uuid.uuid4().hex,
                    speaker_id=participant.speaker_id,
                    source_type="transcript",
                    source_ref=source_ref,
                    event_kind=item.get("event_kind", "general_fact"),
                    content=content,
                    confidence=float(item.get("confidence", 0.5)),
                    status="candidate",
                    created_at=created_at,
                )
            )
        return events

    async def _extract_with_llm(self, transcript_text: str, model: str) -> List[dict]:
        try:
            with interaction_trace("memory_extraction"):
                completion = await self.llm.chat_completion_stream(
                    model=model,
                    messages=[
                        {"role": "system", "content": self.EXTRACTION_PROMPT},
                        {"role": "user", "content": transcript_text},
                    ],
                    tools=None,
                )
            response_text = await self._consume_stream_text(completion)
            return self._parse_json_array(response_text)
        except Exception as exc:
            self.logger.warning("Memory extraction LLM call failed, falling back to heuristics: %s", exc)
            return []

    async def _consume_stream_text(self, completion) -> str:
        parts: List[str] = []
        async for chunk in completion:
            delta = chunk.choices[0].delta
            if delta.content:
                parts.append(delta.content)
        return "".join(parts)

    def _parse_json_array(self, response_text: str) -> List[dict]:
        response_text = response_text.strip()
        if not response_text:
            return []
        if not response_text.startswith("["):
            start = response_text.find("[")
            end = response_text.rfind("]")
            if start != -1 and end != -1:
                response_text = response_text[start : end + 1]
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []

    def _extract_with_heuristics(self, turns: Iterable[TranscriptTurn]) -> List[dict]:
        events: List[dict] = []
        for turn in turns:
            lowered = f" {turn.text.lower()} "
            event_kind = None
            if any(token in lowered for token in self.PREFERENCE_PATTERNS):
                event_kind = "preference"
            elif any(token in lowered for token in self.COMMITMENT_PATTERNS):
                event_kind = "commitment"
            elif any(token in lowered for token in self.SCHEDULE_PATTERNS):
                event_kind = "schedule"
            elif any(token in lowered for token in self.IDENTITY_PATTERNS):
                event_kind = "identity"
            elif len(turn.text.split()) >= 8:
                event_kind = "general_fact"

            if event_kind is None:
                continue
            events.append(
                {
                    "speaker_label": turn.speaker_label,
                    "event_kind": event_kind,
                    "content": turn.text,
                    "confidence": 0.55,
                }
            )
        return events
