import json
import logging
import re
from typing import Dict, List

from application.ports.LLMProvider import LLMProvider
from core.models import TranscriptClassificationResult, TranscriptParticipant, TranscriptTurn


class TranscriptClassificationService:
    """Classify one transcript chunk into a single bounded action bucket."""

    VALID_LABELS = {
        "FACT",
        "REMINDER",
        "PREFERENCE",
        "TASK_SIMPLE",
        "TASK_COMPLEX",
        "NOTHING",
    }
    TRANSCRIPT_LINE_PATTERN = re.compile(
        r"^\[(?P<start>\d+(?:\.\d+)?)\s*-\s*(?P<end>\d+(?:\.\d+)?)\]\s*->\s*(?P<label>.*?):\s*(?P<text>.*)$"
    )
    REMINDER_TOKENS = ("tomorrow", "birthday", "remind", "call", "meeting", "today", "next week", "kal ")
    PREFERENCE_TOKENS = (" like ", " love ", " hate ", " prefer ", " favorite ", "watch ")
    SIMPLE_TOKENS = ("schedule", "book", "create a task", "add reminder", "meeting at", "todo")
    COMPLEX_TOKENS = ("send me", "resources", "learn", "search", "find", "notes", "please help")

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

    async def classify(
        self,
        transcript_text: str,
        participants: Dict[str, TranscriptParticipant],
        system_prompt: str,
        model: str,
    ) -> TranscriptClassificationResult:
        if self.llm is not None:
            llm_result = await self._classify_with_llm(transcript_text, system_prompt, model)
            if llm_result is not None and llm_result.label in self.VALID_LABELS:
                return llm_result
        return self._classify_with_heuristics(transcript_text, participants)

    async def _classify_with_llm(
        self,
        transcript_text: str,
        system_prompt: str,
        model: str,
    ) -> TranscriptClassificationResult | None:
        try:
            completion = await self.llm.chat_completion_stream(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": transcript_text},
                ],
                tools=None,
                temperature=0.1,
                top_p=0.9,
            )
            response_text = await self._consume_stream_text(completion)
            return self._parse_llm_response(response_text)
        except Exception as exc:
            self.logger.warning("Transcript classification LLM call failed, using heuristics: %s", exc)
            return None

    async def _consume_stream_text(self, completion) -> str:
        parts: List[str] = []
        async for chunk in completion:
            delta = chunk.choices[0].delta
            if delta.content:
                parts.append(delta.content)
        return "".join(parts)

    def _parse_llm_response(self, response_text: str) -> TranscriptClassificationResult | None:
        response_text = response_text.strip()
        if not response_text:
            return None
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start != -1 and end != -1:
            response_text = response_text[start : end + 1]
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            return None
        label = str(parsed.get("label", "")).strip()
        if label not in self.VALID_LABELS:
            return None
        return TranscriptClassificationResult(
            label=label,
            speaker_label=str(parsed.get("speaker_label", "")).strip(),
            summary=str(parsed.get("summary", "")).strip(),
            confidence=float(parsed.get("confidence", 0.0)),
            reason=str(parsed.get("reason", "")).strip(),
            suggested_action=self._clean_optional(parsed.get("suggested_action")),
            memory_content=self._clean_optional(parsed.get("memory_content")),
        )

    def _classify_with_heuristics(
        self,
        transcript_text: str,
        participants: Dict[str, TranscriptParticipant],
    ) -> TranscriptClassificationResult:
        turns = self.parse_transcript(transcript_text)
        if not turns:
            return TranscriptClassificationResult(
                label="NOTHING",
                speaker_label="",
                summary="No valid transcript turns found.",
                confidence=0.1,
                reason="no_parsed_turns",
            )

        for turn in turns:
            lowered = f" {turn.text.lower()} "
            if any(token in lowered for token in self.REMINDER_TOKENS):
                return TranscriptClassificationResult(
                    label="REMINDER",
                    speaker_label=turn.speaker_label,
                    summary=turn.text,
                    confidence=0.7,
                    reason="matched_reminder_tokens",
                    suggested_action=turn.text,
                )
            if any(token in lowered for token in self.PREFERENCE_TOKENS):
                return TranscriptClassificationResult(
                    label="PREFERENCE",
                    speaker_label=turn.speaker_label,
                    summary=turn.text,
                    confidence=0.72,
                    reason="matched_preference_tokens",
                    memory_content=turn.text,
                )
            if any(token in lowered for token in self.SIMPLE_TOKENS):
                return TranscriptClassificationResult(
                    label="TASK_SIMPLE",
                    speaker_label=turn.speaker_label,
                    summary=turn.text,
                    confidence=0.68,
                    reason="matched_simple_task_tokens",
                    suggested_action=turn.text,
                )
            if any(token in lowered for token in self.COMPLEX_TOKENS):
                return TranscriptClassificationResult(
                    label="TASK_COMPLEX",
                    speaker_label=turn.speaker_label,
                    summary=turn.text,
                    confidence=0.75,
                    reason="matched_complex_task_tokens",
                    suggested_action=turn.text,
                )
            if len(turn.text.split()) >= 8:
                return TranscriptClassificationResult(
                    label="FACT",
                    speaker_label=turn.speaker_label,
                    summary=turn.text,
                    confidence=0.55,
                    reason="long_statement_fallback",
                    memory_content=turn.text,
                )

        first_turn = turns[0]
        fallback_label = "NOTHING"
        if first_turn.speaker_label in participants:
            fallback_summary = first_turn.text
        else:
            fallback_summary = "No clear action or durable fact detected."
        return TranscriptClassificationResult(
            label=fallback_label,
            speaker_label=first_turn.speaker_label,
            summary=fallback_summary,
            confidence=0.2,
            reason="no_heuristic_match",
        )

    def _clean_optional(self, value) -> str | None:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None
