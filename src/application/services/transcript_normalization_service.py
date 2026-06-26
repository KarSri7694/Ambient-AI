import json
import logging
import re
from typing import List

from application.ports.LLMProvider import LLMProvider
from application.services.interaction_trace import interaction_trace
from core.models import TranscriptTurn


class TranscriptNormalizationService:
    """Clean and normalize raw merged transcripts before downstream reasoning."""

    TRANSCRIPT_LINE_PATTERN = re.compile(
        r"^\[(?P<start>\d+(?:\.\d+)?)\s*-\s*(?P<end>\d+(?:\.\d+)?)\]\s*->\s*(?P<label>.*?):\s*(?P<text>.*)$"
    )
    SCORE_SUFFIX_PATTERN = re.compile(r"\s*-\s*\[(?P<score>-?\d+(?:\.\d+)?)%\]\s*$")
    RAW_DIARIZATION_PATTERN = re.compile(r"^speaker[_\s-]*\d+$", re.IGNORECASE)

    NORMALIZATION_PROMPT = """You normalize noisy diarized transcripts for an ambient agent.

Return JSON only in this exact shape:
{
  "turns": [
    {
      "start_time": 0.0,
      "end_time": 1.0,
      "speaker_label": "normalized speaker label",
      "text": "clean corrected text"
    }
  ]
}

Rules:
- Preserve conversation order.
- Remove confidence suffixes like "- [46.853%]" from speaker labels.
- If a raw diarization label like SPEAKER_01 is clearly/appears to be the same person as a nearby identified speaker, relabel it to that identified speaker.
- Merge fragmented utterances from the same semantic speaker when appropriate.
- Keep Hinglish meaning intact but make the wording cleaner and punctuation sensible.
- Do not invent new content.
- Do not drop substantive turns.
- Output only the normalized turns JSON.
"""

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

    async def normalize(self, transcript_text: str, model: str = "") -> str:
        turns = self.parse_transcript(transcript_text)
        if not turns:
            return transcript_text

        normalized_turns = self._normalize_with_heuristics(turns)
        if self.llm is not None and model:
            llm_turns = await self._normalize_with_llm(turns, model)
            if llm_turns:
                normalized_turns = llm_turns

        rendered = self._render_turns(normalized_turns)
        return rendered or transcript_text

    async def _normalize_with_llm(self, turns: List[TranscriptTurn], model: str) -> List[TranscriptTurn]:
        try:
            transcript_lines = [
                f"[{turn.start_time:.4f} - {turn.end_time:.4f}] -> {turn.speaker_label}: {turn.text}"
                for turn in turns
            ]
            with interaction_trace("transcript_normalization"):
                completion = await self.llm.chat_completion_stream(
                    model=model,
                    messages=[
                        {"role": "system", "content": self.NORMALIZATION_PROMPT},
                        {"role": "user", "content": "\n".join(transcript_lines)},
                    ],
                    tools=None,
                )
            response_text = await self._consume_stream_text(completion)
            parsed = self._parse_json_object(response_text)
            raw_turns = parsed.get("turns", [])
            if not isinstance(raw_turns, list):
                return []
            normalized: List[TranscriptTurn] = []
            for raw_turn in raw_turns:
                if not isinstance(raw_turn, dict):
                    continue
                try:
                    start_time = float(raw_turn.get("start_time"))
                    end_time = float(raw_turn.get("end_time"))
                except (TypeError, ValueError):
                    continue
                speaker_label = self._clean_label(str(raw_turn.get("speaker_label", "")).strip())
                text = self._clean_text(str(raw_turn.get("text", "")).strip())
                if not speaker_label or not text:
                    continue
                normalized.append(
                    TranscriptTurn(
                        speaker_label=speaker_label,
                        text=text,
                        start_time=start_time,
                        end_time=end_time,
                    )
                )
            return normalized
        except Exception as exc:
            self.logger.warning("Transcript normalization LLM call failed, using heuristics: %s", exc)
            return []

    def _normalize_with_heuristics(self, turns: List[TranscriptTurn]) -> List[TranscriptTurn]:
        canonical_turns = [
            TranscriptTurn(
                speaker_label=self._clean_label(turn.speaker_label),
                text=self._clean_text(turn.text),
                start_time=turn.start_time,
                end_time=turn.end_time,
            )
            for turn in turns
        ]

        reassigned: List[TranscriptTurn] = []
        for index, turn in enumerate(canonical_turns):
            if self._is_raw_diarization_label(turn.speaker_label):
                replacement = self._infer_nearby_named_speaker(canonical_turns, index)
                if replacement is not None:
                    turn = TranscriptTurn(
                        speaker_label=replacement,
                        text=turn.text,
                        start_time=turn.start_time,
                        end_time=turn.end_time,
                    )
            reassigned.append(turn)

        merged: List[TranscriptTurn] = []
        for turn in reassigned:
            if not merged:
                merged.append(turn)
                continue
            prev = merged[-1]
            if prev.speaker_label == turn.speaker_label:
                merged[-1] = TranscriptTurn(
                    speaker_label=prev.speaker_label,
                    text=self._merge_text(prev.text, turn.text),
                    start_time=prev.start_time,
                    end_time=turn.end_time,
                )
            else:
                merged.append(turn)
        return merged

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

    def _clean_label(self, label: str) -> str:
        label = self.SCORE_SUFFIX_PATTERN.sub("", label.strip())
        return " ".join(label.split())

    def _clean_text(self, text: str) -> str:
        text = " ".join(text.split())
        text = re.sub(r"\s+([.,!?])", r"\1", text)
        return text.strip()

    def _is_raw_diarization_label(self, label: str) -> bool:
        return bool(self.RAW_DIARIZATION_PATTERN.match(label.strip()))

    def _infer_nearby_named_speaker(self, turns: List[TranscriptTurn], index: int) -> str | None:
        current = turns[index]
        if len(current.text.split()) > 4:
            return None

        nearby_named: List[str] = []
        for offset in (1, 2):
            if index - offset >= 0:
                label = turns[index - offset].speaker_label
                if not self._is_raw_diarization_label(label):
                    nearby_named.append(label)
            if index + offset < len(turns):
                label = turns[index + offset].speaker_label
                if not self._is_raw_diarization_label(label):
                    nearby_named.append(label)

        if not nearby_named:
            return None
        if len(set(nearby_named)) == 1:
            return nearby_named[0]
        return nearby_named[0]

    def _merge_text(self, left: str, right: str) -> str:
        if not left:
            return right
        if not right:
            return left
        if left.endswith((".", "!", "?")):
            return f"{left} {right}"
        return f"{left} {right}"

    def _render_turns(self, turns: List[TranscriptTurn]) -> str:
        lines: List[str] = []
        for turn in turns:
            if turn.start_time is None or turn.end_time is None:
                continue
            lines.append(
                f"[{turn.start_time:.4f} - {turn.end_time:.4f}] -> {turn.speaker_label}: {turn.text}"
            )
        return "\n".join(lines) + ("\n" if lines else "")
