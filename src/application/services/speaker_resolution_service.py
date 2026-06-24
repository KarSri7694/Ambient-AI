import re
from typing import Dict, Iterable

from application.ports.memory_port import MemoryPort
from core.models import TranscriptParticipant


class SpeakerResolutionService:
    """Resolve transcript speaker labels into durable memory identities."""

    SPEAKER_SCORE_PATTERN = re.compile(r"^(?P<label>.+?)\s*-\s*\[(?P<score>\d+(?:\.\d+)?)%\]$")
    RAW_DIARIZATION_PATTERN = re.compile(r"^speaker[_\s-]*\d+$", re.IGNORECASE)

    def __init__(self, memory: MemoryPort):
        self.memory = memory

    def resolve_labels(
        self,
        speaker_labels: Iterable[str],
        source_ref: str,
    ) -> Dict[str, TranscriptParticipant]:
        resolved: Dict[str, TranscriptParticipant] = {}
        for raw_label in speaker_labels:
            normalized_label, confidence = self._normalize_label(raw_label)
            speaker = None

            if not self._is_raw_diarization_label(normalized_label):
                speaker = self.memory.find_speaker_by_display_name(normalized_label)

            if speaker is None:
                display_name = normalized_label
                if self._is_raw_diarization_label(display_name):
                    display_name = f"{display_name} @ {source_ref}"
                speaker = self.memory.upsert_speaker(
                    display_name=display_name,
                    source_label=raw_label,
                    is_user=self._is_user_label(normalized_label),
                )

            resolved[raw_label] = TranscriptParticipant(
                speaker_label=raw_label,
                speaker_id=speaker.speaker_id,
                display_name=speaker.display_name,
                confidence=confidence,
            )
        return resolved

    def _normalize_label(self, raw_label: str) -> tuple[str, float]:
        match = self.SPEAKER_SCORE_PATTERN.match(raw_label.strip())
        if not match:
            return raw_label.strip(), 0.0
        return match.group("label").strip(), float(match.group("score")) / 100.0

    def _is_raw_diarization_label(self, label: str) -> bool:
        return bool(self.RAW_DIARIZATION_PATTERN.match(label.strip()))

    def _is_user_label(self, label: str) -> bool:
        normalized = label.strip().upper()
        return normalized in {"USER", "ME"}
