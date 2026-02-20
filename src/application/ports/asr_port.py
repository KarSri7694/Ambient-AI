from abc import ABC, abstractmethod
from core.models import TranscriptionResult, DiarizationResult
from typing import List

class TranscriptionPort(ABC):
    @abstractmethod
    def transcribe_audio(self, audio_file_path: str, vad_filter: bool, word_timestamps: bool) -> List[TranscriptionResult]:
        """Transcribe the given audio file and return a list of TranscriptionResult objects."""
        pass

class DiarizationPort(ABC):
    @abstractmethod
    def diarize_audio(self, audio_file_path: str) -> List[DiarizationResult]:
        """Diarize the given audio file and return a list of DiarizationResult objects with speaker labels."""
        pass

