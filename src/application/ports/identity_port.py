from abc import ABC, abstractmethod
from core.models import AudioMetadata, DiarizationResult

class SpeakerIdentityPort(ABC):
    @abstractmethod
    def identify_speaker(self, audio_metadata: AudioMetadata, diarization_result: DiarizationResult, threshold: float) -> str:
        """Identify the speaker from the given audio metadata and return the speaker label."""
        pass 
    
    @abstractmethod
    def create_speaker_embedding(self, audio_file_path: str) :
        """Create a speaker embedding from the given audio file and return the audio metadata with embedding."""
        pass
