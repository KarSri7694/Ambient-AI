from abc import ABC, abstractmethod
from core.models import SpeakerEmbedding

class VoiceRepository(ABC):
    
    @abstractmethod
    def save_embedding(speaker_label: str, embedding: list[float]) -> None:
        """Save the speaker embedding to the repository."""
        pass
    
    @abstractmethod
    def get_all_embeddings() -> dict[str, list[float]]:
        """Retrieve all speaker embeddings from the repository."""
        pass
    
    @abstractmethod
    def store_embedding(speaker_embedding: SpeakerEmbedding) -> None:
        """Store the given speaker embedding in the repository."""
        pass
    