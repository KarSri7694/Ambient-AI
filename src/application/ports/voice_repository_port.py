from abc import ABC, abstractmethod
from core.models import SpeakerEmbedding

class VoiceRepository(ABC):
        
    @abstractmethod
    def get_all_embeddings() -> dict[str, list[float]]:
        """Retrieve all speaker embeddings from the repository."""
        pass
    
    @abstractmethod
    def store_embedding(speaker_embedding: SpeakerEmbedding) -> None:
        """Store the given speaker embedding in the repository."""
        pass
    