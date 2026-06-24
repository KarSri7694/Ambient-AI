from abc import ABC, abstractmethod
from typing import List, Optional

from core.models import MemoryEvent, MemoryFact, MemoryReflection, SpeakerRecord


class MemoryPort(ABC):
    """Port for durable memory storage and prompt-facing memory context."""

    @abstractmethod
    def upsert_speaker(
        self,
        display_name: str,
        source_label: str,
        voice_embedding_uid: Optional[int] = None,
        is_user: bool = False,
        speaker_id: Optional[str] = None,
    ) -> SpeakerRecord:
        pass

    @abstractmethod
    def get_speaker(self, speaker_id: str) -> Optional[SpeakerRecord]:
        pass

    @abstractmethod
    def find_speaker_by_display_name(self, display_name: str) -> Optional[SpeakerRecord]:
        pass

    @abstractmethod
    def list_speakers(self) -> List[SpeakerRecord]:
        pass

    @abstractmethod
    def append_event(self, event: MemoryEvent) -> None:
        pass

    @abstractmethod
    def get_recent_events(
        self,
        speaker_ids: Optional[List[str]] = None,
        limit: int = 10,
        status: Optional[str] = None,
    ) -> List[MemoryEvent]:
        pass

    @abstractmethod
    def get_pending_consolidation(self, limit: int = 100) -> List[MemoryEvent]:
        pass

    @abstractmethod
    def upsert_fact(self, fact: MemoryFact) -> MemoryFact:
        pass

    @abstractmethod
    def get_facts(self, speaker_id: str) -> List[MemoryFact]:
        pass

    @abstractmethod
    def mark_events_consolidated(self, event_ids: List[str], consolidated_at: str) -> None:
        pass

    @abstractmethod
    def add_reflection(self, reflection: MemoryReflection) -> None:
        pass

    @abstractmethod
    def get_speaker_profile(self, speaker_id: str) -> str:
        pass

    @abstractmethod
    def save_speaker_profile(self, speaker_id: str, content: str) -> None:
        pass

    @abstractmethod
    def get_recent_context(self) -> str:
        pass

    @abstractmethod
    def save_recent_context(self, content: str) -> None:
        pass
