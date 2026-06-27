from abc import ABC, abstractmethod
from typing import List, Optional

from core.models import (
    ConversationSession,
    FusedContextEpisode,
    MemoryEvent,
    MemoryFact,
    MemoryReflection,
    OpenLoop,
    SpeakerRecord,
    TranscriptEvidence,
    UserProfileFacet,
    VisualObservation,
    VisualUserFact,
    VisualSession,
)


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

    @abstractmethod
    def append_evidence(self, evidence: TranscriptEvidence) -> None:
        pass

    @abstractmethod
    def get_recent_evidence(
        self,
        speaker_ids: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[TranscriptEvidence]:
        pass

    @abstractmethod
    def upsert_session(self, session: ConversationSession) -> ConversationSession:
        pass

    @abstractmethod
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        pass

    @abstractmethod
    def list_sessions(self, statuses: Optional[List[str]] = None, limit: int = 20) -> List[ConversationSession]:
        pass

    @abstractmethod
    def upsert_open_loop(self, loop: OpenLoop) -> OpenLoop:
        pass

    @abstractmethod
    def get_open_loop(self, loop_id: str) -> Optional[OpenLoop]:
        pass

    @abstractmethod
    def list_open_loops(self, statuses: Optional[List[str]] = None, limit: int = 20) -> List[OpenLoop]:
        pass

    @abstractmethod
    def upsert_profile_facet(self, facet: UserProfileFacet) -> UserProfileFacet:
        pass

    @abstractmethod
    def get_profile_facets(
        self,
        speaker_id: str,
        categories: Optional[List[str]] = None,
        statuses: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[UserProfileFacet]:
        pass

    @abstractmethod
    def save_session_digest(self, content: str) -> None:
        pass

    @abstractmethod
    def get_session_digest(self) -> str:
        pass

    @abstractmethod
    def save_open_loop_digest(self, content: str) -> None:
        pass

    @abstractmethod
    def get_open_loop_digest(self) -> str:
        pass

    @abstractmethod
    def append_visual_observation(self, observation: VisualObservation) -> None:
        pass

    @abstractmethod
    def get_visual_observation(self, observation_id: str) -> Optional[VisualObservation]:
        pass

    @abstractmethod
    def get_recent_visual_observations(self, limit: int = 10) -> List[VisualObservation]:
        pass

    @abstractmethod
    def get_recent_unsent_visual_observations(self, limit: int = 10) -> List[VisualObservation]:
        pass

    @abstractmethod
    def mark_visual_observations_followup_sent(
        self,
        observation_ids: List[str],
        sent_at: str,
    ) -> None:
        pass

    @abstractmethod
    def upsert_visual_session(self, session: VisualSession) -> VisualSession:
        pass

    @abstractmethod
    def get_visual_session(self, session_id: str) -> Optional[VisualSession]:
        pass

    @abstractmethod
    def list_visual_sessions(self, statuses: Optional[List[str]] = None, limit: int = 20) -> List[VisualSession]:
        pass

    @abstractmethod
    def save_visual_digest(self, content: str) -> None:
        pass

    @abstractmethod
    def get_visual_digest(self) -> str:
        pass

    @abstractmethod
    def upsert_fused_context_episode(self, episode: FusedContextEpisode) -> FusedContextEpisode:
        pass

    @abstractmethod
    def get_fused_context_episode(self, episode_id: str) -> Optional[FusedContextEpisode]:
        pass

    @abstractmethod
    def list_fused_context_episodes(
        self,
        statuses: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[FusedContextEpisode]:
        pass

    @abstractmethod
    def save_fused_context_digest(self, content: str) -> None:
        pass

    @abstractmethod
    def get_fused_context_digest(self) -> str:
        pass

    @abstractmethod
    def upsert_visual_user_fact(self, fact: VisualUserFact) -> VisualUserFact:
        pass

    @abstractmethod
    def get_visual_user_fact(self, fact_key: str) -> Optional[VisualUserFact]:
        pass

    @abstractmethod
    def list_visual_user_facts(
        self,
        statuses: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[VisualUserFact]:
        pass

    @abstractmethod
    def save_user_info(self, content: str) -> None:
        pass

    @abstractmethod
    def get_user_info(self) -> str:
        pass
