from abc import ABC, abstractmethod
from typing import List, Optional

from core.models import ProactiveTopicCandidate


class ProactiveTopicQueuePort(ABC):
    """Port for the proactive research backlog."""

    @abstractmethod
    def upsert_topic(self, topic: ProactiveTopicCandidate) -> ProactiveTopicCandidate:
        pass

    @abstractmethod
    def get_topic(self, topic_id: str) -> Optional[ProactiveTopicCandidate]:
        pass

    @abstractmethod
    def find_by_normalized_topic(self, normalized_topic: str) -> Optional[ProactiveTopicCandidate]:
        pass

    @abstractmethod
    def get_pending_topics(self, limit: int = 20) -> List[ProactiveTopicCandidate]:
        pass

    @abstractmethod
    def list_topics(
        self,
        statuses: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[ProactiveTopicCandidate]:
        pass

    @abstractmethod
    def mark_topic_status(
        self,
        topic_id: str,
        status: str,
        artifact_path: Optional[str] = None,
        last_researched_at: Optional[str] = None,
    ) -> None:
        pass
