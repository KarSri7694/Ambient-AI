from abc import ABC, abstractmethod
from typing import List, Optional

from core.models import AmbientAgendaItem


class AmbientAgendaPort(ABC):
    """Port for durable ambient agenda storage."""

    @abstractmethod
    def create_item(self, item: AmbientAgendaItem) -> AmbientAgendaItem:
        pass

    @abstractmethod
    def update_item(self, item: AmbientAgendaItem) -> AmbientAgendaItem:
        pass

    @abstractmethod
    def get_item(self, agenda_id: str) -> Optional[AmbientAgendaItem]:
        pass

    @abstractmethod
    def list_items(
        self,
        statuses: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[AmbientAgendaItem]:
        pass

    @abstractmethod
    def find_by_source(
        self,
        source_type: str,
        source_ref: str,
        kind: Optional[str] = None,
        statuses: Optional[List[str]] = None,
    ) -> Optional[AmbientAgendaItem]:
        pass
