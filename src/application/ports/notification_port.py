from abc import ABC, abstractmethod
from typing import List
from core.models import Notification


class NotificationPort(ABC):
    """Port for system notification operations."""

    @abstractmethod
    def get_unread_notifications(self) -> List[Notification]:
        """Retrieve all unread notifications and mark them as read."""
        pass

    @abstractmethod
    def add_notification(self, message: str, source: str = "system") -> None:
        """Add a new system notification."""
        pass

    @abstractmethod
    def mark_read(self, notification_id: int) -> None:
        """Mark a specific notification as read."""
        pass
