from typing import List

import night_mode
from application.ports.notification_port import NotificationPort
from core.models import Notification


class SQLiteNotificationAdapter(NotificationPort):
    """Adapter wrapping night_mode notification functions behind NotificationPort."""

    def get_unread_notifications(self) -> List[Notification]:
        rows = night_mode.get_unread_notifications()
        return [
            Notification(
                id=row["id"],
                message=row["message"],
                source=row.get("source", "system"),
            )
            for row in rows
        ]

    def add_notification(self, message: str, source: str = "system") -> None:
        night_mode.add_notification(message, source)

    def mark_read(self, notification_id: int) -> None:
        night_mode.mark_notification_read(notification_id)
