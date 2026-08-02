"""Best-effort Windows toast delivery with no effect on durable Inbox storage."""

from __future__ import annotations

import logging


class WindowsToastNotifier:
    def __init__(self, app_id: str = "Ambient AI") -> None:
        self.app_id = app_id
        self.logger = logging.getLogger(self.__class__.__name__)

    def notify(self, title: str, message: str) -> bool:
        try:
            from winotify import Notification, audio  # type: ignore
        except ImportError:
            self.logger.info("winotify is unavailable; recurring notification remains in Ambient Inbox.")
            return False
        try:
            toast = Notification(app_id=self.app_id, title=str(title)[:120], msg=str(message)[:240])
            toast.set_audio(audio.Default, loop=False)
            toast.show()
            return True
        except Exception:
            self.logger.exception("Windows toast delivery failed")
            return False
