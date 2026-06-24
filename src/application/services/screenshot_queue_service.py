import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, Optional


@dataclass(frozen=True)
class ScreenshotJob:
    screenshot_path: str
    captured_at: str


class ScreenshotQueueService:
    """In-memory bounded FIFO queue for deferred screenshot processing."""

    def __init__(self, maxlen: int = 180, logger: logging.Logger | None = None):
        self.maxlen = maxlen
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._queue: Deque[ScreenshotJob] = deque()

    def enqueue(self, screenshot_path: str, captured_at: str | None = None) -> ScreenshotJob:
        if len(self._queue) >= self.maxlen:
            dropped = self._queue.popleft()
            self._remove_file(dropped.screenshot_path)
            self.logger.info("Dropped oldest queued screenshot due to queue overflow: %s", dropped.screenshot_path)
        job = ScreenshotJob(
            screenshot_path=screenshot_path,
            captured_at=captured_at or datetime.now().isoformat(),
        )
        self._queue.append(job)
        return job

    def dequeue(self) -> Optional[ScreenshotJob]:
        if not self._queue:
            return None
        return self._queue.popleft()

    def size(self) -> int:
        return len(self._queue)

    def is_empty(self) -> bool:
        return not self._queue

    def peek_oldest(self) -> Optional[ScreenshotJob]:
        return self._queue[0] if self._queue else None

    def _remove_file(self, path: str) -> None:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            self.logger.debug("Failed to remove dropped screenshot file %s", path)
