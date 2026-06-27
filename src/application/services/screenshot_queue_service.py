import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, Optional

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ScreenshotJob:
    screenshot_path: str
    captured_at: str


class ScreenshotQueueService:
    """In-memory bounded FIFO queue for deferred screenshot processing."""

    def __init__(
        self,
        maxlen: int = 180,
        *,
        ssim_threshold: float = 0.92,
        ssim_compare_count: int = 4,
        logger: logging.Logger | None = None,
    ):
        self.maxlen = maxlen
        self.ssim_threshold = max(0.0, min(1.0, ssim_threshold))
        self.ssim_compare_count = max(0, ssim_compare_count)
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._queue: Deque[ScreenshotJob] = deque()
        self._recent_images: Deque[np.ndarray] = deque(maxlen=self.ssim_compare_count)

    def enqueue(self, screenshot_path: str, captured_at: str | None = None) -> Optional[ScreenshotJob]:
        candidate = self._load_similarity_image(screenshot_path)
        if candidate is not None and self._is_too_similar(candidate):
            self.logger.info("Skipped queued screenshot because it is too similar to a recent capture: %s", screenshot_path)
            return None

        if len(self._queue) >= self.maxlen:
            dropped = self._queue.popleft()
            self.logger.info("Dropped oldest queued screenshot due to queue overflow: %s", dropped.screenshot_path)
        job = ScreenshotJob(
            screenshot_path=screenshot_path,
            captured_at=captured_at or datetime.now().isoformat(),
        )
        self._queue.append(job)
        if candidate is not None and self.ssim_compare_count > 0:
            self._recent_images.append(candidate)
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

    def _load_similarity_image(self, path: str) -> Optional[np.ndarray]:
        if self.ssim_compare_count <= 0:
            return None
        try:
            with Image.open(path) as image:
                image = image.convert("L").resize((128, 72))
                return np.asarray(image, dtype=np.float32)
        except Exception as exc:
            self.logger.debug("Failed to load screenshot for SSIM comparison %s: %s", path, exc)
            return None

    def _is_too_similar(self, candidate: np.ndarray) -> bool:
        if not self._recent_images:
            return False
        for previous in list(self._recent_images)[-self.ssim_compare_count:]:
            if self._ssim(candidate, previous) >= self.ssim_threshold:
                return True
        return False

    def _ssim(self, left: np.ndarray, right: np.ndarray) -> float:
        if left.shape != right.shape:
            return 0.0
        left_mean = float(left.mean())
        right_mean = float(right.mean())
        left_var = float(left.var())
        right_var = float(right.var())
        covariance = float(((left - left_mean) * (right - right_mean)).mean())
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        numerator = (2 * left_mean * right_mean + c1) * (2 * covariance + c2)
        denominator = (left_mean ** 2 + right_mean ** 2 + c1) * (left_var + right_var + c2)
        if denominator == 0:
            return 1.0
        return max(0.0, min(1.0, numerator / denominator))
