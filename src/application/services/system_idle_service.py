import ctypes
import logging
from ctypes import Structure, byref, c_uint, sizeof


class LASTINPUTINFO(Structure):
    _fields_ = [
        ("cbSize", c_uint),
        ("dwTime", c_uint),
    ]


class SystemIdleService:
    """Poll Windows input idle time using GetLastInputInfo."""

    def __init__(self, idle_threshold_seconds: int = 300, logger: logging.Logger | None = None):
        self.idle_threshold_seconds = idle_threshold_seconds
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def get_idle_seconds(self) -> float:
        user32 = getattr(ctypes, "windll", None)
        if user32 is None or not hasattr(user32, "user32"):
            return 0.0

        info = LASTINPUTINFO()
        info.cbSize = sizeof(LASTINPUTINFO)
        if not user32.user32.GetLastInputInfo(byref(info)):
            self.logger.warning("GetLastInputInfo failed; treating system as active.")
            return 0.0

        tick_count = user32.kernel32.GetTickCount()
        elapsed_ms = max(int(tick_count) - int(info.dwTime), 0)
        return elapsed_ms / 1000.0

    def is_user_idle(self, threshold_seconds: int | None = None) -> bool:
        threshold = self.idle_threshold_seconds if threshold_seconds is None else threshold_seconds
        return self.get_idle_seconds() >= threshold

    def is_user_active(self) -> bool:
        return not self.is_user_idle()
