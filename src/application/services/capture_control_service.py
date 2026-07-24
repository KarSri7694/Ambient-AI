import threading
from datetime import datetime, timezone


class CaptureControlService:
    """Thread-safe global capture pause/kill switch."""

    def __init__(self, *, excluded_apps=None, excluded_domains=None):
        self._paused = threading.Event()
        self._lock = threading.RLock()
        self._excluded_apps = self._normalize(excluded_apps or [])
        self._excluded_domains = self._normalize(excluded_domains or [])
        self._updated_at = datetime.now(timezone.utc).isoformat()

    def pause(self) -> None:
        self._paused.set()
        self._updated_at = datetime.now(timezone.utc).isoformat()

    def resume(self) -> None:
        self._paused.clear()
        self._updated_at = datetime.now(timezone.utc).isoformat()

    def is_paused(self) -> bool:
        return self._paused.is_set()

    def set_exclusions(self, *, apps=None, domains=None) -> None:
        with self._lock:
            if apps is not None:
                self._excluded_apps = self._normalize(apps)
            if domains is not None:
                self._excluded_domains = self._normalize(domains)
            self._updated_at = datetime.now(timezone.utc).isoformat()

    def is_excluded(self, *, app_name: str = "", domain: str = "") -> bool:
        app = str(app_name or "").strip().lower()
        host = str(domain or "").strip().lower()
        with self._lock:
            return any(value in app for value in self._excluded_apps) or any(
                host == value or host.endswith(f".{value}") for value in self._excluded_domains
            )

    def status(self) -> dict:
        with self._lock:
            return {
                "paused": self.is_paused(),
                "excluded_apps": sorted(self._excluded_apps),
                "excluded_domains": sorted(self._excluded_domains),
                "updated_at": self._updated_at,
            }

    @staticmethod
    def _normalize(values) -> set[str]:
        return {str(value).strip().lower() for value in values if str(value).strip()}
