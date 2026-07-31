from __future__ import annotations

import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Iterator


class WorkInterrupted(RuntimeError):
    """Raised when the local user asks Ambient AI to skip the active work unit."""

    def __init__(self, reason: str = "Interrupted by local user"):
        super().__init__(reason)
        self.reason = reason


class RuntimeInterruptController:
    """Cooperative interrupt flag shared by runtime work units and the web UI."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._requested = False
        self._request_id = ""
        self._requested_at: str | None = None
        self._reason = ""
        self._active_work: dict[str, Any] | None = None
        self._last_interrupted: dict[str, Any] | None = None

    def request_interrupt(self, reason: str = "Interrupted by local user") -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._requested = True
            self._request_id = uuid.uuid4().hex
            self._requested_at = now
            self._reason = str(reason or "Interrupted by local user").strip()
            return self.status()

    def clear(self) -> None:
        with self._lock:
            self._requested = False
            self._request_id = ""
            self._requested_at = None
            self._reason = ""

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "requested": self._requested,
                "request_id": self._request_id,
                "requested_at": self._requested_at,
                "reason": self._reason,
                "active_work": dict(self._active_work) if self._active_work else None,
                "last_interrupted": (
                    dict(self._last_interrupted) if self._last_interrupted else None
                ),
            }

    def check(self) -> None:
        with self._lock:
            if not self._requested:
                return
            reason = self._reason or "Interrupted by local user"
        raise WorkInterrupted(reason)

    @contextmanager
    def active(self, **metadata: Any) -> Iterator[None]:
        started_at = datetime.now(timezone.utc).isoformat()
        work = {
            **{key: value for key, value in metadata.items() if value is not None},
            "started_at": started_at,
        }
        with self._lock:
            previous = self._active_work
            self._active_work = work
        try:
            yield
        except WorkInterrupted as exc:
            finished = {
                **work,
                "interrupted_at": datetime.now(timezone.utc).isoformat(),
                "reason": exc.reason,
            }
            with self._lock:
                self._last_interrupted = finished
                self._requested = False
                self._request_id = ""
                self._requested_at = None
                self._reason = ""
            raise
        finally:
            with self._lock:
                if self._active_work == work:
                    if self._requested:
                        self._last_interrupted = {
                            **work,
                            "interrupted_at": datetime.now(timezone.utc).isoformat(),
                            "reason": self._reason or "Interrupted by local user",
                        }
                        self._requested = False
                        self._request_id = ""
                        self._requested_at = None
                        self._reason = ""
                    self._active_work = previous


def make_interrupt_checker(
    controller: RuntimeInterruptController | None,
) -> Callable[[], None] | None:
    return controller.check if controller is not None else None
