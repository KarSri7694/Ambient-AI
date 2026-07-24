from abc import ABC, abstractmethod

from core.models import ResourceSnapshot


class ResourceMonitorPort(ABC):
    """Read-only host telemetry used by the inference resource governor."""

    @abstractmethod
    def snapshot(self, *, user_idle: bool = False, force: bool = False) -> ResourceSnapshot:
        pass
