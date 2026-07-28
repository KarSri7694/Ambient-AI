from __future__ import annotations


class LocalControlError(RuntimeError):
    """Base error for local-control failures."""


class PathGrantError(PermissionError):
    """Raised when a filesystem tool attempts to escape its grants."""


class ComputerControlTerminated(LocalControlError):
    """Raised when the user terminates the active computer-use session."""


class UnsafeComputerAction(LocalControlError):
    """Raised when a requested desktop action is outside the safe subset."""

