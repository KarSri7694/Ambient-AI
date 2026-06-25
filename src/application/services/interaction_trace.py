from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict


_SOURCE: ContextVar[str] = ContextVar("interaction_source", default="unknown")
_METADATA: ContextVar[Dict[str, Any]] = ContextVar("interaction_metadata", default={})


@contextmanager
def interaction_trace(source: str, metadata: Dict[str, Any] | None = None):
    current_metadata = dict(_METADATA.get())
    merged_metadata = dict(current_metadata)
    merged_metadata.update(dict(metadata or {}))
    source_token = _SOURCE.set(source)
    metadata_token = _METADATA.set(merged_metadata)
    try:
        yield
    finally:
        _SOURCE.reset(source_token)
        _METADATA.reset(metadata_token)


def current_interaction_source() -> str:
    return _SOURCE.get()


def current_interaction_metadata() -> Dict[str, Any]:
    return dict(_METADATA.get())
