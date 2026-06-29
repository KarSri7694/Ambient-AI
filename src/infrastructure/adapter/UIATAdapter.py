import logging
from typing import Any, Dict


class UIATAdapter:
    """Thin wrapper around utils.UIAT with graceful failure handling."""

    def __init__(self, mode: str = "screen_content", logger: logging.Logger | None = None):
        self.mode = mode
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def inspect_foreground_window(self) -> Dict[str, Any]:
        try:
            from utils import UIAT

            payload = UIAT.inspect_foreground_window(mode=self.mode)
            if isinstance(payload, dict):
                self.logger.debug(
                    "UIAT inspect mode=%s ok=%s window=%r items=%s",
                    self.mode,
                    payload.get("ok"),
                    payload.get("window_title"),
                    len(payload.get("visible_items") or []),
                )
                return payload
        except Exception as exc:
            self.logger.debug("UIAT inspection failed: %s", exc)
        return {
            "ok": False,
            "mode": self.mode,
            "error": "uiat_unavailable",
            "window_title": None,
            "window_class": None,
            "process_id": None,
            "is_chromium": False,
            "visible_items": [],
            "visible_text_summary": "",
            "contains_dialog": False,
            "contains_notification": False,
            "contains_editable_fields": False,
        }
