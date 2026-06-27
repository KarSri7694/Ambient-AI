import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.todoist_helper import TodoistHelper


def test_todoist_helper_is_enabled_when_token_can_be_resolved(monkeypatch):
    monkeypatch.setattr(TodoistHelper, "_resolve_api_token", lambda self: "token-from-test")

    helper = TodoistHelper()

    assert helper.api_token == "token-from-test"
    assert helper.is_enabled() is True
