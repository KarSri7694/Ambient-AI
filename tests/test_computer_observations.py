import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from local_control.computer import ComputerControlSession


def _uiat_payload(count=80):
    return {
        "ok": True,
        "window_title": "Editor - Important document",
        "window_class": "EditorWindow",
        "process_name": "editor.exe",
        "foreground_url": None,
        "contains_dialog": True,
        "contains_notification": False,
        "visible_items": [
            {
                "name": "Save document",
                "control_type": "ButtonControl",
                "enabled": True,
                "editable": False,
                "bbox": [10, 20, 120, 60],
                "value": "Save",
            },
            *[
                {
                    "name": f"Very long visible control {index} " + ("x" * 300),
                    "control_type": "TextControl",
                    "enabled": True,
                    "editable": False,
                    "bbox": [index, index, index + 100, index + 30],
                    "value": "",
                }
                for index in range(count)
            ],
        ],
    }


def test_computer_inspect_bounds_uiat_and_keeps_actionable_control():
    session = ComputerControlSession(
        max_actions=10,
        uiat_max_items=4,
        uiat_max_chars=900,
        uiat_name_max_chars=48,
        uiat_inspector=lambda: _uiat_payload(),
    )
    try:
        result = json.loads(asyncio.run(session.execute_tool("computer_inspect", {})))
    finally:
        asyncio.run(session.cleanup())

    uiat = result["ui_automation"]
    assert len(json.dumps(uiat, ensure_ascii=False, separators=(",", ":"))) <= 900
    assert uiat["truncated"] is True
    assert uiat["items"][0]["name"] == "Save document"
    assert len(uiat["items"][0]["name"]) <= 48
    assert result["progress"]["actions_used"] == 1


def test_computer_inspect_degrades_when_uia_is_unavailable():
    session = ComputerControlSession(
        uiat_inspector=lambda: (_ for _ in ()).throw(RuntimeError("not on Windows")),
    )
    try:
        result = json.loads(asyncio.run(session.execute_tool("computer_inspect", {})))
    finally:
        asyncio.run(session.cleanup())

    assert result["status"] == "ok"
    assert result["ui_automation"]["ok"] is False
    assert result["ui_automation"]["error"] == "uiat_unavailable"


def test_password_values_are_not_returned_to_model_context():
    payload = _uiat_payload(0)
    payload["visible_items"][0].update({"name": "Password", "value": "secret", "is_password": True})
    session = ComputerControlSession(uiat_inspector=lambda: payload)
    try:
        result = json.loads(asyncio.run(session.execute_tool("computer_inspect", {})))
    finally:
        asyncio.run(session.cleanup())

    item = result["ui_automation"]["items"][0]
    assert "secret" not in json.dumps(result)
    assert item["value_redacted"] is True


def test_inspect_is_on_demand_and_action_turns_remain_screenshot_driven():
    calls = []
    session = ComputerControlSession(uiat_inspector=lambda: calls.append(True) or _uiat_payload(1))
    try:
        asyncio.run(session.execute_tool("computer_wait", {"seconds": 0}))
        assert calls == []
        inspect_result = json.loads(asyncio.run(session.execute_tool("computer_inspect", {})))
    finally:
        asyncio.run(session.cleanup())

    assert calls == [True]
    assert inspect_result["observation"] == "fresh_screenshot_attached_each_model_turn"
