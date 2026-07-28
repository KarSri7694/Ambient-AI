from __future__ import annotations

import ctypes
import threading
import time
from typing import Any

from local_control.safety import ComputerControlTerminated, UnsafeComputerAction


class EmergencyStopController:
    VK_SHIFT = 0x10
    VK_ESCAPE = 0x1B

    def __init__(self, *, poll_interval_seconds: float = 0.05):
        self.poll_interval_seconds = max(0.01, float(poll_interval_seconds))
        self._terminated = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._poll, name="ComputerUseEmergencyStop", daemon=True)
        self._thread.start()

    def terminate(self) -> None:
        self._terminated.set()

    def check(self) -> None:
        if self._terminated.is_set():
            raise ComputerControlTerminated("Computer-use session was terminated by Shift+Esc.")

    def cleanup(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.25)
        self._thread = None

    def _poll(self) -> None:
        while not self._stop.wait(self.poll_interval_seconds):
            try:
                shift_pressed = bool(ctypes.windll.user32.GetAsyncKeyState(self.VK_SHIFT) & 0x8000)
                esc_pressed = bool(ctypes.windll.user32.GetAsyncKeyState(self.VK_ESCAPE) & 0x8000)
            except Exception:
                return
            if shift_pressed and esc_pressed:
                self._terminated.set()
                return


class ComputerControlSession:
    BLOCKED_KEYS = {
        "alt+f4",
        "ctrl+alt+delete",
        "win+l",
        "win+r",
        "ctrl+shift+esc",
    }
    SAFE_KEYS = {
        "enter", "tab", "esc", "escape", "backspace", "delete", "space",
        "left", "right", "up", "down", "home", "end", "pageup", "pagedown",
    }

    def __init__(self, *, max_actions: int = 40):
        self.max_actions = max(1, int(max_actions))
        self.actions = 0
        self.stop = EmergencyStopController()
        self.stop.start()
        self._allowed_tool_names = {
            "computer_inspect",
            "computer_move_mouse",
            "computer_click",
            "computer_scroll",
            "computer_type_text",
            "computer_press_key",
        }

    async def get_all_tools(self) -> list[dict[str, Any]]:
        return [
            self._tool("computer_inspect", "Inspect the active foreground window using UI Automation.", {}, []),
            self._tool("computer_move_mouse", "Move the mouse to absolute screen coordinates.", {"x": {"type": "integer"}, "y": {"type": "integer"}}, ["x", "y"]),
            self._tool("computer_click", "Click at absolute screen coordinates.", {"x": {"type": "integer"}, "y": {"type": "integer"}}, ["x", "y"]),
            self._tool("computer_scroll", "Scroll the active view by a bounded amount.", {"amount": {"type": "integer"}}, ["amount"]),
            self._tool("computer_type_text", "Type text into the currently focused field.", {"text": {"type": "string"}}, ["text"]),
            self._tool("computer_press_key", "Press a single safe key.", {"key": {"type": "string"}}, ["key"]),
        ]

    async def execute_tool(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        self._before_action(tool_name)
        try:
            if tool_name == "computer_inspect":
                result = self._inspect()
            elif tool_name == "computer_move_mouse":
                result = self._pyautogui_call("moveTo", int(tool_args.get("x")), int(tool_args.get("y")))
            elif tool_name == "computer_click":
                result = self._pyautogui_call("click", int(tool_args.get("x")), int(tool_args.get("y")))
            elif tool_name == "computer_scroll":
                amount = max(-10, min(10, int(tool_args.get("amount"))))
                result = self._pyautogui_call("scroll", amount)
            elif tool_name == "computer_type_text":
                text = str(tool_args.get("text") or "")
                if len(text) > 1000:
                    raise UnsafeComputerAction("Typed text is capped at 1000 characters.")
                result = self._pyautogui_call("write", text)
            elif tool_name == "computer_press_key":
                key = str(tool_args.get("key") or "").strip().lower()
                if "+" in key or key not in self.SAFE_KEYS or key in self.BLOCKED_KEYS:
                    raise UnsafeComputerAction(f"Key is not in the safe key set: {key}")
                result = self._pyautogui_call("press", key)
            else:
                raise ValueError(f"Computer tool '{tool_name}' is not allowed.")
        finally:
            self.stop.check()
        return result

    async def cleanup(self) -> None:
        self.stop.cleanup()

    @staticmethod
    def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }

    def _before_action(self, tool_name: str) -> None:
        self.stop.check()
        if tool_name not in self._allowed_tool_names:
            raise ValueError(f"Computer tool '{tool_name}' is not allowed.")
        if self.actions >= self.max_actions:
            raise UnsafeComputerAction("Computer-use action limit reached.")
        self.actions += 1

    def _inspect(self) -> str:
        try:
            from infrastructure.adapter.UIATAdapter import UIATAdapter

            return UIATAdapter(mode="screen_content").inspect_foreground_window().__repr__()
        except Exception as exc:
            return f"Error: foreground inspection failed: {exc}"

    @staticmethod
    def _pyautogui_call(method_name: str, *args: Any) -> str:
        try:
            import pyautogui
        except Exception as exc:
            return f"Error: pyautogui is unavailable: {exc}"
        getattr(pyautogui, method_name)(*args)
        time.sleep(0.05)
        return f"Executed {method_name}."
