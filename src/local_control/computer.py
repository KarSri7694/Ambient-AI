from __future__ import annotations

import ctypes
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
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
        "win", "command", "ctrl", "control", "alt", "shift",
    }

    def __init__(
        self,
        *,
        max_actions: int = 40,
        screenshot_dir: str | Path = ".ambient_data/computer/screenshots",
        read_only: bool = False,
    ):
        self.max_actions = max(1, int(max_actions))
        self.actions = 0
        self.screenshot_dir = Path(screenshot_dir)
        self.last_screenshot_path = ""
        self.read_only = bool(read_only)
        self.stop = EmergencyStopController()
        self.stop.start()
        self._allowed_tool_names = {
            "computer_inspect",
            "computer_move_mouse",
            "computer_click",
            "computer_double_click",
            "computer_right_click",
            "computer_drag",
            "computer_scroll",
            "computer_wait",
        }
        if not self.read_only:
            self._allowed_tool_names.update({"computer_type_text", "computer_press_key"})

    async def get_all_tools(self) -> list[dict[str, Any]]:
        tools = [
            self._tool(
                "computer_inspect",
                "Return current visual-observation metadata. A fresh screenshot is automatically attached to every computer-use model turn; use that screenshot for inspection instead of UI Automation.",
                {},
                [],
            ),
            self._tool(
                "computer_move_mouse",
                "Move the mouse using Gemma-normalized screen coordinates. x and y are integers from 0 to 1000, where (0,0) is top-left and (1000,1000) is bottom-right.",
                {"x": {"type": "integer"}, "y": {"type": "integer"}},
                ["x", "y"],
            ),
            self._tool(
                "computer_click",
                "Click using Gemma-normalized screen coordinates. x and y are integers from 0 to 1000, where (0,0) is top-left and (1000,1000) is bottom-right.",
                {"x": {"type": "integer"}, "y": {"type": "integer"}},
                ["x", "y"],
            ),
            self._tool(
                "computer_double_click",
                "Double-click using Gemma-normalized screen coordinates.",
                {"x": {"type": "integer"}, "y": {"type": "integer"}},
                ["x", "y"],
            ),
            self._tool(
                "computer_right_click",
                "Right-click using Gemma-normalized screen coordinates.",
                {"x": {"type": "integer"}, "y": {"type": "integer"}},
                ["x", "y"],
            ),
            self._tool(
                "computer_drag",
                "Drag from one Gemma-normalized coordinate to another.",
                {
                    "start_x": {"type": "integer"},
                    "start_y": {"type": "integer"},
                    "end_x": {"type": "integer"},
                    "end_y": {"type": "integer"},
                },
                ["start_x", "start_y", "end_x", "end_y"],
            ),
            self._tool("computer_scroll", "Scroll the active view by a bounded amount.", {"amount": {"type": "integer"}}, ["amount"]),
            self._tool("computer_wait", "Wait briefly for the desktop UI to settle.", {"seconds": {"type": "number"}}, []),
        ]
        if not self.read_only:
            tools.extend(
                [
                    self._tool("computer_type_text", "Type text into the currently focused field.", {"text": {"type": "string"}}, ["text"]),
                    self._tool("computer_press_key", "Press a single safe key.", {"key": {"type": "string"}}, ["key"]),
                ]
            )
        return tools

    async def execute_tool(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        self._before_action(tool_name)
        try:
            if tool_name == "computer_inspect":
                result = self._inspect()
            elif tool_name == "computer_move_mouse":
                x, y = self._scale_normalized_coordinate(tool_args.get("x"), tool_args.get("y"))
                result = self._pyautogui_call("moveTo", x, y)
            elif tool_name == "computer_click":
                x, y = self._scale_normalized_coordinate(tool_args.get("x"), tool_args.get("y"))
                result = self._pyautogui_call("click", x, y)
            elif tool_name == "computer_double_click":
                x, y = self._scale_normalized_coordinate(tool_args.get("x"), tool_args.get("y"))
                result = self._pyautogui_call("doubleClick", x, y)
            elif tool_name == "computer_right_click":
                x, y = self._scale_normalized_coordinate(tool_args.get("x"), tool_args.get("y"))
                result = self._pyautogui_call("rightClick", x, y)
            elif tool_name == "computer_drag":
                start_x, start_y = self._scale_normalized_coordinate(
                    tool_args.get("start_x"), tool_args.get("start_y")
                )
                end_x, end_y = self._scale_normalized_coordinate(
                    tool_args.get("end_x"), tool_args.get("end_y")
                )
                self._pyautogui_call("moveTo", start_x, start_y)
                result = self._pyautogui_call("dragTo", end_x, end_y, 0.25, button="left")
            elif tool_name == "computer_scroll":
                amount = max(-10, min(10, int(tool_args.get("amount"))))
                result = self._pyautogui_call("scroll", amount)
            elif tool_name == "computer_wait":
                seconds = max(0.0, min(10.0, float(tool_args.get("seconds") or 1.0)))
                time.sleep(seconds)
                result = f"Waited {seconds:.1f} seconds."
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

    def _scale_normalized_coordinate(self, x_value: Any, y_value: Any) -> tuple[int, int]:
        width, height = self._screen_size()
        x_norm = max(0, min(1000, int(x_value)))
        y_norm = max(0, min(1000, int(y_value)))
        x = round(x_norm / 1000 * max(0, width - 1))
        y = round(y_norm / 1000 * max(0, height - 1))
        return x, y

    @staticmethod
    def _screen_size() -> tuple[int, int]:
        try:
            import pyautogui
            size = pyautogui.size()
        except Exception as exc:
            raise UnsafeComputerAction(f"Cannot resolve screen size for normalized coordinates: {exc}") from exc
        width = getattr(size, "width", None)
        height = getattr(size, "height", None)
        if width is None or height is None:
            width, height = size
        width = int(width)
        height = int(height)
        if width <= 0 or height <= 0:
            raise UnsafeComputerAction(f"Invalid screen size reported by pyautogui: {width}x{height}")
        return width, height

    def _inspect(self) -> str:
        width = height = None
        try:
            width, height = self._screen_size()
        except Exception:
            pass
        return json.dumps(
            {
                "status": "ok",
                "observation": "fresh_screenshot_attached_each_model_turn",
                "latest_screenshot_path": self.last_screenshot_path,
                "coordinate_system": "Use normalized 0..1000 x/y coordinates for all mouse actions.",
                "screen_size": {"width": width, "height": height},
            },
            ensure_ascii=False,
        )

    def capture_screenshot_for_model(self) -> str:
        self.stop.check()
        try:
            import pyautogui
        except Exception as exc:
            return f""
        try:
            self.screenshot_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            path = self.screenshot_dir / f"computer-{timestamp}-{uuid.uuid4().hex[:8]}.png"
            image = pyautogui.screenshot()
            image.save(path)
            self.last_screenshot_path = str(path)
            return str(path)
        except Exception:
            return ""

    @staticmethod
    def _pyautogui_call(method_name: str, *args: Any, **kwargs: Any) -> str:
        try:
            import pyautogui
        except Exception as exc:
            return f"Error: pyautogui is unavailable: {exc}"
        getattr(pyautogui, method_name)(*args, **kwargs)
        time.sleep(0.05)
        return f"Executed {method_name}."
