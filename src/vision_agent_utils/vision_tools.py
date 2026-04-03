import pyautogui
from fastmcp import FastMCP
from typing import Annotated
import logging
import ast
import time
import cv2
import numpy as np
from mss import mss
from skimage.metrics import structural_similarity as ssim 
try:
    from ..utils.OSController import OSController
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from utils.OSController import OSController

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

mcp = FastMCP()

#For Qwen3-VL
RELATIVE_WIDTH = 1000
RELATIVE_HEIGHT = 1000
STEP_TREE = ''

class StepTree:
    step_counter = 0
    def __init__(self):
        self.tree = ""
        self.step_counter = 0

    def add_step(self, step: str):
        self.tree += f"Step: {self.step_counter}: {step}\n"
        self.step_counter += 1

    def reset_tree(self):
        self.tree = ""

    def get_tree(self) -> str:
        return self.tree

tree = StepTree()
#Helper function to convert coordinates
def convert_to_original_coordinates(x: int, y: int, current_width: int, current_height: int, original_width: int, original_height: int):
    '''
    Convert coordinates from the current image size back to the original image size.

    Args:
        x: X coordinate in the current image size.
        y: Y coordinate in the current image size.
        current_width: total width of current coordinate system.
        current_height: total height of current coordinate system.
        original_width: Width of the original coordinate system.
        original_height: Height of the original coordinate system.
    '''
    original_x = int(x * original_width / current_width)
    original_y = int(y * original_height / current_height)
    logging.info(f"Converted coordinates ({x}, {y}) from current size ({current_width}x{current_height}) to original size ({original_width}x{original_height}): ({original_x}, {original_y})")
    return original_x, original_y

#mouse Tools
@mcp.tool
def mouse_click(x: Annotated[int, "mouse position X coordinate"], 
                y: Annotated[int, "mouse position Y coordinate"],
                area_clicked: Annotated[str, "description of the area clicked, e.g., 'start button', 'text field', 'close icon'"]
                ):
    '''
    Simulate a mouse click at the current cursor position.
    '''
    x_new,y_new = convert_to_original_coordinates(x, y, RELATIVE_WIDTH, RELATIVE_HEIGHT, pyautogui.size().width, pyautogui.size().height)
    pyautogui.click(x_new, y_new)
    logging.info(f"Clicked at ({x}, {y}) in area: {area_clicked}, original coordinates: ({x_new}, {y_new})")
    tree.add_step(f"Clicked at ({x}, {y}) in area: {area_clicked}")
    return f"Clicked at ({x}, {y}) in area: {area_clicked}"
    
@mcp.tool
def mouse_double_click(x: Annotated[int, "mouse position X coordinate"], 
                       y: Annotated[int, "mouse position Y coordinate"]):
    '''
    Simulate a mouse double click at the current cursor position.
    '''
    x_new,y_new = convert_to_original_coordinates(x, y, RELATIVE_WIDTH, RELATIVE_HEIGHT, pyautogui.size().width, pyautogui.size().height)
    pyautogui.doubleClick(x_new, y_new)
    tree.add_step(f"Double clicked at ({x}, {y})")
    return f"Double clicked at ({x}, {y})."
    
@mcp.tool
def mouse_right_click(x: Annotated[int, "mouse position X coordinate"], 
                      y: Annotated[int, "mouse position Y coordinate"]):
    '''
    Simulate a mouse right click at the current cursor position.
    '''
    x_new,y_new = convert_to_original_coordinates(x, y, RELATIVE_WIDTH, RELATIVE_HEIGHT, pyautogui.size().width, pyautogui.size().height)
    pyautogui.rightClick(x_new, y_new)
    tree.add_step(f"Right clicked at ({x}, {y})")
    return f"Right clicked at ({x}, {y})."

@mcp.tool
def move_mouse(x: Annotated[int, "mouse position X coordinate"], 
               y: Annotated[int, "mouse position Y coordinate"]):
    '''
    Move the mouse cursor to a specific position on the screen.
    '''
    x_new,y_new = convert_to_original_coordinates(x, y, RELATIVE_WIDTH, RELATIVE_HEIGHT, pyautogui.size().width, pyautogui.size().height)
    pyautogui.moveTo(x_new, y_new)
    tree.add_step(f"Moved mouse to ({x}, {y})")
    return f"Moved mouse to ({x}, {y})."

@mcp.tool
def scroll(amount: Annotated[int, "amount to scroll (positive for up, negative for down)"]):
    '''
    Scroll the mouse wheel up or down.
    Positive amount scrolls up, negative scrolls down.
    '''
    pyautogui.scroll(amount)
    tree.add_step(f"Scrolled {'up' if amount > 0 else 'down'} by {abs(amount)} units.")
    return f"Scrolled {'up' if amount > 0 else 'down'} by {abs(amount)} units."


@mcp.tool
def input_text(text: Annotated[str, "text to be typed"]):
    '''
    Type a string of text at the current cursor location.
    '''
    pyautogui.typewrite(text)
    tree.add_step(f"Typed text: {text}")
    return f"Typed text: {text}"
    
@mcp.tool
def press_key(key: Annotated[str, "key  to press (e.g., 'enter' "]):
    '''
    Press a specific special key or hotkey combination.
    '''
    pyautogui.press(key)
    tree.add_step(f"Pressed key: {key}")
    return f"Pressed key: {key}"

@mcp.tool
def keyboard_hotkey(keys: Annotated[str, "key to press eg. 'ctrl+c'"]):
    '''
    Press a specific hotkey combination.
    '''
    keys_split = keys.split('+')
    pyautogui.hotkey(*keys_split)
    tree.add_step(f"Pressed hotkey combination: {keys}")
    return f"Pressed hotkey combination: {keys}"

@mcp.tool
def open_app(app_name: Annotated[str, "name of the application to open"]):
    '''
    Open the specified application, if it is not already running it opens it, if it is already running it focuses it.
    '''
    controller = OSController()
    controller._set_foreground_lock(0)
    
    if controller.focus_existing_window(app_name):
        print(f"SUCCESS: Found existing window for '{app_name}' and focused it.")
        tree.add_step(f"Opened existing application: {app_name}")
        return f"Opened existing application: {app_name}"
    else:
        result = controller.open_and_verify_app(app_name)
        if result == "FAILURE: Could not open application. Application name may be incorrect.":
            return "FAILURE: Could not open application. Application name may be incorrect."
        tree.add_step(f"Opened new application: {app_name}")
        return f"Opened new application: {app_name}"
        
def _get_screenshot():
    with mss() as sct:
        monitor = sct.monitors[1]
        sct_img = sct.grab(monitor)
        frame = np.array(sct_img)
        gray_scaled = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    return gray_scaled


@mcp.tool
def tool_chain(actions: Annotated[str, "List of tool calls with arguments to execute in sequence."]):
    '''
    Run multiple tools in one call.

    Prefer this tool when the task is a short deterministic sequence and
    intermediate screenshot checks are not required.

    Use this only when the screen is expected to stay stable across steps.
    Keep chains short: 2 to 4 actions.
    Each action must be a dict with:
    - "function": tool name
    - "args": dict of arguments
    Never include "tool_chain" inside the chain.
    If a chain fails, execute the same steps one by one.

        Input format:
        - list of action dicts

        Expected action shape:
    [
      {"function": "move_mouse", "args": {"x": 100, "y": 200}},
      {"function": "mouse_click", "args": {"x": 100, "y": 200, "area_clicked": "search box"}},
      {"function": "input_text", "args": {"text": "weather today"}},
      {"function": "press_key", "args": {"key": "enter"}}
    ]
    
    Example usecases:
    1. Web search flow: click address bar, type URL, press Enter.
       Chain: mouse_click -> input_text -> press_key.
    2. Form submit flow: click name field, type name, press Tab, type email, press Enter.
       Chain: mouse_click -> input_text -> press_key -> input_text -> press_key.
    3. File explorer flow: click folder, double-click file, press Ctrl+C.
       Chain: mouse_click -> mouse_double_click -> keyboard_hotkey.
    '''
    actions_json = ast.literal_eval(actions)
    def _resolve_callable(tool_name: str):
        '''Resolve a tool name to a directly callable Python function.

        FastMCP's @mcp.tool decorator can replace functions with FunctionTool
        wrapper objects, so we fall back to common underlying function attrs.
        '''
        if tool_name == "tool_chain":
            raise ValueError("tool_chain cannot call itself recursively.")
        tool_obj = globals().get(tool_name)
        if tool_obj is None:
            raise ValueError(f"Tool '{tool_name}' not found in module globals.")

        if callable(tool_obj):
            return tool_obj

        wrapped = getattr(tool_obj, "__wrapped__", None)
        if callable(wrapped):
            return wrapped

        for attr in ("fn", "func", "function", "_fn", "_func"):
            candidate = getattr(tool_obj, attr, None)
            if callable(candidate):
                return candidate

        raise TypeError(
            f"Tool '{tool_name}' is not directly callable and no wrapped function was found. "
            f"Got type: {type(tool_obj).__name__}"
        )
    
    for function_call in actions_json:
        func_name = function_call.get("function")
        func_args = function_call.get("args", {})
        image_before = _get_screenshot()
        callable_func = _resolve_callable(func_name)
        callable_func(**func_args)
        time.sleep(0.5)
        image_after = _get_screenshot()
        score, _ = ssim(image_before, image_after, full=True)
        if score < 0.85:
            tree.add_step("Tool Chain aborted due to abrupt change in the screen falling below threshold")
            return f"WARNING: Chain interrupted. Visual environment changed after {func_name}. Re-evaluate screen"
        
    tree.add_step(f"Executed tool chain with actions: {actions}")
    return f"Executed tool chain with actions: {actions}"

if __name__ == "__main__":
    # image_before = _get_screenshot()
    # print("Sleeping for 3 seconds.")
    # time.sleep(3)
    # image_after = _get_screenshot()
    # score, _ = ssim(image_before, image_after, full=True)
    # print(f"SSIM score: {score}")
    # open_app("djklsfjkldsj")
    pass 