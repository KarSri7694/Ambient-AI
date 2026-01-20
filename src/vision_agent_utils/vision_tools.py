import pyautogui
from fastmcp import FastMCP
from typing import Annotated

mcp = FastMCP()

#For Qwen3-VL
RELATIVE_WIDTH = 1000
RELATIVE_HEIGHT = 1000

#Helper function to convert coordinates
def convert_to_original_coordinates(x: int, y: int, current_width: int, current_height: int, original_width: int, original_height: int):
    """
    Convert coordinates from the current image size back to the original image size.

    Args:
        x: X coordinate in the current image size.
        y: Y coordinate in the current image size.
        current_width: total width of current coordinate system.
        current_height: total height of current coordinate system.
        original_width: Width of the original coordinate system.
        original_height: Height of the original coordinate system.
    """
    original_x = int(x * original_width / current_width)
    original_y = int(y * original_height / current_height)
    return original_x, original_y

#mouse Tools
@mcp.tool
def mouse_click(x: Annotated[int, "mouse position X coordinate"], y: Annotated[int, "mouse position Y coordinate"]):
    """
    Simulate a mouse click at the current cursor position.
    """
    x_new,y_new = convert_to_original_coordinates(x, y, RELATIVE_WIDTH, RELATIVE_HEIGHT, pyautogui.size().width, pyautogui.size().height)
    pyautogui.click(x_new, y_new)
    return f"Clicked at ({x}, {y})."
    
@mcp.tool
def mouse_double_click(x: Annotated[int, "mouse position X coordinate"], y: Annotated[int, "mouse position Y coordinate"]):
    """
    Simulate a mouse double click at the current cursor position.
    """
    x_new,y_new = convert_to_original_coordinates(x, y, RELATIVE_WIDTH, RELATIVE_HEIGHT, pyautogui.size().width, pyautogui.size().height)
    pyautogui.doubleClick(x_new, y_new)
    return f"Double clicked at ({x}, {y})."
    
@mcp.tool
def mouse_right_click(x: Annotated[int, "mouse position X coordinate"], y: Annotated[int, "mouse position Y coordinate"]):
    """
    Simulate a mouse right click at the current cursor position.
    """
    x_new,y_new = convert_to_original_coordinates(x, y, RELATIVE_WIDTH, RELATIVE_HEIGHT, pyautogui.size().width, pyautogui.size().height)
    pyautogui.rightClick(x_new, y_new)
    return f"Right clicked at ({x}, {y})."

@mcp.tool
def move_mouse(x: Annotated[int, "mouse position X coordinate"], y: Annotated[int, "mouse position Y coordinate"]):
    """
    Move the mouse cursor to a specific position on the screen.
    """
    x_new,y_new = convert_to_original_coordinates(x, y, RELATIVE_WIDTH, RELATIVE_HEIGHT, pyautogui.size().width, pyautogui.size().height)
    pyautogui.moveTo(x_new, y_new)
    return f"Moved mouse to ({x}, {y})."

@mcp.tool
def scroll(amount: Annotated[int, "amount to scroll (positive for up, negative for down)"]):
    """
    Scroll the mouse wheel up or down.
    Positive amount scrolls up, negative scrolls down.
    """
    pyautogui.scroll(amount)
    return f"Scrolled {'up' if amount > 0 else 'down'} by {abs(amount)} units."


@mcp.tool
def input_text(text: Annotated[str, "text to be typed"]):
    """
    Type a string of text at the current cursor location.
    """
    pyautogui.typewrite(text)
    return f"Typed text: {text}"
    
@mcp.tool
def press_key(key: Annotated[str, "key  to press (e.g., 'enter' "]):
    """
    Press a specific special key or hotkey combination.
    """
    pyautogui.press(key)
    return f"Pressed key: {key}"

@mcp.tool
def keyboard_hotkey(keys: Annotated[str, "key to press eg. 'ctrl+c'"]):
    """
    Press a specific hotkey combination.
    """
    keys_split = keys.split('+')
    pyautogui.hotkey(*keys_split)
    return f"Pressed hotkey combination: {keys}"