"""
inspect_window_visible_only.py
------------------------------
Scans the active foreground window and prints visible UI elements.

Printed columns:
S.No., Item_Name, Start_X, Start_Y, Length_X_px, Length_Y_px, End_X, End_Y, Bounding_Box

Usage:
  python inspect_window_visible_only.py [delay_seconds]
"""

import sys
import time
import ctypes
import uiautomation as auto


def is_chromium_window(window) -> bool:
    """Return True if the top-level class looks Chromium/Electron-based."""
    try:
        return window.ClassName in ("Chrome_WidgetWin_1", "Chrome_WidgetWin_0")
    except Exception:
        return False


def _get_screen_reader_state() -> bool:
    """Return current SPI_GETSCREENREADER state."""
    value = ctypes.c_int(0)
    ctypes.windll.user32.SystemParametersInfoW(0x0046, 0, ctypes.byref(value), 0)
    return bool(value.value)


def enable_chromium_accessibility() -> None:
    """Enable Chromium UIA bridge for the current scan session."""
    ctypes.windll.user32.SystemParametersInfoW(0x0047, 1, None, 0x0002)
    time.sleep(0.8)


def disable_chromium_accessibility() -> None:
    """Disable Chromium UIA bridge after scan."""
    ctypes.windll.user32.SystemParametersInfoW(0x0047, 0, None, 0x0002)


CLICKABLE_CONTROL_TYPES = {
    auto.ControlType.ButtonControl,
    auto.ControlType.CheckBoxControl,
    auto.ControlType.RadioButtonControl,
    auto.ControlType.HyperlinkControl,
    auto.ControlType.ListItemControl,
    auto.ControlType.MenuItemControl,
    auto.ControlType.TabItemControl,
    auto.ControlType.TreeItemControl,
    auto.ControlType.SplitButtonControl,
    auto.ControlType.ComboBoxControl,
    auto.ControlType.HeaderItemControl,
    auto.ControlType.DataItemControl,
}

WRITABLE_CONTROL_TYPES = {
    auto.ControlType.EditControl,
}

CONTENT_CONTROL_TYPES = {
    auto.ControlType.TextControl,
    auto.ControlType.DocumentControl,
    auto.ControlType.PaneControl,
    auto.ControlType.GroupControl,
    auto.ControlType.ListControl,
    auto.ControlType.ListItemControl,
    auto.ControlType.TreeControl,
    auto.ControlType.TreeItemControl,
    auto.ControlType.TabControl,
    auto.ControlType.TabItemControl,
    auto.ControlType.TableControl,
    auto.ControlType.DataGridControl,
    auto.ControlType.DataItemControl,
    auto.ControlType.HeaderControl,
    auto.ControlType.HeaderItemControl,
    auto.ControlType.MenuBarControl,
    auto.ControlType.MenuControl,
    auto.ControlType.MenuItemControl,
    auto.ControlType.StatusBarControl,
    auto.ControlType.ToolBarControl,
    auto.ControlType.HyperlinkControl,
    auto.ControlType.ButtonControl,
    auto.ControlType.CheckBoxControl,
    auto.ControlType.RadioButtonControl,
    auto.ControlType.ComboBoxControl,
    auto.ControlType.WindowControl,
}

MAX_ITEM_NAME_CHARS = 50
DEDUPE_OVERLAP_THRESHOLD = 0.85


def _virtual_screen_bounds() -> tuple[int, int, int, int]:
    """Return virtual desktop bounds as (left, top, right, bottom)."""
    user32 = ctypes.windll.user32
    left = user32.GetSystemMetrics(76)   # SM_XVIRTUALSCREEN
    top = user32.GetSystemMetrics(77)    # SM_YVIRTUALSCREEN
    width = user32.GetSystemMetrics(78)  # SM_CXVIRTUALSCREEN
    height = user32.GetSystemMetrics(79) # SM_CYVIRTUALSCREEN

    if width <= 0 or height <= 0:
        left = 0
        top = 0
        width = user32.GetSystemMetrics(0)   # SM_CXSCREEN
        height = user32.GetSystemMetrics(1)  # SM_CYSCREEN

    return left, top, left + width, top + height


def rect_is_visible_on_screen(rect: auto.Rect) -> bool:
    """True if rectangle has area and intersects the visible virtual desktop."""
    try:
        left = rect.left
        top = rect.top
        right = rect.right
        bottom = rect.bottom
        width = rect.width()
        height = rect.height()
    except Exception:
        return False

    if width <= 0 or height <= 0:
        return False

    vx1, vy1, vx2, vy2 = _virtual_screen_bounds()
    return right > vx1 and bottom > vy1 and left < vx2 and top < vy2


def _rect_area(bbox: tuple[int, int, int, int]) -> int:
    """Return area of (left, top, right, bottom) bbox."""
    width = max(0, bbox[2] - bbox[0])
    height = max(0, bbox[3] - bbox[1])
    return width * height


def _intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    """Return intersection area of two bboxes."""
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0
    return (right - left) * (bottom - top)


def _is_duplicate_row(
    existing: tuple,
    candidate: tuple,
    overlap_threshold: float = DEDUPE_OVERLAP_THRESHOLD,
) -> bool:
    """Return True for rows with same normalized name and strong bbox overlap."""
    existing_name = str(existing[0]).strip().lower()
    candidate_name = str(candidate[0]).strip().lower()
    if existing_name != candidate_name:
        return False

    existing_bbox = existing[7]
    candidate_bbox = candidate[7]
    inter = _intersection_area(existing_bbox, candidate_bbox)
    if inter <= 0:
        return False

    existing_area = _rect_area(existing_bbox)
    candidate_area = _rect_area(candidate_bbox)
    smaller_area = min(existing_area, candidate_area)
    if smaller_area <= 0:
        return False

    # Using overlap over the smaller rectangle catches parent/child duplicates.
    overlap_ratio = inter / smaller_area
    return overlap_ratio >= overlap_threshold


def _is_same_or_ancestor(ancestor: auto.Control, node: auto.Control, max_hops: int = 40) -> bool:
    """Return True if ancestor == node or ancestor appears in node parent chain."""
    try:
        if auto.ControlsAreSame(ancestor, node):
            return True
    except Exception:
        return False

    current = node
    for _ in range(max_hops):
        try:
            current = current.GetParentControl()
        except Exception:
            return False
        if current is None:
            return False
        try:
            if auto.ControlsAreSame(ancestor, current):
                return True
        except Exception:
            return False
    return False


def is_visible_on_screen(control: auto.Control) -> bool:
    """Stricter visibility: UIA visible + non-zero area + screen overlap + hit-test."""
    try:
        if control.IsOffscreen:
            return False
        rect = control.BoundingRectangle
    except Exception:
        return False

    if not rect_is_visible_on_screen(rect):
        return False

    try:
        left, top = rect.left, rect.top
        right, bottom = rect.right, rect.bottom
        points = [
            (int((left + right) / 2), int((top + bottom) / 2)),
        ]
        if rect.width() > 4 and rect.height() > 4:
            points.append((int(left + 1), int(top + 1)))
            points.append((int(right - 2), int(bottom - 2)))
    except Exception:
        return False

    for px, py in points:
        try:
            hit = auto.ControlFromPoint(px, py)
        except Exception:
            hit = None
        if hit is None:
            continue
        if _is_same_or_ancestor(control, hit) or _is_same_or_ancestor(hit, control):
            return True

    return False


def is_enabled(control: auto.Control) -> bool:
    try:
        return control.IsEnabled
    except Exception:
        return False


def supports_pattern(control: auto.Control, pattern_id: int) -> bool:
    try:
        return control.GetPattern(pattern_id) is not None
    except Exception:
        return False


def is_clickable(control: auto.Control) -> bool:
    if control.ControlType in CLICKABLE_CONTROL_TYPES:
        return True
    if supports_pattern(control, auto.PatternId.InvokePattern):
        return True
    if supports_pattern(control, auto.PatternId.SelectionItemPattern):
        return True
    if supports_pattern(control, auto.PatternId.TogglePattern):
        return True
    if supports_pattern(control, auto.PatternId.ExpandCollapsePattern):
        return True
    return False


def is_writable(control: auto.Control) -> bool:
    if control.ControlType in WRITABLE_CONTROL_TYPES:
        try:
            value_pattern = control.GetPattern(auto.PatternId.ValuePattern)
            if value_pattern is not None:
                return not value_pattern.IsReadOnly
        except Exception:
            pass
        return True

    if supports_pattern(control, auto.PatternId.TextEditPattern):
        return True

    try:
        value_pattern = control.GetPattern(auto.PatternId.ValuePattern)
        if value_pattern is not None and not value_pattern.IsReadOnly:
            return True
    except Exception:
        pass

    return False


def _safe_control_name(control: auto.Control) -> str:
    try:
        return str(control.Name or "").strip()
    except Exception:
        return ""


def _safe_control_value(control: auto.Control) -> str:
    try:
        value_pattern = control.GetPattern(auto.PatternId.ValuePattern)
        if value_pattern is not None:
            return str(value_pattern.Value or "").strip()
    except Exception:
        pass
    try:
        legacy = control.GetPattern(auto.PatternId.LegacyIAccessiblePattern)
        if legacy is not None:
            return str(legacy.Value or "").strip()
    except Exception:
        pass
    return ""


def _text_candidate(control: auto.Control) -> str:
    for value in (_safe_control_name(control), _safe_control_value(control)):
        if value and len(value) > 1:
            return value
    return ""


def is_content_candidate(control: auto.Control) -> bool:
    try:
        if control.ControlType in CONTENT_CONTROL_TYPES and _text_candidate(control):
            return True
    except Exception:
        return False
    return False


def get_same_process_windows(target_pid: int) -> list:
    """Collect visible top-level windows belonging to target_pid."""
    windows = []
    desktop = auto.GetRootControl()
    child = desktop.GetFirstChildControl()
    while child:
        try:
            if child.ProcessId == target_pid and is_visible_on_screen(child):
                windows.append(child)
        except Exception:
            pass
        child = child.GetNextSiblingControl()
    return windows


def scan_window(window: auto.Control, mode: str = "interactive_only") -> list:
    """Return visible UI rows for the foreground app.

    interactive_only:
      Visible clickable or writable controls only.
    screen_content:
      Visible interactive controls plus passive text-bearing content controls.
    """
    results = []
    normalized_mode = str(mode or "interactive_only").strip().lower()
    include_content = normalized_mode == "screen_content"

    try:
        target_pid = window.ProcessId
    except Exception:
        target_pid = None

    windows_to_scan = [window]
    if target_pid:
        for w in get_same_process_windows(target_pid):
            try:
                if not auto.ControlsAreSame(w, window):
                    windows_to_scan.append(w)
            except Exception:
                windows_to_scan.append(w)

    for win in windows_to_scan:
        for control, _depth in auto.WalkControl(win, includeTop=False, maxDepth=50):
            try:
                if not is_visible_on_screen(control):
                    continue
                if not include_content and not is_enabled(control):
                    continue
                if include_content:
                    if not (is_clickable(control) or is_writable(control) or is_content_candidate(control)):
                        continue
                elif not (is_clickable(control) or is_writable(control)):
                    continue

                clean_name = _text_candidate(control)
                if not clean_name:
                    continue
                if len(clean_name) == 1:
                    continue

                try:
                    rect = control.BoundingRectangle
                    start_x = int(rect.left)
                    start_y = int(rect.top)
                    length_x = int(rect.width())
                    length_y = int(rect.height())
                except Exception:
                    continue

                if length_x <= 0 or length_y <= 0:
                    continue

                end_x = start_x + length_x
                end_y = start_y + length_y
                bbox = (start_x, start_y, end_x, end_y)

                results.append((
                    clean_name,
                    start_x,
                    start_y,
                    length_x,
                    length_y,
                    end_x,
                    end_y,
                    bbox,
                ))
            except Exception:
                continue

    return results


def deduplicate_results(results: list) -> list:
    """Remove repeated UI rows with same name and highly overlapping boxes."""
    deduped = []
    for row in results:
        match_index = -1
        for idx, kept in enumerate(deduped):
            if _is_duplicate_row(kept, row):
                match_index = idx
                break

        if match_index == -1:
            deduped.append(row)
            continue

        # Keep the larger region when duplicate parent/child nodes are found.
        kept_area = _rect_area(deduped[match_index][7])
        row_area = _rect_area(row[7])
        if row_area > kept_area:
            deduped[match_index] = row

    return deduped


def print_compact_bbox_table(results: list) -> None:
    print("Compact Output Format (Visible On-Screen Elements Only)")
    print("-" * 100)
    print(
        "S.No., Item_Name, Start_X, Start_Y, Length_X_px, Length_Y_px, End_X, End_Y, Bounding_Box"
    )

    serial = 0
    for row in results:
        (
            name,
            start_x,
            start_y,
            length_x,
            length_y,
            end_x,
            end_y,
            bbox,
        ) = row

        truncated_name = name[:MAX_ITEM_NAME_CHARS]
        serial += 1
        print(
            f"{serial}, {truncated_name!r}, {bbox}"
        )

    if serial == 0:
        print("No interactive visible elements found.")


def inspect_foreground_window(mode: str = "interactive_only") -> dict:
    """Return structured UIA snapshot for the active foreground window."""
    window = auto.GetForegroundControl()
    if window is None:
        return {
            "ok": False,
            "mode": mode,
            "error": "foreground_window_not_found",
        }

    try:
        window_title = window.Name or window.ClassName or "(untitled)"
    except Exception:
        window_title = "(untitled)"

    window_class = None
    process_id = None
    try:
        window_class = window.ClassName or None
    except Exception:
        window_class = None
    try:
        process_id = int(window.ProcessId)
    except Exception:
        process_id = None

    chromium = is_chromium_window(window)
    accessibility_activated = False
    if chromium:
        original_state = _get_screen_reader_state()
        if not original_state:
            enable_chromium_accessibility()
            accessibility_activated = True

    try:
        rows = deduplicate_results(scan_window(window, mode=mode))
    finally:
        if accessibility_activated:
            disable_chromium_accessibility()

    items = [
        {
            "name": row[0],
            "start_x": row[1],
            "start_y": row[2],
            "length_x": row[3],
            "length_y": row[4],
            "end_x": row[5],
            "end_y": row[6],
            "bbox": row[7],
        }
        for row in rows
    ]
    visible_text_summary = "\n".join(item["name"] for item in items[:80])
    lowered = visible_text_summary.lower()
    return {
        "ok": True,
        "mode": mode,
        "window_title": str(window_title),
        "window_class": window_class,
        "process_id": process_id,
        "is_chromium": chromium,
        "visible_items": items,
        "visible_text_summary": visible_text_summary,
        "contains_dialog": any(keyword in lowered for keyword in ("dialog", "confirm", "warning", "permission")),
        "contains_notification": any(keyword in lowered for keyword in ("notification", "reminder", "alert", "error")),
        "contains_editable_fields": any(isinstance(item["name"], str) and item["name"] for item in items),
    }


def main() -> None:
    try:
        delay = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    except ValueError:
        delay = 3

    print("\nUIAutomation - Visible Interactive Element Inspector")
    print("-" * 60)
    print(
        f"You have {delay}s - switch to your app and open any menu/dialog you want captured."
    )

    for remaining in range(delay, 0, -1):
        print(f"  Scanning in {remaining}s ...", end="\r")
        time.sleep(1)
    print()

    window = auto.GetForegroundControl()
    if window is None:
        print("ERROR: Could not find a foreground window.")
        return

    try:
        win_title = window.Name or window.ClassName or "(untitled)"
    except Exception:
        win_title = "(untitled)"

    print(f"Window: {win_title!r}")

    chromium = is_chromium_window(window)
    accessibility_activated = False
    if chromium:
        original_state = _get_screen_reader_state()
        if not original_state:
            enable_chromium_accessibility()
            accessibility_activated = True

    try:
        results = scan_window(window)
    finally:
        if accessibility_activated:
            disable_chromium_accessibility()

    results = deduplicate_results(results)

    print_compact_bbox_table(results)
    print()


if __name__ == "__main__":
    main()
