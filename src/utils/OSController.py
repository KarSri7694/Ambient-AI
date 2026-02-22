import ctypes
import logging
import AppOpener
import win32gui
import win32con
import subprocess
import time
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CUSTOM_ALIASES = {
    "notepad": {
        "command": r"C:\Windows\System32\notepad.exe",
        "expected_title": "notepad"
    }
}

class OSController:
    """Controller for OS-level interactions"""
    def _set_foreground_lock(self, timeout_ms: int = 200000):
        """
        Sets the Windows ForegroundLockTimeout.
        timeout_ms: 200000 (200 seconds) is the standard Windows default.
        """
        SPI_SETFOREGROUNDLOCKTIMEOUT = 0x2001
        SPIF_UPDATEINIFILE = 0x01
        SPIF_SENDCHANGE = 0x02

        result = ctypes.windll.user32.SystemParametersInfoW(
            SPI_SETFOREGROUNDLOCKTIMEOUT, 
            0,
            timeout_ms,
            SPIF_UPDATEINIFILE | SPIF_SENDCHANGE
        )

        if result:
            logging.info(f"SYSTEM LOG: Foreground lock set to {timeout_ms} milliseconds.")
        else:
            logging.warning("SYSTEM WARNING: Failed to set foreground lock.")
    
    def open_and_verify_app(self, app_name: str, timeout: int = 10) -> str:
        """
        Opens an application by name using AppOpener.
        """
        target_app = app_name.lower().strip()
        expected_title = target_app
        if target_app in CUSTOM_ALIASES:
            config = CUSTOM_ALIASES[target_app]
            command = config["command"]
            expected_title = config["expected_title"]
            try:
                subprocess.Popen(command)
                logging.info(f"SYSTEM LOG: Opening application '{app_name}' using custom alias.")
            except Exception as e:
                logging.error(f"SYSTEM ERROR: Failed to open application '{app_name}' using custom alias. Error: {e}")
                return
        else:
            try:
                AppOpener.open(app_name, match_closest=True)
                logging.info(f"SYSTEM LOG: Opening application '{app_name}'.")
            except Exception as e:
                logging.error(f"SYSTEM ERROR: Failed to open application '{app_name}'. Error: {e}")
                
        start_time = time.time()
    
        while (time.time() - start_time) < timeout:
            # Get the current active window
            hwnd = win32gui.GetForegroundWindow()
            active_window_title = win32gui.GetWindowText(hwnd).lower()
            
            if active_window_title and expected_title in active_window_title:
                load_time = round(time.time() - start_time, 2)
                logging.info(f"SUCCESS: '{active_window_title}' opened and verified in {load_time} seconds. You may proceed.")
                return 

            time.sleep(0.5)

        # 3. Timeout Failure
        return f"FAILURE: Thread timed out after {timeout} seconds. The active window is currently '{active_window_title}'. The app either failed to launch, or launched in the background."
    

    def focus_existing_window(self, expected_title: str) -> bool:
        """
        Searches all open windows for a partial title match. 
        If found, it un-minimizes the window and forces it to the foreground.
        Returns True if successful, False if the window does not exist.
        """
        found_hwnds = []
        search_title = expected_title.lower().strip()

        # Callback function required by EnumWindows
        def enum_windows_callback(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd).lower()
                if search_title in window_text:
                    found_hwnds.append(hwnd)

        win32gui.EnumWindows(enum_windows_callback, None)

        if not found_hwnds:
            return False # The app is not running or has no visible window

        # Grab the first matching window handle
        target_hwnd = found_hwnds[0]

        try:
            # SW_MAXIMIZE un-minimizes and maximizes the window
            win32gui.ShowWindow(target_hwnd, win32con.SW_MAXIMIZE)
            
            win32gui.SetForegroundWindow(target_hwnd)
            return True
        except Exception as e:
            print(f"DEBUG: Found window but failed to force focus. Error: {e}")
            return False

def main():
    controller = OSController()
    controller._set_foreground_lock(0) #
    name = input("Enter the name of the application you want to open: ")
    if controller.focus_existing_window(name):
        print(f"SUCCESS: Found existing window for '{name}' and focused it. You may proceed.")
    else:
        controller.open_and_verify_app(name)


if __name__ == "__main__":
    main()