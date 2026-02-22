import ctypes
import logging
import AppOpener
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class OSController:
    """Controller for OS-level interactions"""
    def set_foreground_lock(timeout_ms: int = 200000):
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
    
    def open_app(app_name: str):
        """
        Opens an application by name using AppOpener.
        """
        try:
            AppOpener.open(app_name)
            logging.info(f"SYSTEM LOG: Opened application '{app_name}'.")
        except Exception as e:
            logging.error(f"SYSTEM ERROR: Failed to open application '{app_name}'. Error: {e}")