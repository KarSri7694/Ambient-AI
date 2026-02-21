from abc import ABC, abstractmethod
from typing import Optional

class ScreenCapturePort(ABC):
    @abstractmethod
    def capture_screenshot(self, output_path: Optional[str] = None) -> str:
        """
        Captures a screenshot of the primary monitor.
        
        Args:
            output_path: The file path to save the screenshot. If None, a default path or memory buffer might be used.
            
        Returns:
            The path to the saved screenshot file.
        """
        pass
