import os
import time
from typing import Optional
import cv2
from mss import mss
import numpy as np
from application.ports.screen_capture_port import ScreenCapturePort


class MssScreenCaptureAdapter(ScreenCapturePort):
    def __init__(self, output_dir: str = "outputs/screenshots", target_width: int = 1280, target_height: int = 720):
        self.output_dir = output_dir
        self.target_width = target_width
        self.target_height = target_height
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _resize_for_inference(self, frame: np.ndarray) -> np.ndarray:
        """Normalize screenshots to a bounded inference resolution."""
        return cv2.resize(
            frame,
            (self.target_width, self.target_height),
            interpolation=cv2.INTER_AREA,
        )

    def capture_screenshot(self, output_path: Optional[str] = None) -> str:
        """
        Captures a screenshot of the primary monitor and stores a 1280x720 image.
        """
        if output_path is None:
            filename = f"screenshot_{int(time.time())}.png"
            output_path = os.path.join(self.output_dir, filename)

        with mss() as sct:
            # The screen part to capture (the first monitor)
            monitor = sct.monitors[1]
            
            # Grab the data
            sct_img = sct.grab(monitor)
            frame = np.array(sct_img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            resized_frame = self._resize_for_inference(frame)
            # Save to the picture file
            cv2.imwrite(output_path, resized_frame)
            
        return os.path.abspath(output_path)
