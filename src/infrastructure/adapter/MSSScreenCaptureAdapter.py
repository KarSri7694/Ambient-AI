import os
import time
from typing import Optional
import cv2
from mss import mss
import numpy as np
from application.ports.screen_capture_port import ScreenCapturePort

class MssScreenCaptureAdapter(ScreenCapturePort):
    def __init__(self, output_dir: str = "outputs/screenshots"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def capture_screenshot(self, output_path: Optional[str] = None) -> str:
        """
        Captures a screenshot of the primary monitor using the mss library.
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

            frame_1080p = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
            # Save to the picture file
            cv2.imwrite(output_path, frame_1080p)
            
        return os.path.abspath(output_path)
