import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from infrastructure.adapter.MSSScreenCaptureAdapter import MssScreenCaptureAdapter


class ScreenCaptureResizeTests(unittest.TestCase):
    def test_resize_for_inference_uses_1280x720_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = MssScreenCaptureAdapter(output_dir=tmpdir)
            frame = np.zeros((1440, 2560, 3), dtype=np.uint8)

            resized = adapter._resize_for_inference(frame)

            self.assertEqual(resized.shape, (720, 1280, 3))


if __name__ == "__main__":
    unittest.main()
