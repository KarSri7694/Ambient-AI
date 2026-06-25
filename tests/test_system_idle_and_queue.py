import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from application.services.screenshot_queue_service import ScreenshotQueueService
from application.services.system_idle_service import SystemIdleService


class _FakeUser32:
    def __init__(self, last_input_tick: int):
        self.last_input_tick = last_input_tick

    def GetLastInputInfo(self, info_ptr):
        info_ptr._obj.dwTime = self.last_input_tick
        return 1


class _FakeKernel32:
    def __init__(self, tick_count: int):
        self.tick_count = tick_count

    def GetTickCount(self):
        return self.tick_count


class SystemIdleAndQueueTests(unittest.TestCase):
    def test_system_idle_service_reports_idle_after_threshold(self):
        fake_windll = types.SimpleNamespace(
            user32=_FakeUser32(last_input_tick=1000),
            kernel32=_FakeKernel32(tick_count=301000),
        )
        with patch("application.services.system_idle_service.ctypes.windll", fake_windll, create=True):
            service = SystemIdleService(idle_threshold_seconds=300)
            self.assertAlmostEqual(service.get_idle_seconds(), 300.0, places=2)
            self.assertTrue(service.is_user_idle())
            self.assertFalse(service.is_user_active())

    def test_system_idle_service_reports_active_under_threshold(self):
        fake_windll = types.SimpleNamespace(
            user32=_FakeUser32(last_input_tick=1000),
            kernel32=_FakeKernel32(tick_count=250000),
        )
        with patch("application.services.system_idle_service.ctypes.windll", fake_windll, create=True):
            service = SystemIdleService(idle_threshold_seconds=300)
            self.assertFalse(service.is_user_idle())
            self.assertTrue(service.is_user_active())

    def test_screenshot_queue_drops_oldest_when_full(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            first = temp_path / "first.png"
            second = temp_path / "second.png"
            third = temp_path / "third.png"
            first.write_bytes(b"1")
            second.write_bytes(b"2")
            third.write_bytes(b"3")

            queue = ScreenshotQueueService(maxlen=2)
            queue.enqueue(str(first), captured_at="2026-06-25T10:00:00")
            queue.enqueue(str(second), captured_at="2026-06-25T10:00:10")
            queue.enqueue(str(third), captured_at="2026-06-25T10:00:20")

            self.assertFalse(first.exists())
            self.assertEqual(queue.size(), 2)
            oldest = queue.dequeue()
            newest = queue.dequeue()
            self.assertEqual(oldest.screenshot_path, str(second))
            self.assertEqual(newest.screenshot_path, str(third))

    def test_screenshot_queue_skips_similar_recent_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            first = temp_path / "first.png"
            duplicate = temp_path / "duplicate.png"
            self._write_solid_image(first, color=(32, 64, 96))
            self._write_solid_image(duplicate, color=(32, 64, 96))

            queue = ScreenshotQueueService(maxlen=5, ssim_threshold=0.92, ssim_compare_count=4)
            first_job = queue.enqueue(str(first), captured_at="2026-06-25T10:00:00")
            duplicate_job = queue.enqueue(str(duplicate), captured_at="2026-06-25T10:00:10")

            self.assertIsNotNone(first_job)
            self.assertIsNone(duplicate_job)
            self.assertFalse(duplicate.exists())
            self.assertEqual(queue.size(), 1)

    def test_screenshot_queue_accepts_different_recent_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            first = temp_path / "first.png"
            different = temp_path / "different.png"
            self._write_solid_image(first, color=(0, 0, 0))
            self._write_solid_image(different, color=(255, 255, 255))

            queue = ScreenshotQueueService(maxlen=5, ssim_threshold=0.92, ssim_compare_count=4)
            queue.enqueue(str(first), captured_at="2026-06-25T10:00:00")
            different_job = queue.enqueue(str(different), captured_at="2026-06-25T10:00:10")

            self.assertIsNotNone(different_job)
            self.assertTrue(different.exists())
            self.assertEqual(queue.size(), 2)

    def test_screenshot_queue_can_disable_similarity_filter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            first = temp_path / "first.png"
            duplicate = temp_path / "duplicate.png"
            self._write_solid_image(first, color=(32, 64, 96))
            self._write_solid_image(duplicate, color=(32, 64, 96))

            queue = ScreenshotQueueService(maxlen=5, ssim_threshold=0.92, ssim_compare_count=0)
            queue.enqueue(str(first), captured_at="2026-06-25T10:00:00")
            duplicate_job = queue.enqueue(str(duplicate), captured_at="2026-06-25T10:00:10")

            self.assertIsNotNone(duplicate_job)
            self.assertTrue(duplicate.exists())
            self.assertEqual(queue.size(), 2)

    def _write_solid_image(self, path: Path, color: tuple[int, int, int]) -> None:
        Image.new("RGB", (32, 32), color=color).save(path)


if __name__ == "__main__":
    unittest.main()
