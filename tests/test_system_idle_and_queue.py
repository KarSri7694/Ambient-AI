import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
