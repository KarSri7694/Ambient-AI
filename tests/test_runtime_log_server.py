import logging
import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from infrastructure.runtime_log_server import RuntimeLogBuffer, RuntimeLogBufferHandler, create_runtime_log_app


class RuntimeLogServerTests(unittest.TestCase):
    def test_api_returns_recent_buffered_logs(self):
        buffer = RuntimeLogBuffer(max_entries=10)
        handler = RuntimeLogBufferHandler(buffer)
        handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))

        logger = logging.getLogger("runtime-log-test")
        logger.setLevel(logging.INFO)
        logger.handlers = []
        logger.propagate = False
        logger.addHandler(handler)

        logger.info("first")
        logger.warning("second")

        app = create_runtime_log_app(buffer)
        client = TestClient(app)
        response = client.get("/api/logs")
        payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["count"], 2)
        self.assertEqual(payload["entries"][0]["message"], "INFO:first")
        self.assertEqual(payload["entries"][1]["message"], "WARNING:second")

    def test_api_filters_by_after_id(self):
        buffer = RuntimeLogBuffer(max_entries=10)
        fake_record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello",
            args=(),
            exc_info=None,
        )
        first = buffer.append(fake_record, "one")
        second = buffer.append(fake_record, "two")

        app = create_runtime_log_app(buffer)
        client = TestClient(app)
        response = client.get(f"/api/logs?after_id={first['id']}")
        payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual([entry["message"] for entry in payload["entries"]], ["two"])
        self.assertEqual(payload["latest_id"], second["id"])


if __name__ == "__main__":
    unittest.main()
