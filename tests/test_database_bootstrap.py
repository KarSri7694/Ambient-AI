import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

import night_mode
from database_bootstrap import ensure_runtime_databases


class DatabaseBootstrapTests(unittest.TestCase):
    def test_bootstrap_creates_required_runtime_tables(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            original_night_db = night_mode.DB_FILE
            try:
                night_mode.DB_FILE = str(temp_path / "night_queue.db")
                ensure_runtime_databases(
                    memory_db_path=str(temp_path / "memory.db"),
                    memory_root=str(temp_path / "memory"),
                    ambient_agenda_db_path=str(temp_path / "ambient_agenda.db"),
                    interaction_log_db_path=str(temp_path / "interaction_logs.db"),
                    proactive_topics_db_path=str(temp_path / "proactive_topics.db"),
                    activity_ledger_db_path=str(temp_path / "activity_ledger.db"),
                    voice_db_path=str(temp_path / "voice_database.db"),
                    finance_db_path=str(temp_path / "finance.db"),
                    facts_db_path=str(temp_path / "facts.db"),
                )

                self._assert_table_exists(temp_path / "night_queue.db", "night_queue")
                self._assert_table_exists(temp_path / "night_queue.db", "system_notifications")
                self._assert_table_exists(temp_path / "memory.db", "speakers")
                self._assert_table_exists(temp_path / "memory.db", "visual_user_facts")
                self._assert_table_exists(temp_path / "ambient_agenda.db", "ambient_agenda")
                self._assert_table_exists(temp_path / "interaction_logs.db", "interaction_logs")
                self._assert_table_exists(temp_path / "proactive_topics.db", "proactive_topics")
                self._assert_table_exists(temp_path / "activity_ledger.db", "activity_runs")
                self._assert_table_exists(temp_path / "activity_ledger.db", "activity_steps")
                self._assert_table_exists(temp_path / "activity_ledger.db", "activity_artifacts")
                self._assert_table_exists(temp_path / "activity_ledger.db", "activity_links")
                self._assert_table_exists(temp_path / "activity_ledger.db", "activity_tags")
                self._assert_table_exists(temp_path / "voice_database.db", "VOICE_EMBEDDINGS")
                self._assert_table_exists(temp_path / "finance.db", "transactions")
                self._assert_table_exists(temp_path / "finance.db", "categories")
                self._assert_table_exists(temp_path / "finance.db", "accounts")
                self._assert_table_exists(temp_path / "facts.db", "facts")
            finally:
                night_mode.DB_FILE = original_night_db

    def _assert_table_exists(self, db_path: Path, table_name: str) -> None:
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            )
            row = cursor.fetchone()
            self.assertIsNotNone(row, f"Expected table '{table_name}' in {db_path}")
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
