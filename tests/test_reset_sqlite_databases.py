import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from reset_sqlite_databases import DatabaseSpec, hard_reset_database, soft_reset_database


class ResetSqliteDatabasesTests(unittest.TestCase):
    def test_soft_reset_clears_rows_but_preserves_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "sample.db"

            def init_db() -> None:
                conn = sqlite3.connect(db_path)
                try:
                    conn.execute("CREATE TABLE IF NOT EXISTS alpha (id INTEGER PRIMARY KEY AUTOINCREMENT, value TEXT)")
                    conn.execute("CREATE TABLE IF NOT EXISTS beta (id INTEGER PRIMARY KEY AUTOINCREMENT, value TEXT)")
                    conn.commit()
                finally:
                    conn.close()

            init_db()
            conn = sqlite3.connect(db_path)
            try:
                conn.execute("INSERT INTO alpha(value) VALUES ('x')")
                conn.execute("INSERT INTO beta(value) VALUES ('y')")
                conn.commit()
            finally:
                conn.close()

            spec = DatabaseSpec(
                name="sample",
                path=db_path,
                clear_statements=("DELETE FROM alpha", "DELETE FROM beta"),
                init_func=init_db,
                description="sample db",
            )

            soft_reset_database(spec)

            conn = sqlite3.connect(db_path)
            try:
                alpha_count = conn.execute("SELECT COUNT(*) FROM alpha").fetchone()[0]
                beta_count = conn.execute("SELECT COUNT(*) FROM beta").fetchone()[0]
                tables = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                    ).fetchall()
                }
            finally:
                conn.close()

            self.assertEqual(alpha_count, 0)
            self.assertEqual(beta_count, 0)
            self.assertEqual(tables, {"alpha", "beta"})

    def test_hard_reset_recreates_deleted_database_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "sample.db"

            def init_db() -> None:
                conn = sqlite3.connect(db_path)
                try:
                    conn.execute("CREATE TABLE IF NOT EXISTS alpha (id INTEGER PRIMARY KEY AUTOINCREMENT, value TEXT)")
                    conn.commit()
                finally:
                    conn.close()

            init_db()
            conn = sqlite3.connect(db_path)
            try:
                conn.execute("INSERT INTO alpha(value) VALUES ('x')")
                conn.commit()
            finally:
                conn.close()

            spec = DatabaseSpec(
                name="sample",
                path=db_path,
                clear_statements=("DELETE FROM alpha",),
                init_func=init_db,
                description="sample db",
            )

            hard_reset_database(spec)

            self.assertTrue(db_path.exists())
            conn = sqlite3.connect(db_path)
            try:
                count = conn.execute("SELECT COUNT(*) FROM alpha").fetchone()[0]
            finally:
                conn.close()

            self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
