import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from config import CONFIG


USER_DATA_DIR = Path(CONFIG.get_str("runtime", "user_data_dir", "D:\\USER_DATA"))
DB_FILE = str(USER_DATA_DIR / "database" / "night_queue.db")

def init_db():
    Path(DB_FILE).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Create Tasks Table
    c.execute('''CREATE TABLE IF NOT EXISTS night_queue
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  description TEXT,
                  priority TEXT,
                  status TEXT,
                  created_at TEXT,
                  meta_data TEXT,
                  run_at_utc TEXT,
                  claimed_at TEXT,
                  completed_at TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS system_notifications
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  message TEXT,
                  source TEXT, 
                  read INTEGER DEFAULT 0,
                  created_at TEXT)''')
    existing_columns = {
        row[1] for row in c.execute("PRAGMA table_info(night_queue)").fetchall()
    }
    for column_name in ("run_at_utc", "claimed_at", "completed_at"):
        if column_name not in existing_columns:
            c.execute(f"ALTER TABLE night_queue ADD COLUMN {column_name} TEXT")
    conn.commit()
    conn.close()


def _ensure_db():
    init_db()

#Functions for managing night shift tasks
def add_task(description, priority="medium", metadata=None, run_at_utc=None):
    _ensure_db()
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "INSERT INTO night_queue "
        "(description, priority, status, created_at, meta_data, run_at_utc) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            description,
            priority,
            "pending",
            str(datetime.now()),
            json.dumps(metadata, ensure_ascii=False) if metadata is not None else None,
            run_at_utc,
        ),
    )
    conn.commit()
    conn.close()
    return "Task queued."

def get_pending_tasks():
    _ensure_db()
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row # Allows accessing columns by name
    c = conn.cursor()
    c.execute(
        "SELECT * FROM night_queue WHERE status='pending' AND run_at_utc IS NULL "
        "ORDER BY created_at ASC"
    )
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_due_tasks(now_utc):
    _ensure_db()
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute(
        "SELECT * FROM night_queue WHERE status='pending' "
        "AND run_at_utc IS NOT NULL AND run_at_utc <= ? "
        "ORDER BY run_at_utc ASC, created_at ASC",
        (now_utc,),
    )
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_all_pending_tasks():
    _ensure_db()
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute(
        "SELECT * FROM night_queue WHERE status='pending' "
        "ORDER BY CASE WHEN run_at_utc IS NULL THEN 1 ELSE 0 END, run_at_utc, created_at"
    )
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def claim_task(task_id):
    _ensure_db()
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "UPDATE night_queue SET status='running', claimed_at=? "
        "WHERE id=? AND status='pending'",
        (datetime.now(timezone.utc).isoformat(), task_id),
    )
    claimed = c.rowcount == 1
    conn.commit()
    conn.close()
    return claimed

def mark_task_complete(task_id, status="completed"):
    _ensure_db()
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "UPDATE night_queue SET status=?, completed_at=? WHERE id=?",
        (status, datetime.now(timezone.utc).isoformat(), task_id),
    )
    conn.commit()
    conn.close()


def cancel_task(task_id):
    _ensure_db()
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "UPDATE night_queue SET status='cancelled', completed_at=? "
        "WHERE id=? AND status='pending'",
        (datetime.now(timezone.utc).isoformat(), task_id),
    )
    cancelled = c.rowcount == 1
    conn.commit()
    conn.close()
    return cancelled


def recover_interrupted_tasks():
    """Mark uncertain in-flight tasks failed instead of risking duplicate actions."""
    _ensure_db()
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "UPDATE night_queue SET status='failed', completed_at=? WHERE status='running'",
        (datetime.now(timezone.utc).isoformat(),),
    )
    recovered = c.rowcount
    conn.commit()
    conn.close()
    return recovered

# Functions for system notifications
def add_notification(message, source="system"):
    _ensure_db()
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO system_notifications (message, source, created_at) VALUES (?, ?, ?)",
              (message, source, str(datetime.now())))
    conn.commit()
    conn.close()

def get_unread_notifications():
    _ensure_db()
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM system_notifications WHERE read=0 ORDER BY created_at ASC")
    rows = c.fetchall()
    c.execute("UPDATE system_notifications SET read=1 WHERE read=0")
    conn.commit()
    conn.close()
    return [dict(row) for row in rows]

def peek_unread_notifications():
    _ensure_db()
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM system_notifications WHERE read=0 ORDER BY created_at ASC")
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def mark_notification_read(notification_id):
    _ensure_db()
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE system_notifications SET read=1 WHERE id=?", (notification_id,))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    _ensure_db()
    print("Database initialized.")
