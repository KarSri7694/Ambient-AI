import sqlite3
import json
from datetime import datetime

DB_FILE = "database/night_queue.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Create Tasks Table
    c.execute('''CREATE TABLE IF NOT EXISTS night_queue
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  description TEXT,
                  priority TEXT,
                  status TEXT,
                  created_at TEXT,
                  meta_data TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS system_notifications
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  message TEXT,
                  source TEXT, 
                  read INTEGER DEFAULT 0,
                  created_at TEXT)''')
    conn.commit()
    conn.close()
#Functions for managing night shift tasks
def add_task(description, priority="medium"):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO night_queue (description, priority, status, created_at) VALUES (?, ?, ?, ?)",
              (description, priority, "pending", str(datetime.now())))
    conn.commit()
    conn.close()
    return "Task queued."

def get_pending_tasks():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row # Allows accessing columns by name
    c = conn.cursor()
    c.execute("SELECT * FROM night_queue WHERE status='pending' ORDER BY created_at ASC")
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def mark_task_complete(task_id, status="completed"):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE night_queue SET status=? WHERE id=?", (status, task_id))
    conn.commit()
    conn.close()

# Functions for system notifications
def add_notification(message, source="system"):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO system_notifications (message, source, created_at) VALUES (?, ?, ?)",
              (message, source, str(datetime.now())))
    conn.commit()
    conn.close()

def get_unread_notifications():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM system_notifications WHERE read=0 ORDER BY created_at ASC")
    rows = c.fetchall()
    c.execute("UPDATE system_notifications SET read=1 WHERE read=0")
    conn.commit()
    conn.close()
    return [dict(row) for row in rows]

def mark_notification_read(notification_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE system_notifications SET read=1 WHERE id=?", (notification_id,))
    conn.commit()
    conn.close()
# Run init once
if __name__ == "__main__":
    init_db()
    print("Database initialized.")