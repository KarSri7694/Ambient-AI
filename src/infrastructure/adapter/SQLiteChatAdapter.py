import queue
import sqlite3
import threading
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteChatAdapter:
    """Durable chat sessions, messages, and pending direct-turn work."""

    def __init__(self, db_path: str):
        resolved_path = Path(db_path).expanduser()
        if resolved_path.exists() and resolved_path.is_dir():
            raise ValueError(
                "Chat db_path must point to a SQLite database file, not a directory: "
                f"{resolved_path}"
            )
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(resolved_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    reply_to_id TEXT,
                    message_kind TEXT NOT NULL DEFAULT 'chat',
                    task_id INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error_text TEXT,
                    FOREIGN KEY(session_id) REFERENCES chat_sessions(id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS chat_messages_session_idx "
                "ON chat_messages(session_id, created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS chat_messages_status_idx "
                "ON chat_messages(status, created_at)"
            )

    @staticmethod
    def _row(row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
        return dict(row) if row is not None else None

    def create_session(self, title: str = "New conversation") -> dict[str, Any]:
        now = _utc_now()
        session_id = uuid.uuid4().hex
        normalized = title.strip() or "New conversation"
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO chat_sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, normalized[:120], now, now),
            )
        return self.get_session(session_id)

    def get_session(self, session_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            return self._row(
                conn.execute("SELECT * FROM chat_sessions WHERE id=?", (session_id,)).fetchone()
            )

    def list_sessions(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT s.*,
                       (SELECT content FROM chat_messages m
                        WHERE m.session_id=s.id AND m.content != ''
                        ORDER BY m.created_at DESC, m.rowid DESC LIMIT 1) AS preview
                FROM chat_sessions s
                ORDER BY updated_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def rename_session(self, session_id: str, title: str) -> Optional[dict[str, Any]]:
        normalized = title.strip()
        if not normalized:
            raise ValueError("Session title cannot be empty.")
        with self._connect() as conn:
            conn.execute(
                "UPDATE chat_sessions SET title=?, updated_at=? WHERE id=?",
                (normalized[:120], _utc_now(), session_id),
            )
        return self.get_session(session_id)

    def list_messages(self, session_id: str, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chat_messages WHERE session_id=? "
                "ORDER BY created_at ASC, rowid ASC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_message(self, message_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            return self._row(
                conn.execute("SELECT * FROM chat_messages WHERE id=?", (message_id,)).fetchone()
            )

    def enqueue_turn(self, session_id: str, content: str) -> dict[str, Any]:
        normalized = content.strip()
        if not normalized:
            raise ValueError("Message cannot be empty.")
        if self.get_session(session_id) is None:
            raise KeyError(session_id)

        now = _utc_now()
        user_id = uuid.uuid4().hex
        assistant_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO chat_messages "
                "(id, session_id, role, content, status, created_at, updated_at) "
                "VALUES (?, ?, 'user', ?, 'completed', ?, ?)",
                (user_id, session_id, normalized, now, now),
            )
            conn.execute(
                "INSERT INTO chat_messages "
                "(id, session_id, role, content, status, reply_to_id, created_at, updated_at) "
                "VALUES (?, ?, 'assistant', '', 'queued', ?, ?, ?)",
                (assistant_id, session_id, user_id, now, now),
            )
            session = conn.execute(
                "SELECT title FROM chat_sessions WHERE id=?", (session_id,)
            ).fetchone()
            title = session["title"] if session else ""
            next_title = normalized[:72] if title == "New conversation" else title
            conn.execute(
                "UPDATE chat_sessions SET title=?, updated_at=? WHERE id=?",
                (next_title, now, session_id),
            )
        return {
            "user_message": self.get_message(user_id),
            "assistant_message": self.get_message(assistant_id),
        }

    def claim_next_turn(self) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM chat_messages WHERE role='assistant' AND status='queued' "
                "ORDER BY created_at ASC LIMIT 1"
            ).fetchone()
            if row is None:
                conn.commit()
                return None
            now = _utc_now()
            updated = conn.execute(
                "UPDATE chat_messages SET status='running', updated_at=? "
                "WHERE id=? AND status='queued'",
                (now, row["id"]),
            ).rowcount
            conn.commit()
            if updated != 1:
                return None
        claimed = self.get_message(row["id"])
        with self._connect() as conn:
            user_row = conn.execute(
                "SELECT * FROM chat_messages WHERE id=?", (row["reply_to_id"],)
            ).fetchone()
        claimed["user_message"] = dict(user_row)
        return claimed

    def has_queued_turn(self) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM chat_messages WHERE role='assistant' AND status='queued' LIMIT 1"
            ).fetchone()
        return row is not None

    def conversation_history(
        self,
        session_id: str,
        *,
        before_message_id: Optional[str] = None,
        limit: int = 40,
    ) -> list[dict[str, str]]:
        params: list[Any] = [session_id]
        before_clause = ""
        if before_message_id:
            before = self.get_message(before_message_id)
            if before:
                before_clause = "AND created_at < ?"
                params.append(before["created_at"])
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT role, content FROM chat_messages WHERE session_id=? "
                f"AND status='completed' AND role IN ('user', 'assistant') "
                f"AND content != '' {before_clause} ORDER BY created_at DESC, rowid DESC LIMIT ?",
                tuple(params),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def update_partial(self, message_id: str, content: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE chat_messages SET content=?, updated_at=? "
                "WHERE id=? AND status='running'",
                (content, _utc_now(), message_id),
            )

    def complete_message(self, message_id: str, content: str) -> None:
        now = _utc_now()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT session_id FROM chat_messages WHERE id=?", (message_id,)
            ).fetchone()
            conn.execute(
                "UPDATE chat_messages SET content=?, status='completed', updated_at=?, error_text=NULL "
                "WHERE id=?",
                (content, now, message_id),
            )
            if row:
                conn.execute(
                    "UPDATE chat_sessions SET updated_at=? WHERE id=?",
                    (now, row["session_id"]),
                )

    def fail_message(self, message_id: str, error_text: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE chat_messages SET status='failed', error_text=?, updated_at=? WHERE id=?",
                (error_text[:2000], _utc_now(), message_id),
            )

    def defer_message(self, message_id: str, reason: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE chat_messages SET status='queued', error_text=?, updated_at=? WHERE id=? AND status='running'",
                (reason[:2000], _utc_now(), message_id),
            )

    def append_scheduled_result(
        self,
        *,
        session_id: str,
        task_id: int,
        content: str,
        failed: bool = False,
    ) -> dict[str, Any]:
        now = _utc_now()
        message_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO chat_messages "
                "(id, session_id, role, content, status, message_kind, task_id, "
                "created_at, updated_at, error_text) "
                "VALUES (?, ?, 'assistant', ?, ?, 'scheduled_result', ?, ?, ?, ?)",
                (
                    message_id,
                    session_id,
                    content,
                    "failed" if failed else "completed",
                    task_id,
                    now,
                    now,
                    content if failed else None,
                ),
            )
            conn.execute(
                "UPDATE chat_sessions SET updated_at=? WHERE id=?",
                (now, session_id),
            )
        return self.get_message(message_id)

    def recover_interrupted(self) -> int:
        with self._connect() as conn:
            result = conn.execute(
                "UPDATE chat_messages SET status='failed', "
                "error_text='Ambient AI stopped before this response completed.', updated_at=? "
                "WHERE status='running'",
                (_utc_now(),),
            )
        return result.rowcount


class ChatEventBroker:
    """Thread-safe fan-out for live chat events across runtime and API loops."""

    def __init__(self, max_queue_size: int = 1000):
        self.max_queue_size = max_queue_size
        self._lock = threading.Lock()
        self._subscribers: Dict[str, list[queue.Queue]] = defaultdict(list)
        self._turn_enqueued_callback: Optional[Callable[[], None]] = None

    def set_turn_enqueued_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """Register a thread-safe wake-up hook for the runtime scheduler."""
        with self._lock:
            self._turn_enqueued_callback = callback

    def notify_turn_enqueued(self) -> None:
        with self._lock:
            callback = self._turn_enqueued_callback
        if callback is not None:
            callback()

    def subscribe(self, message_id: str) -> queue.Queue:
        subscriber: queue.Queue = queue.Queue(maxsize=self.max_queue_size)
        with self._lock:
            self._subscribers[message_id].append(subscriber)
        return subscriber

    def unsubscribe(self, message_id: str, subscriber: queue.Queue) -> None:
        with self._lock:
            subscribers = self._subscribers.get(message_id, [])
            if subscriber in subscribers:
                subscribers.remove(subscriber)
            if not subscribers:
                self._subscribers.pop(message_id, None)

    def publish(self, message_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            subscribers = list(self._subscribers.get(message_id, []))
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(dict(event))
            except queue.Full:
                try:
                    subscriber.get_nowait()
                    subscriber.put_nowait({"type": "snapshot_required"})
                except (queue.Empty, queue.Full):
                    pass
