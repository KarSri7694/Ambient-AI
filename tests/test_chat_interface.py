import asyncio
import json
import queue
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

import night_mode
from application.services.interaction_trace import interaction_trace
from application.services.llm_interaction_service import LLMInteractionService
from application.services.scheduled_task_service import ScheduledTaskService
from infrastructure.adapter.SQLiteChatAdapter import ChatEventBroker, SQLiteChatAdapter
from infrastructure.adapter.SQLiteTaskQueueAdapter import SQLiteTaskQueueAdapter
from infrastructure.runtime_log_server import RuntimeLogBuffer, create_runtime_log_app


class _FakeTaskQueue:
    def __init__(self):
        self.added = []

    def add_task(self, description, priority="medium", metadata=None, run_at_utc=None):
        self.added.append(
            {
                "description": description,
                "priority": priority,
                "metadata": metadata,
                "run_at_utc": run_at_utc,
            }
        )
        return "Task queued."


class _FakeToolBridge:
    async def start_servers(self, config_path):
        return None

    async def get_all_tools(self):
        return []

    async def execute_tool(self, tool_name, tool_args):
        return "ok"

    async def cleanup(self):
        return None


class _StreamingProvider:
    async def chat_completion_stream(self, **kwargs):
        async def _stream():
            delta = SimpleNamespace(content="hello", reasoning_content=None, tool_calls=None)
            yield SimpleNamespace(choices=[SimpleNamespace(delta=delta)])

        return _stream()


def test_chat_store_persists_resumable_sessions_and_turns(tmp_path):
    db_path = tmp_path / "chat.db"
    store = SQLiteChatAdapter(str(db_path))
    session = store.create_session()
    turn = store.enqueue_turn(session["id"], "Remember this conversation")

    claimed = store.claim_next_turn()
    assert claimed["id"] == turn["assistant_message"]["id"]
    assert claimed["user_message"]["content"] == "Remember this conversation"
    store.update_partial(claimed["id"], "Partial")
    store.complete_message(claimed["id"], "Completed answer")

    reopened = SQLiteChatAdapter(str(db_path))
    sessions = reopened.list_sessions()
    messages = reopened.list_messages(session["id"])

    assert sessions[0]["title"] == "Remember this conversation"
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[-1]["content"] == "Completed answer"
    assert reopened.conversation_history(session["id"]) == [
        {"role": "user", "content": "Remember this conversation"},
        {"role": "assistant", "content": "Completed answer"},
    ]


def test_chat_store_rejects_directory_as_database_path(tmp_path):
    with pytest.raises(ValueError, match="must point to a SQLite database file"):
        SQLiteChatAdapter(str(tmp_path))


def test_chat_store_marks_interrupted_responses_failed(tmp_path):
    store = SQLiteChatAdapter(str(tmp_path / "chat.db"))
    session = store.create_session()
    turn = store.enqueue_turn(session["id"], "Do something")
    store.claim_next_turn()

    assert store.recover_interrupted() == 1
    message = store.get_message(turn["assistant_message"]["id"])
    assert message["status"] == "failed"
    assert "stopped" in message["error_text"]


def test_loopback_chat_api_supports_terminal_sse_without_authentication(tmp_path):
    store = SQLiteChatAdapter(str(tmp_path / "chat.db"))
    broker = ChatEventBroker()
    app = create_runtime_log_app(
        RuntimeLogBuffer(),
        chat_store=store,
        chat_event_broker=broker,
    )
    client = TestClient(app)

    assert client.get("/api/chat/sessions").status_code == 200
    session = client.post("/api/chat/sessions", json={}).json()["session"]
    turn = client.post(
        f"/api/chat/sessions/{session['id']}/messages",
        json={"content": "hello"},
    ).json()
    assistant_id = turn["assistant_message"]["id"]
    store.complete_message(assistant_id, "world")

    response = client.get(f"/api/chat/messages/{assistant_id}/events")
    assert response.status_code == 200
    assert "event: snapshot" in response.text
    assert "event: done" in response.text
    assert "world" in response.text


def test_scheduled_task_service_normalizes_time_and_preserves_origin():
    task_queue = _FakeTaskQueue()
    service = ScheduledTaskService(task_queue)
    run_at = (datetime.now(timezone.utc) + timedelta(hours=2)).astimezone(
        timezone(timedelta(hours=5, minutes=30))
    )

    result = service.schedule(
        task="Check local reports",
        run_at=run_at.isoformat(),
        priority="high",
        metadata={"origin_chat_session_id": "session-1"},
    )

    assert result["status"] == "scheduled"
    assert result["run_at_utc"].endswith("+00:00")
    assert task_queue.added[0]["metadata"]["origin_chat_session_id"] == "session-1"
    assert task_queue.added[0]["run_at_utc"] == result["run_at_utc"]


def test_exact_time_queue_separates_idle_future_and_due_tasks(tmp_path):
    original_db_file = night_mode.DB_FILE
    night_mode.DB_FILE = str(tmp_path / "night_queue.db")
    try:
        adapter = SQLiteTaskQueueAdapter()
        now = datetime.now(timezone.utc)
        adapter.add_task("idle task")
        adapter.add_task("future task", run_at_utc=(now + timedelta(hours=1)).isoformat())
        adapter.add_task("due task", run_at_utc=(now - timedelta(minutes=1)).isoformat())

        assert [task.description for task in adapter.get_pending_tasks()] == ["idle task"]
        assert [task.description for task in adapter.get_due_tasks(now.isoformat())] == ["due task"]
        assert len(adapter.get_all_pending_tasks()) == 3
        due = adapter.get_due_tasks(now.isoformat())[0]
        assert adapter.claim_task(due.id) is True
        assert adapter.cancel_task(due.id) is False
    finally:
        night_mode.DB_FILE = original_db_file


def test_interaction_stream_callback_receives_text_deltas(tmp_path):
    service = LLMInteractionService(
        llm_provider=_StreamingProvider(),
        tool_bridge=_FakeToolBridge(),
        artifact_root=str(tmp_path / "artifacts"),
    )
    events = []

    result = asyncio.run(
        service.run_interaction(
            user_input="say hello",
            system_prompt="Answer.",
            model="model-a",
            event_callback=events.append,
        )
    )

    assert result == "hello"
    assert events == [{"type": "delta", "content": "hello"}]


def test_schedule_tool_interception_attaches_chat_metadata(tmp_path):
    scheduled_calls = []

    class _Scheduler:
        def schedule(self, **kwargs):
            scheduled_calls.append(kwargs)
            return {"status": "scheduled", "run_at_utc": kwargs["run_at"]}

    service = LLMInteractionService(
        llm_provider=_StreamingProvider(),
        tool_bridge=_FakeToolBridge(),
        scheduled_task_service=_Scheduler(),
        artifact_root=str(tmp_path / "artifacts"),
    )
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    call = {
        "id": "call-1",
        "type": "function",
        "function": {
            "name": "schedule_task_at",
            "arguments": json.dumps({"task": "check reports", "run_at": future}),
        },
    }

    with interaction_trace(
        "direct_chat",
        {"chat_session_id": "session-1", "chat_message_id": "message-1"},
    ):
        result = asyncio.run(service._execute_tool_calls([call]))

    assert result[0][0] == "schedule_task_at"
    assert scheduled_calls[0]["metadata"] == {
        "source": "direct_chat",
        "origin_chat_session_id": "session-1",
        "origin_chat_message_id": "message-1",
    }
