import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.todoist_helper import TodoistHelper


def test_todoist_helper_is_enabled_when_token_can_be_resolved(monkeypatch):
    monkeypatch.setattr(TodoistHelper, "_resolve_api_token", lambda self: "token-from-test")

    helper = TodoistHelper()

    assert helper.api_token == "token-from-test"
    assert helper.is_enabled() is True


def test_todoist_helper_get_tasks_reads_from_configured_project(monkeypatch):
    monkeypatch.setattr(TodoistHelper, "_resolve_api_token", lambda self: "token-from-test")

    calls = []

    class FakeAPI:
        def get_tasks(self, **kwargs):
            calls.append(kwargs)
            return [SimpleNamespace(content="queue item", id="task-1")]

    helper = TodoistHelper()
    helper.api = FakeAPI()
    helper.data["Project ID"] = "ambient-project-id"

    tasks = helper.get_tasks()

    assert tasks == [{"content": "queue item", "id": "task-1"}]
    assert calls == [{"project_id": "ambient-project-id"}]


def test_todoist_helper_add_task_writes_to_inbox_without_project_id(monkeypatch):
    monkeypatch.setattr(TodoistHelper, "_resolve_api_token", lambda self: "token-from-test")

    calls = []

    class FakeAPI:
        def add_task(self, **kwargs):
            calls.append(kwargs)
            return {"id": "todo-1", "content": kwargs["content"]}

    helper = TodoistHelper()
    helper.api = FakeAPI()

    task = helper.add_task("agent reminder")

    assert task == {"id": "todo-1", "content": "agent reminder"}
    assert calls == [{"content": "agent reminder"}]


def test_todoist_helper_add_task_preserves_due_datetime_for_inbox(monkeypatch):
    monkeypatch.setattr(TodoistHelper, "_resolve_api_token", lambda self: "token-from-test")

    calls = []
    due_datetime = datetime.fromisoformat("2026-06-30T21:00:00+05:30")

    class FakeAPI:
        def add_task(self, **kwargs):
            calls.append(kwargs)
            return {"id": "todo-2", "content": kwargs["content"], "due_datetime": kwargs.get("due_datetime")}

    helper = TodoistHelper()
    helper.api = FakeAPI()

    task = helper.add_task("join the event tonight", due_datetime=due_datetime)

    assert task == {"id": "todo-2", "content": "join the event tonight", "due_datetime": due_datetime}
    assert calls == [{"content": "join the event tonight", "due_datetime": due_datetime}]
