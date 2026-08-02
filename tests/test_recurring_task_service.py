from types import SimpleNamespace

from application.services.recurring_task_service import RecurringTaskService
from infrastructure.adapter.SQLiteAutonomyAdapter import SQLiteAutonomyAdapter


def _service(tmp_path, toasts=None):
    return RecurringTaskService(
        autonomy_store=SQLiteAutonomyAdapter(str(tmp_path / "autonomy.db")),
        absence_threshold_minutes=5,
        toast_notifier=lambda title, message: (toasts if toasts is not None else []).append((title, message)),
    )


def _observation(text: str, observation_id: str = "obs-1"):
    return SimpleNamespace(
        observation_id=observation_id,
        app_name="Browser",
        summary=text,
        detailed_description=text,
        inferred_user_activity="watching download",
        completed_items=[],
        raw_payload_json="{}",
    )


def test_recurring_task_persists_and_can_pause_resume(tmp_path):
    service = _service(tmp_path)
    task = service.create(
        title="Daily notes", instruction="Create a local note", task_kind="interval", interval_seconds=60
    )
    assert service.due_tasks()[0].task_id == task.task_id
    assert service.set_status(task.task_id, "paused").status == "paused"
    assert service.due_tasks() == []
    assert service.set_status(task.task_id, "active").status == "active"


def test_visual_monitor_waits_for_absence_then_toasts(tmp_path):
    toasts = []
    service = _service(tmp_path, toasts)
    task = service.create(
        title="Movie download", instruction="Monitor the movie download", task_kind="monitor",
        monitor_condition="download finished", interval_seconds=10,
    )
    service.evaluate_visual_observation(_observation("The movie download is finished", "one"), user_idle=True, idle_seconds=30)
    assert not toasts
    assert service.store.list_inbox_items(limit=10)[0].title.startswith("Monitor complete")
    service.evaluate_visual_observation(_observation("The movie download is finished", "two"), user_idle=True, idle_seconds=301)
    assert len(toasts) == 1
    assert service.list(status="completed")[0].task_id == task.task_id


def test_active_user_visual_confirmation_suppresses_toast(tmp_path):
    toasts = []
    service = _service(tmp_path, toasts)
    task = service.create(
        title="Movie download", instruction="Monitor the movie download", task_kind="monitor",
        monitor_condition="download finished",
    )
    service.evaluate_visual_observation(_observation("download finished", "one"), user_idle=True, idle_seconds=30)
    assert service.mark_visual_completion_seen(_observation("download finished", "two")) == 1
    assert not toasts
    assert service.list(status="completed")[0].task_id == task.task_id


def test_todoist_sync_requires_ambient_label(tmp_path):
    service = _service(tmp_path)
    result = service.sync_todoist_tasks(
        [
            {"id": "normal", "content": "do not import", "labels": []},
            {"id": "ambient", "content": "every 2 hours create a local note", "labels": ["ambient"]},
        ],
        label="ambient",
    )
    assert result["created"] == 1
    task = service.list()[0]
    assert task.origin_kind == "todoist"
    assert task.interval_seconds == 7200
