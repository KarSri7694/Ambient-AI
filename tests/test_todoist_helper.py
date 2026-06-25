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

fake_todoist_module = types.ModuleType("todoist_api_python")
fake_api_module = types.ModuleType("todoist_api_python.api")


class _ImportTimeApi:
    def __init__(self, *_args, **_kwargs):
        pass


fake_api_module.TodoistAPI = _ImportTimeApi
sys.modules.setdefault("todoist_api_python", fake_todoist_module)
sys.modules.setdefault("todoist_api_python.api", fake_api_module)

from utils import todoist_helper


class _Task:
    def __init__(self, task_id, content):
        self.id = task_id
        self.content = content


class _FakeTodoistApi:
    def get_projects(self):
        return [types.SimpleNamespace(id="ambient-project", name="Ambient AI Tasks")]

    def add_project(self, name):
        return types.SimpleNamespace(id="ambient-project", name=name)

    def get_tasks(self, project_id):
        self.last_project_id = project_id
        return [[_Task("1", "first task"), _Task("2", "second task")]]


class TodoistHelperTests(unittest.TestCase):
    def test_get_tasks_flattens_nested_api_response(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "todoist.json"
            config_path.write_text('{"Project": "Ambient AI Tasks", "Project ID": ""}', encoding="utf-8")
            fake_api = _FakeTodoistApi()
            with patch.object(todoist_helper, "TODOIST_CONFIG_PATH", config_path):
                with patch.object(todoist_helper, "api", fake_api):
                    helper = todoist_helper.TodoistHelper()
                    tasks = helper.get_tasks()

            self.assertEqual(
                tasks,
                [
                    {"content": "first task", "id": "1"},
                    {"content": "second task", "id": "2"},
                ],
            )
            self.assertEqual(fake_api.last_project_id, "ambient-project")


if __name__ == "__main__":
    unittest.main()
