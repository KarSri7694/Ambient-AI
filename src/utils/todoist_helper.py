import json
import logging
import os
from pathlib import Path

from todoist_api_python.api import TodoistAPI

from config import CONFIG


TODOIST_CONFIG_PATH = Path(CONFIG.get_str("todoist", "project_state_path", "todoist.json"))
DEFAULT_PROJECT_NAME = CONFIG.get_str("todoist", "project_name", "Ambient AI Tasks")
MCP_CONFIG_PATH = Path(CONFIG.get_str("runtime", "mcp_config_path", "mcp.json"))


class TodoistHelper:
    """Helper class for Todoist operations."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.api_token = self._resolve_api_token()
        self.api = TodoistAPI(self.api_token) if self.api_token else None
        if TODOIST_CONFIG_PATH.exists():
            self.data = json.loads(TODOIST_CONFIG_PATH.read_text(encoding="utf-8"))
        else:
            self.data = {"Project": DEFAULT_PROJECT_NAME, "Project ID": ""}
        if not self.data.get("Project"):
            self.data["Project"] = DEFAULT_PROJECT_NAME

    def is_enabled(self) -> bool:
        return self.api is not None

    def _resolve_api_token(self) -> str:
        config_token = CONFIG.get_str("todoist", "api_token", "").strip()
        if config_token:
            return config_token

        env_token = os.environ.get("TODOIST_API_TOKEN", "").strip()
        if env_token:
            return env_token

        if MCP_CONFIG_PATH.exists():
            try:
                payload = json.loads(MCP_CONFIG_PATH.read_text(encoding="utf-8"))
                for server in payload.get("mcpServers", {}).values():
                    env_map = server.get("env", {}) or {}
                    token = str(env_map.get("TODOIST_API_TOKEN", "")).strip()
                    if token and not token.startswith("%"):
                        return token
            except Exception as exc:
                self.logger.warning("Failed to read Todoist token from %s: %s", MCP_CONFIG_PATH, exc)

        return ""

    def update_project_id(self, new_id):
        """Update the project ID in the todoist.json file."""
        self.data["Project ID"] = new_id
        TODOIST_CONFIG_PATH.write_text(json.dumps(self.data), encoding="utf-8")

    def _ensure_project_id(self):
        if self.api is None:
            return ""
        project_id = self.data.get("Project ID", "").strip()
        if project_id:
            return project_id

        project_name = self.data.get("Project", DEFAULT_PROJECT_NAME)
        try:
            projects = self.api.get_projects()
            for project in projects:
                if getattr(project, "name", "") == project_name:
                    self.update_project_id(project.id)
                    return project.id
            project = self.api.add_project(name=project_name)
            self.update_project_id(project.id)
            return project.id
        except Exception as exc:
            self.logger.warning("Error ensuring Todoist project '%s': %s", project_name, exc)
            return ""

    def get_tasks(self):
        """Fetch tasks from the configured Todoist project."""
        if self.api is None:
            return []
        try:
            project_id = self._ensure_project_id()
            if not project_id:
                return []
            tasks = self.api.get_tasks(project_id=project_id)
            normalized_tasks = []
            for task in self._flatten_task_items(tasks):
                content = getattr(task, "content", None)
                task_id = getattr(task, "id", None)
                if content is None or task_id is None:
                    continue
                normalized_tasks.append({"content": content, "id": task_id})
            return normalized_tasks
        except Exception as exc:
            self.logger.warning("Error fetching Todoist tasks: %s", exc)
            return []

    def _flatten_task_items(self, tasks):
        for task in tasks:
            if isinstance(task, (list, tuple)):
                yield from self._flatten_task_items(task)
            else:
                yield task

    def complete_task(self, task_id):
        """Mark a task as complete by its ID."""
        if self.api is None:
            return
        try:
            self.api.complete_task(task_id)
        except Exception as exc:
            self.logger.warning("Error completing Todoist task %s: %s", task_id, exc)

    def add_task(self, content: str, due_datetime=None):
        """Create a Todoist task in the user's Todoist Inbox."""
        if self.api is None:
            return None
        normalized = str(content or "").strip()
        if not normalized:
            return None
        try:
            kwargs = {
                "content": normalized,
            }
            if due_datetime is not None:
                kwargs["due_datetime"] = due_datetime
            return self.api.add_task(**kwargs)
        except Exception as exc:
            self.logger.warning("Error creating Todoist task %r: %s", normalized, exc)
            return None


if __name__ == "__main__":
    todoist_helper = TodoistHelper()
    todoist_helper._ensure_project_id()
    todoist_helper.get_tasks()
