from todoist_api_python.api import TodoistAPI
import os
import json
from pathlib import Path

TODOIST_API_TOKEN = os.getenv("TODOIST_API_TOKEN")
api = TodoistAPI(TODOIST_API_TOKEN)
TODOIST_CONFIG_PATH = Path("todoist.json")
DEFAULT_PROJECT_NAME = "Ambient AI Tasks"

data = None

class TodoistHelper:
    """Helper class for Todoist operations."""
    
    def __init__(self):
        global data
        if TODOIST_CONFIG_PATH.exists():
            data = json.loads(TODOIST_CONFIG_PATH.read_text(encoding="utf-8"))
        else:
            data = {"Project": DEFAULT_PROJECT_NAME, "Project ID": ""}
        if not data.get("Project"):
            data["Project"] = DEFAULT_PROJECT_NAME

    def update_project_id(self, new_id):
        """Update the project ID in the todoist.json file."""
        data['Project ID'] = new_id
        TODOIST_CONFIG_PATH.write_text(json.dumps(data), encoding="utf-8")

    def _ensure_project_id(self):
        project_id = data.get("Project ID", "").strip()
        if project_id:
            return project_id

        project_name = data.get("Project", DEFAULT_PROJECT_NAME)
        try:
            projects = api.get_projects()
            for project in projects:
                if getattr(project, "name", "") == project_name:
                    self.update_project_id(project.id)
                    return project.id
            project = api.add_project(name=project_name)
            self.update_project_id(project.id)
            return project.id
        except Exception as e:
            print(f"Error ensuring Todoist project '{project_name}': {e}")
            return ""
            
    def get_tasks(self):
        """Fetch tasks from the specified Todoist project."""
        try:
            project_id = self._ensure_project_id()
            if not project_id:
                return []
            tasks = api.get_tasks(project_id=project_id)
            normalized_tasks = []
            for task in self._flatten_task_items(tasks):
                content = getattr(task, "content", None)
                task_id = getattr(task, "id", None)
                if content is None or task_id is None:
                    continue
                normalized_tasks.append({"content": content, "id": task_id})
            return normalized_tasks
        except Exception as e:
            print(f"Error fetching tasks: {e}")
            return []

    def _flatten_task_items(self, tasks):
        for task in tasks:
            if isinstance(task, (list, tuple)):
                yield from self._flatten_task_items(task)
            else:
                yield task

    def complete_task(self,task_id):
        """Mark a task as complete by its ID."""
        try:
            api.complete_task(task_id)
            print(f"Task {task_id} marked as complete.")
        except Exception as e:
            print(f"Error completing task {task_id}: {e}")

if __name__ == "__main__":
    todoist_helper = TodoistHelper()
    todoist_helper._ensure_project_id()
    todoist_helper.get_tasks()
