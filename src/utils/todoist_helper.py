from todoist_api_python.api import TodoistAPI
import os
import json

TODOIST_API_TOKEN = os.getenv("TODOIST_API_TOKEN")
api = TodoistAPI(TODOIST_API_TOKEN)

data = None

class TodoistHelper:
    """Helper class for Todoist operations."""
    
    def __init__(self):
        global data
        data = json.loads(open("todoist.json").read())

    def update_project_id(self, new_id):
        """Update the project ID in the todoist.json file."""
        data['Project ID'] = new_id
        with open("todoist.json", "w") as f:
            json.dump(data, f)
            
    def get_tasks(self):
        """Fetch tasks from the specified Todoist project."""
        try:
            tasks = api.get_tasks(project_id=data['Project ID'])
            task_list = []
            for task in tasks:
                for t in task:
                    task_list.append({"content": t.content, "id": t.id})
            return task_list
        except Exception as e:
            print(f"Error fetching tasks: {e}")
            return []

    def complete_task(self,task_id):
        """Mark a task as complete by its ID."""
        try:
            api.complete_task(task_id)
            print(f"Task {task_id} marked as complete.")
        except Exception as e:
            print(f"Error completing task {task_id}: {e}")

if __name__ == "__main__":
    todoist_helper = TodoistHelper()
    if data['Project ID'] == "":
        project = api.add_project(name="Ambient AI Tasks")
        todoist_helper.update_project_id(project.id)

    todoist_helper.get_tasks()