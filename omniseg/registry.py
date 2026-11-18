"""Discovers segmentation tasks from the repo's tasks/ folder.

Each tasks/<id>/task.json describes one trained model (body region, nnU-Net
dataset/fold/checkpoint, input channels, labels). No Qt dependency here so it
stays easy to test and to reuse from scripts outside the desktop app.
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = REPO_ROOT / 'tasks'


def list_tasks(tasks_dir=TASKS_DIR):
    """Returns all valid tasks found under tasks_dir, sorted by id."""
    tasks = []
    if not tasks_dir.is_dir():
        return tasks
    for entry in sorted(tasks_dir.iterdir()):
        task_file = entry / 'task.json'
        if entry.is_dir() and task_file.is_file():
            task = json.loads(task_file.read_text())
            task.setdefault('id', entry.name)
            task['_dir'] = str(entry)
            tasks.append(task)
    return tasks


def get_task(task_id, tasks_dir=TASKS_DIR):
    """Returns the task dict matching task_id, or None if not found."""
    for task in list_tasks(tasks_dir):
        if task['id'] == task_id:
            return task
    return None
