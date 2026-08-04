"""Run Ambient AI against isolated demo data and replayed project images.

The launcher starts the normal ``src/app.py`` entrypoint in a child process,
but injects a temporary config before app import. No real user database,
memory, capture directory, browser profile, or runtime lock is used.

Usage:
    python scripts/run_demo.py
    python scripts/run_demo.py --images-dir test_images --spacing-seconds 6
    python scripts/run_demo.py --keep-temp
"""

from __future__ import annotations

import argparse
import configparser
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
LOGGER = logging.getLogger("ambient_demo")


DEMO_USER_INFO = """# Demo User Profile

- This is a synthetic user profile for the isolated Ambient AI demonstration.
- The user is evaluating an ambient computer agent and prefers concise, actionable summaries.
- The user is interested in software development, browser research, and measurable workflow improvements.
- The user allows the demo agent to use browser and computer agents for the demonstrated tasks.
"""

DEMO_MEMORY = """# Demo Working Memory

- Demo objective: observe replayed desktop states and turn useful visual context into bounded agent work.
- Synthetic project: Ambient AI visual processing and computer-agent demonstration.
- Open loop: identify visible work, verify useful facts, and report what the agent completed.
- All data in this run is synthetic and must remain inside the temporary demo directory.
"""


BOOTSTRAP = r'''
import os
import runpy
import sys
from pathlib import Path

repo_root = Path(os.environ["AMBIENT_DEMO_REPO"])
src_root = repo_root / "src"
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(src_root))

import config
config.CONFIG = config.AppConfig(Path(os.environ["AMBIENT_DEMO_CONFIG"]))

# The demo must never capture the real desktop. The normal capture loop keeps
# running, but its adapter fails before touching mss and the app's existing
# error handling leaves replayed durable events as the only visual input.
from infrastructure.adapter.MSSScreenCaptureAdapter import MssScreenCaptureAdapter
def _demo_capture_disabled(self, output_path=None):
    raise RuntimeError("live screen capture is disabled for isolated demo replay")
MssScreenCaptureAdapter.capture_screenshot = _demo_capture_disabled

# app.py normally protects the repository runtime with a project lock. The
# demo has its own temporary process scope and must not touch .ambient_data.
import real_world_testing.runtime_lock as runtime_lock
class _DemoRuntimeLock:
    def __init__(self, path, owner):
        self.path = path
        self.owner = owner
    def acquire(self):
        return None
    def release(self):
        return None
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False
runtime_lock.RuntimeOwnershipLock = _DemoRuntimeLock

runpy.run_path(str(repo_root / "src" / "app.py"), run_name="__main__")
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.ini")
    parser.add_argument("--images-dir", type=Path, default=REPO_ROOT / "test_images")
    parser.add_argument("--spacing-seconds", type=float, default=6.0)
    parser.add_argument("--model", default="", help="Optional override for all primary demo models")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def discover_images(images_dir: Path) -> list[Path]:
    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Demo image directory does not exist: {images_dir}. "
            "Add deterministic images under test_images/ before running the demo."
        )
    images = sorted(
        (path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.name.lower(),
    )
    if not images:
        raise FileNotFoundError(f"No supported images found in {images_dir}")
    from PIL import Image
    for path in images:
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            raise ValueError(f"Image is not readable: {path}") from exc
    return images


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def absolute_config_path(value: str, *, base: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return str(path.resolve())


def configure_demo(base_config_path: Path, demo_root: Path, model_override: str) -> Path:
    parser = configparser.ConfigParser()
    if not parser.read(base_config_path, encoding="utf-8"):
        raise FileNotFoundError(f"Config file does not exist: {base_config_path}")

    database_root = demo_root / "database"
    memory_root = demo_root / "memory"
    capture_root = demo_root / "captures"
    path_overrides = {
        ("runtime", "user_data_dir"): str(demo_root),
        ("privacy", "capture_root"): str(capture_root),
        ("chat", "db_path"): str(database_root / "chat.db"),
        ("benchmarking", "db_path"): str(database_root / "benchmarking.db"),
        ("training_data", "root"): str(demo_root / "training"),
        ("training_data", "db_path"): str(database_root / "training_data.db"),
        ("audio", "uploads_dir"): str(demo_root / "audio" / "uploads"),
        ("audio", "transcriptions_dir"): str(demo_root / "audio" / "transcriptions"),
        ("audio", "cleaned_audio_dir"): str(demo_root / "audio" / "cleaned"),
        ("audio", "temp_audio_dir"): str(demo_root / "audio" / "temp"),
        ("todoist", "project_state_path"): str(demo_root / "todoist.json"),
        ("browser", "persistent_profile_dir"): str(demo_root / "browser" / "profile"),
        ("browser", "screenshot_dir"): str(demo_root / "browser" / "screenshots"),
        ("computer", "screenshot_dir"): str(demo_root / "computer" / "screenshots"),
        ("reflection", "history_path"): str(demo_root / "reflection" / "history.json"),
        ("artifacts", "archive_dir"): str(demo_root / "artifacts" / "archive"),
    }
    for (section, option), value in path_overrides.items():
        if not parser.has_section(section):
            parser.add_section(section)
        parser.set(section, option, value)

    parser.set("runtime", "always_on", "true")
    parser.set("runtime", "perform_queue_tasks", "true")
    parser.set("runtime", "debug_mode", "true")
    parser.set("runtime", "enable_parallel_image_processing", "true")
    parser.set("runtime", "parallel_image_workers", "3")
    parser.set("passive_observer", "enabled", "true")
    parser.set("passive_observer", "capture_interval_seconds", "86400")
    parser.set("passive_observer", "visual_context_batch_size", "1")
    parser.set("computer", "enabled", "true")
    parser.set("browser", "headless", "false")
    parser.set("autonomy", "mode", "active")
    parser.set("todoist", "enabled", "false")
    parser.set("log_api", "host", "127.0.0.1")
    parser.set("log_api", "port", str(free_port()))
    if model_override:
        for option in (
            "default_model", "passive_observer_model", "full_passive_observer_model",
            "passive_followup_model", "followup_execution_model", "chat_model",
            "browser_agent_model", "computer_agent_model",
        ):
            parser.set("runtime" if option == "default_model" else "models", option, model_override)

    config_path = demo_root / "config.ini"
    with config_path.open("w", encoding="utf-8") as handle:
        parser.write(handle)
    (memory_root).mkdir(parents=True, exist_ok=True)
    (demo_root / "MEMORY.md").write_text(DEMO_MEMORY, encoding="utf-8")
    (demo_root / "USER_INFO.md").write_text(DEMO_USER_INFO, encoding="utf-8")
    return config_path


def start_app(config_path: Path, demo_root: Path) -> subprocess.Popen:
    environment = os.environ.copy()
    environment["AMBIENT_DEMO_CONFIG"] = str(config_path)
    environment["AMBIENT_DEMO_REPO"] = str(REPO_ROOT)
    kwargs: dict[str, Any] = {
        "cwd": str(REPO_ROOT),
        "env": environment,
        "stdin": None,
        "stdout": None,
        "stderr": None,
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    return subprocess.Popen([sys.executable, "-c", BOOTSTRAP], **kwargs)


def wait_for_database(database_path: Path, process: subprocess.Popen, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"app.py exited before initializing the demo database (exit={process.returncode})")
        if database_path.exists():
            return
        time.sleep(0.25)
    raise TimeoutError(f"Timed out waiting for demo database: {database_path}")


def approve_pending(database_path: Path, stop_event: threading.Event, summary: dict[str, int]) -> None:
    from infrastructure.adapter.SQLiteAutonomyAdapter import SQLiteAutonomyAdapter
    while not stop_event.is_set():
        try:
            store = SQLiteAutonomyAdapter(str(database_path))
            for approval in store.list_approvals(status="pending", limit=500):
                if store.decide_approval(
                    approval.approval_id,
                    approved=True,
                    approver="demo_auto_approval",
                ):
                    summary["approvals_auto_approved"] += 1
                    LOGGER.info(
                        "Auto-approved demo capability=%s approval=%s",
                        approval.capability,
                        approval.approval_id,
                    )
        except Exception:
            LOGGER.exception("Demo approval poll failed")
        stop_event.wait(0.5)


def inject_sequence(images: list[Path], database_path: Path, capture_root: Path, spacing: float, summary: dict[str, int]) -> None:
    from scripts.inject_visual_captures import inject_images
    base_time = datetime.now(timezone.utc)
    for index, image in enumerate(images):
        result = inject_images(
            [image],
            autonomy_db=database_path,
            capture_root=capture_root,
            app_name="Ambient AI Demo",
            window_title=f"Demo replay: {image.name}",
            accessible_text="Synthetic demo visual capture. Use visible evidence only.",
            start_at=base_time + timedelta(seconds=index * max(0.001, spacing)),
            spacing_seconds=0.001,
        )[0]
        summary["images_injected"] += 1
        LOGGER.info("Injected image %s/%s: %s at %s", index + 1, len(images), image.name, result.occurred_at)
        if index + 1 < len(images):
            time.sleep(max(0.0, spacing))


def summarize(database_path: Path, summary: dict[str, int]) -> dict[str, Any]:
    from infrastructure.adapter.SQLiteAutonomyAdapter import SQLiteAutonomyAdapter
    try:
        store = SQLiteAutonomyAdapter(str(database_path))
        summary["pending_approvals"] = len(store.list_approvals(status="pending", limit=500))
        summary["delegations"] = len(store.list_delegated_tasks(limit=500)) if hasattr(store, "list_delegated_tasks") else 0
        return {"counters": summary, "event_counts": store.event_counts()}
    except Exception as exc:
        return {"counters": summary, "summary_error": str(exc)}


def stop_child(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.send_signal(getattr(signal, "CTRL_BREAK_EVENT", signal.SIGINT))
        else:
            process.send_signal(signal.SIGINT)
        process.wait(timeout=35)
    except (subprocess.TimeoutExpired, OSError):
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    images = discover_images(args.images_dir.resolve())
    temp_path = Path(tempfile.mkdtemp(prefix="ambient-ai-demo-"))
    process: subprocess.Popen | None = None
    approval_stop = threading.Event()
    approval_summary = {"approvals_auto_approved": 0}
    approval_thread: threading.Thread | None = None
    summary = {"images_injected": 0, **approval_summary, "pending_approvals": 0, "delegations": 0}
    started = time.monotonic()
    try:
        config_path = configure_demo(args.config.resolve(), temp_path, args.model.strip())
        database_path = temp_path / "database" / "autonomy.db"
        capture_root = temp_path / "captures"
        LOGGER.info("Starting isolated Ambient AI demo in %s", temp_path)
        LOGGER.info("Replaying %s images from %s", len(images), args.images_dir.resolve())
        process = start_app(config_path, temp_path)
        wait_for_database(database_path, process)
        approval_thread = threading.Thread(
            target=approve_pending,
            args=(database_path, approval_stop, approval_summary),
            name="DemoApprovalLoop",
            daemon=True,
        )
        approval_thread.start()
        inject_sequence(images, database_path, capture_root, args.spacing_seconds, summary)
        LOGGER.info("All demo images injected. App remains active; press Ctrl+C to stop.")
        while process.poll() is None:
            time.sleep(0.5)
    except KeyboardInterrupt:
        LOGGER.info("Stopping demo after Ctrl+C")
    finally:
        approval_stop.set()
        if process is not None:
            stop_child(process)
        if approval_thread is not None:
            approval_thread.join(timeout=2)
        summary["approvals_auto_approved"] = approval_summary["approvals_auto_approved"]
        summary.update(summarize(temp_path / "database" / "autonomy.db", summary).get("counters", {}))
        summary["runtime_seconds"] = round(time.monotonic() - started, 2)
        LOGGER.info("Demo summary: %s", json.dumps(summary, sort_keys=True))
        if args.keep_temp:
            LOGGER.info("Preserved temporary demo directory: %s", temp_path)
        else:
            shutil.rmtree(temp_path, ignore_errors=True)
            LOGGER.info("Removed temporary demo directory")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
