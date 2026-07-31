import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from application.services.runtime_interrupt_service import RuntimeInterruptController, WorkInterrupted


def test_runtime_interrupt_controller_marks_active_work_interrupted():
    controller = RuntimeInterruptController()

    with pytest.raises(WorkInterrupted):
        with controller.active(kind="visual_perception", model="vision-model"):
            controller.request_interrupt("skip current image")
            controller.check()

    status = controller.status()
    assert status["requested"] is False
    assert status["active_work"] is None
    assert status["last_interrupted"]["kind"] == "visual_perception"
    assert status["last_interrupted"]["reason"] == "skip current image"


def test_runtime_interrupt_controller_clears_request_when_work_scope_consumes_interrupt():
    controller = RuntimeInterruptController()

    with controller.active(kind="autonomy_backlog"):
        controller.request_interrupt("skip current event")
        try:
            controller.check()
        except WorkInterrupted:
            pass

    status = controller.status()
    assert status["requested"] is False
    assert status["active_work"] is None
    assert status["last_interrupted"]["kind"] == "autonomy_backlog"
