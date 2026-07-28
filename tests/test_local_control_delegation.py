import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from application.services.autonomy_coordinator_service import AutonomyCoordinatorService
from application.services.capability_policy_service import CapabilityPolicyService
from application.services.llm_interaction_service import LLMInteractionService
from infrastructure.adapter.SQLiteAutonomyAdapter import SQLiteAutonomyAdapter
from local_control.filesystem import FilesystemControlSession
from local_control.safety import PathGrantError
from core.models import AmbientEvent


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


class _Delta:
    def __init__(self, *, content=None, tool_calls=None):
        self.content = content
        self.reasoning_content = None
        self.tool_calls = tool_calls


class _Chunk:
    def __init__(self, *, content=None, tool_calls=None):
        self.choices = [SimpleNamespace(delta=_Delta(content=content, tool_calls=tool_calls))]


class _Provider:
    def __init__(self):
        self.current_model = "main-model"
        self.parent_model = "main-model"
        self.events = []
        self.calls = []

    def get_current_model(self):
        return self.current_model

    async def save_and_unload(self, messages):
        self.events.append("save")
        self.current_model = None
        return Path("saved.bin")

    async def load_model(self, model_name):
        self.events.append(f"load:{model_name}")
        self.current_model = model_name

    async def unload_model(self):
        self.events.append(f"unload:{self.current_model}")
        self.current_model = None

    async def load_and_restore(self):
        self.events.append("restore")
        self.current_model = self.parent_model
        return Path("saved.bin")

    async def chat_completion_stream(self, *, model, messages, tools, image="", **kwargs):
        self.calls.append({"model": model, "tools": tools})
        finish = "finish_computer_task" if model == "computer-model" else "finish_filesystem_task"
        tool_call = SimpleNamespace(
            index=0,
            id="finish-1",
            function=SimpleNamespace(
                name=finish,
                arguments=json.dumps({"status": "completed", "summary": "done"}),
            ),
        )

        async def stream():
            yield _Chunk(tool_calls=[tool_call])

        return stream()


class _Bridge:
    async def start_servers(self, config_path):
        return None

    async def get_all_tools(self):
        return [_tool("request_computer_use"), _tool("use_filesystem")]

    async def execute_tool(self, tool_name, tool_args):
        return "ok"

    async def cleanup(self):
        return None


class _Judgment:
    def qualifies_for_enrichment(self, candidate):
        return True


def test_filesystem_session_rejects_ungranted_paths(tmp_path):
    granted = tmp_path / "granted"
    outside = tmp_path / "outside"
    granted.mkdir()
    outside.mkdir()
    (granted / "note.txt").write_text("hello", encoding="utf-8")
    (outside / "secret.txt").write_text("secret", encoding="utf-8")

    session = FilesystemControlSession(granted_paths=[str(granted)])

    assert asyncio.run(session.execute_tool("fs_read_text", {"path": str(granted / "note.txt")})) == "hello"
    with pytest.raises(PathGrantError):
        asyncio.run(session.execute_tool("fs_read_text", {"path": str(outside / "secret.txt")}))


def test_request_computer_use_creates_pending_approval(tmp_path):
    store = SQLiteAutonomyAdapter(str(tmp_path / "autonomy.db"))
    service = LLMInteractionService(
        llm_provider=_Provider(),
        tool_bridge=_Bridge(),
        capability_policy=CapabilityPolicyService(store=store),
        computer_agent_model="computer-model",
        computer_enabled=True,
    )

    result = asyncio.run(
        service._execute_tool_calls(
            [
                {
                    "id": "request-1",
                    "type": "function",
                    "function": {
                        "name": "request_computer_use",
                        "arguments": json.dumps({"task": "Click save", "reason": "User asked"}),
                    },
                }
            ],
            agent_depth=0,
        )
    )

    payload = json.loads(result[0][1])
    approvals = store.list_approvals(status="pending")
    assert payload["status"] == "awaiting_user_approval"
    assert len(approvals) == 1
    assert approvals[0].capability == "computer.use"


def test_approval_event_deploys_computer_agent_once(tmp_path):
    store = SQLiteAutonomyAdapter(str(tmp_path / "autonomy.db"))
    provider = _Provider()
    service = LLMInteractionService(
        llm_provider=provider,
        tool_bridge=_Bridge(),
        capability_policy=CapabilityPolicyService(store=store),
        computer_agent_model="computer-model",
        computer_enabled=True,
    )
    coordinator = AutonomyCoordinatorService(
        store=store,
        judgment=_Judgment(),
        policy=CapabilityPolicyService(store=store),
        mode="active",
    )
    now = datetime.now(timezone.utc).isoformat()
    event = AmbientEvent(
        event_id="approval-event",
        event_type="approval_granted",
        source_kind="local_approval",
        source_ref="approval-1",
        occurred_at=now,
        payload_json=json.dumps(
            {
                "tool_name": "request_computer_use",
                "approval_kind": "computer_use_deployment",
                "arguments": {"task": "Click save", "reason": "User asked"},
            }
        ),
        confidence=1.0,
        privacy_label="private",
        fingerprint="approval-fingerprint",
        priority=1.0,
        available_at=now,
    )
    store.enqueue_event(event)

    result = asyncio.run(
        coordinator.process_next(
            model="main-model",
            llm_service=service,
            personalization_context="",
        )
    )

    assert result["outcome"] == "computer_use_completed"
    assert provider.events == ["save", "load:computer-model", "unload:computer-model", "restore"]
    assert [call["model"] for call in provider.calls] == ["computer-model"]
