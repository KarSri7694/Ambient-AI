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
from application.services.llm_interaction_service import InteractionSuspended, LLMInteractionService
from application.services.interaction_trace import interaction_trace
from infrastructure.adapter.SQLiteAutonomyAdapter import SQLiteAutonomyAdapter
from infrastructure.adapter.SQLiteChatAdapter import SQLiteChatAdapter
from local_control.computer import ComputerControlSession
from local_control.filesystem import FilesystemControlSession
from local_control.safety import PathGrantError, UnsafeComputerAction
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


class _ComputerToolProvider(_Provider):
    async def chat_completion_stream(self, *, model, messages, tools, image="", **kwargs):
        self.calls.append({"model": model, "messages": json.loads(json.dumps(messages)), "tools": tools, "image": image})
        tool_messages = [message for message in messages if message.get("role") == "tool"]
        if not tool_messages:
            call = SimpleNamespace(
                index=0,
                id="inspect-1",
                function=SimpleNamespace(name="computer_inspect", arguments="{}"),
            )
        else:
            last = tool_messages[-1]["content"]
            assert "Permission denied" not in last
            call = SimpleNamespace(
                index=0,
                id="finish-1",
                function=SimpleNamespace(
                    name="finish_computer_task",
                    arguments=json.dumps({"status": "completed", "summary": "screen inspected"}),
                ),
            )

        async def stream():
            yield _Chunk(tool_calls=[call])

        return stream()


class _FilesystemPromptProvider(_Provider):
    async def chat_completion_stream(self, *, model, messages, tools, image="", **kwargs):
        self.calls.append({"model": model, "messages": json.loads(json.dumps(messages)), "tools": tools})
        call = SimpleNamespace(
            index=0,
            id="finish-fs-1",
            function=SimpleNamespace(
                name="finish_filesystem_task",
                arguments=json.dumps({"status": "completed", "summary": "filesystem checked"}),
            ),
        )

        async def stream():
            yield _Chunk(tool_calls=[call])

        return stream()


class _DelegationProvider(_Provider):
    async def chat_completion_stream(self, *, model, messages, tools, image="", **kwargs):
        self.calls.append({"model": model, "messages": json.loads(json.dumps(messages)), "tools": tools})
        if model == "computer-model":
            call = SimpleNamespace(
                index=0, id="finish-1",
                function=SimpleNamespace(
                    name="finish_computer_task",
                    arguments=json.dumps({"status": "completed", "summary": "The document is saved"}),
                ),
            )
            async def child_stream():
                yield _Chunk(tool_calls=[call])
            return child_stream()
        if messages[-1].get("role") == "tool":
            assert messages[-1]["tool_call_id"] == "request-1"
            async def resumed_stream():
                yield _Chunk(content="The approved desktop task completed, and the document was saved.")
            return resumed_stream()
        call = SimpleNamespace(
            index=0, id="request-1",
            function=SimpleNamespace(
                name="request_computer_use",
                arguments=json.dumps({
                    "task": "Click save", "reason": "User asked",
                    "expected_result": "The document is saved",
                    "continuation_instruction": "Tell the user whether saving succeeded",
                }),
            ),
        )
        async def request_stream():
            yield _Chunk(tool_calls=[call])
        return request_stream()


def _suspend_computer(service, *, metadata=None):
    with interaction_trace("direct_chat", metadata or {}), pytest.raises(InteractionSuspended) as caught:
        asyncio.run(service.run_interaction(
            user_input="Save my document", system_prompt="Help the user.", model="main-model"
        ))
    return caught.value


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


class _ContinuationService:
    def __init__(self):
        self.restored = False

    def available_tool_definitions(self):
        return []

    def restore_conversation(self, *, system_prompt, messages):
        self.restored = True

    def reset_context(self):
        return None

    async def run_interaction(self, **kwargs):
        return "The approved desktop task completed, and the original workflow is now finished."


class _FailAfterControlService:
    def __init__(self):
        self.calls = 0

    async def deploy_computer_agent(self, *, task, approval_id, event_callback=None):
        self.calls += 1
        if event_callback:
            event_callback({"type": "tool_started", "tool_name": "computer_click"})
        raise RuntimeError("desktop control disconnected")


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


def test_filesystem_agent_resolves_relative_grants_before_prompting(tmp_path, monkeypatch):
    provider = _FilesystemPromptProvider()
    service = LLMInteractionService(
        llm_provider=provider,
        tool_bridge=_Bridge(),
        filesystem_agent_model="main-model",
    )
    monkeypatch.chdir(tmp_path)
    (tmp_path / "note.txt").write_text("proactive sweep gmail", encoding="utf-8")

    result = asyncio.run(
        service._run_filesystem_agent(
            task="Search for proactive sweep files.",
            granted_paths=["."],
            agent_depth=0,
        )
    )

    assert "filesystem checked" in result
    user_messages = [
        message["content"]
        for message in provider.calls[-1]["messages"]
        if message["role"] == "user"
    ]
    assert json.dumps(str(tmp_path.resolve()))[1:-1] in user_messages[-1]
    assert '"."' not in user_messages[-1]


def test_computer_session_allows_single_win_key_but_blocks_dangerous_chords(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "local_control.computer.ComputerControlSession._pyautogui_call",
        lambda _self, method, *args: calls.append((method, args)) or "ok",
    )
    session = ComputerControlSession(max_actions=3)
    try:
        assert asyncio.run(session.execute_tool("computer_press_key", {"key": "win"})) == "ok"
        assert calls == [("press", ("win",))]
        with pytest.raises(UnsafeComputerAction):
            asyncio.run(session.execute_tool("computer_press_key", {"key": "win+r"}))
    finally:
        asyncio.run(session.cleanup())


def test_computer_mouse_coordinates_are_scaled_from_gemma_1000_grid(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "local_control.computer.ComputerControlSession._screen_size",
        staticmethod(lambda: (2560, 1440)),
    )
    monkeypatch.setattr(
        "local_control.computer.ComputerControlSession._pyautogui_call",
        lambda _self, method, *args: calls.append((method, args)) or "ok",
    )
    session = ComputerControlSession(max_actions=3)
    try:
        assert asyncio.run(session.execute_tool("computer_move_mouse", {"x": 500, "y": 500})) == "ok"
        assert asyncio.run(session.execute_tool("computer_click", {"x": 1000, "y": 1000})) == "ok"
        assert asyncio.run(session.execute_tool("computer_click", {"x": 1200, "y": -100})) == "ok"
    finally:
        asyncio.run(session.cleanup())

    assert calls == [
        ("moveTo", (1280, 720)),
        ("click", (2559, 1439)),
        ("click", (2559, 0)),
    ]


def test_computer_session_exposes_visual_desktop_actions(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "local_control.computer.ComputerControlSession._screen_size",
        staticmethod(lambda: (1000, 1000)),
    )
    monkeypatch.setattr(
        "local_control.computer.ComputerControlSession._pyautogui_call",
        lambda _self, method, *args, **kwargs: calls.append((method, args, kwargs)) or "ok",
    )
    session = ComputerControlSession(max_actions=5)
    try:
        tool_names = {tool["function"]["name"] for tool in asyncio.run(session.get_all_tools())}
        assert {"computer_double_click", "computer_right_click", "computer_drag", "computer_wait"} <= tool_names
        assert asyncio.run(session.execute_tool("computer_double_click", {"x": 100, "y": 200})) == "ok"
        assert asyncio.run(session.execute_tool("computer_right_click", {"x": 300, "y": 400})) == "ok"
        assert asyncio.run(session.execute_tool("computer_drag", {"start_x": 10, "start_y": 20, "end_x": 30, "end_y": 40})) == "ok"
    finally:
        asyncio.run(session.cleanup())

    assert calls[0][0] == "doubleClick"
    assert calls[1][0] == "rightClick"
    assert [item[0] for item in calls[2:]] == ["moveTo", "dragTo"]


def test_request_computer_use_creates_pending_approval(tmp_path):
    store = SQLiteAutonomyAdapter(str(tmp_path / "autonomy.db"))
    service = LLMInteractionService(
        llm_provider=_DelegationProvider(),
        tool_bridge=_Bridge(),
        capability_policy=CapabilityPolicyService(store=store),
        computer_agent_model="computer-model",
        computer_enabled=True,
    )

    suspended = _suspend_computer(service)
    approvals = store.list_approvals(status="pending")
    assert len(approvals) == 1
    assert suspended.approval_id == approvals[0].approval_id
    assert approvals[0].capability == "computer.use"
    delegated = store.get_delegated_task_by_approval(approvals[0].approval_id)
    assert delegated.checkpoint_json
    assert delegated.suspended_tool_call_id == "request-1"


def test_request_computer_use_allows_negated_risky_words_for_approval(tmp_path):
    store = SQLiteAutonomyAdapter(str(tmp_path / "autonomy.db"))
    service = LLMInteractionService(
        llm_provider=_DelegationProvider(),
        tool_bridge=_Bridge(),
        capability_policy=CapabilityPolicyService(store=store),
        computer_agent_model="computer-model",
        computer_enabled=True,
    )

    with pytest.raises(InteractionSuspended) as caught:
        service._request_computer_use(
            task="Read visible WhatsApp messages only. Do not type, send, delete, or download anything.",
            reason="Approved read-only proactive check",
            agent_depth=0,
        )

    assert caught.value.approval.capability == "computer.use"
    assert caught.value.delegated_task.task.startswith("Read visible WhatsApp")


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


def test_approved_computer_agent_can_use_internal_tools_without_reapproval(tmp_path, monkeypatch):
    store = SQLiteAutonomyAdapter(str(tmp_path / "autonomy.db"))
    provider = _ComputerToolProvider()
    service = LLMInteractionService(
        llm_provider=provider,
        tool_bridge=_Bridge(),
        capability_policy=CapabilityPolicyService(store=store),
        computer_agent_model="computer-model",
        computer_enabled=True,
    )
    monkeypatch.setattr(
        "local_control.computer.ComputerControlSession._inspect",
        lambda _self: "Foreground window: test app",
    )
    screenshots = [str(tmp_path / "screen-1.png"), str(tmp_path / "screen-2.png")]
    for path in screenshots:
        Path(path).write_bytes(b"fake")
    monkeypatch.setattr(
        "local_control.computer.ComputerControlSession.capture_screenshot_for_model",
        lambda _self: screenshots.pop(0) if screenshots else str(tmp_path / "screen-last.png"),
    )

    result = asyncio.run(service.deploy_computer_agent(task="Inspect the screen", approval_id="approved-1"))

    assert "screen inspected" in result
    assert len(provider.calls) == 2
    assert provider.calls[0]["image"].endswith("screen-1.png")
    assert provider.calls[1]["image"].endswith("screen-2.png")
    assert provider.calls[1]["messages"][-1]["role"] == "user"
    assert "Fresh screenshot" in provider.calls[1]["messages"][-1]["content"]
    assert store.list_approvals(status="pending") == []


def test_computer_agent_reuses_same_resident_model_without_swapping(tmp_path, monkeypatch):
    provider = _ComputerToolProvider()
    service = LLMInteractionService(
        llm_provider=provider,
        tool_bridge=_Bridge(),
        computer_agent_model="main-model",
        computer_enabled=True,
    )
    screenshots = [str(tmp_path / "same-model-1.png"), str(tmp_path / "same-model-2.png")]
    for path in screenshots:
        Path(path).write_bytes(b"fake")
    monkeypatch.setattr(
        "local_control.computer.ComputerControlSession.capture_screenshot_for_model",
        lambda _self: screenshots.pop(0) if screenshots else str(tmp_path / "same-model-last.png"),
    )

    result = asyncio.run(service.deploy_computer_agent(task="Inspect the screen"))

    assert "screen inspected" in result
    assert provider.current_model == "main-model"
    assert provider.events == []


def test_delegated_result_resumes_originating_chat(tmp_path):
    store = SQLiteAutonomyAdapter(str(tmp_path / "autonomy.db"))
    chat_store = SQLiteChatAdapter(str(tmp_path / "chat.db"))
    session = chat_store.create_session("Delegation")
    provider = _DelegationProvider()
    service = LLMInteractionService(
        llm_provider=provider,
        tool_bridge=_Bridge(),
        capability_policy=CapabilityPolicyService(store=store),
        computer_agent_model="computer-model",
        computer_enabled=True,
    )
    turn = chat_store.enqueue_turn(session["id"], "Save my document")
    assistant_id = turn["assistant_message"]["id"]
    chat_store.claim_next_turn()
    suspended = _suspend_computer(
        service,
        metadata={"chat_session_id": session["id"], "chat_message_id": assistant_id},
    )
    chat_store.mark_awaiting_approval(assistant_id, "Waiting for approval")

    approval = store.get_approval(suspended.approval_id)
    delegation = store.get_delegated_task_by_approval(approval.approval_id)
    assert delegation is not None
    assert delegation.origin_kind == "direct_chat"
    store.decide_approval(approval.approval_id, approved=True, approver="local_user")
    store.update_delegated_task(delegation.delegation_id, status="approved")
    details = json.loads(approval.constraints_json)
    now = datetime.now(timezone.utc).isoformat()
    store.enqueue_event(
        AmbientEvent(
            event_id="approval-event-new", event_type="approval_granted",
            source_kind="local_approval", source_ref=approval.approval_id,
            occurred_at=now, payload_json=json.dumps(details), confidence=1.0,
            privacy_label="private", fingerprint="approval-new-fingerprint",
            priority=1.0, available_at=now,
        )
    )
    coordinator = AutonomyCoordinatorService(
        store=store, judgment=_Judgment(), policy=CapabilityPolicyService(store=store),
        mode="active", chat_store=chat_store,
    )

    executed = asyncio.run(
        coordinator.process_next(model="main-model", llm_service=service, personalization_context="")
    )
    assert executed["outcome"] == "delegation_completed"
    assert store.get_delegated_task(delegation.delegation_id).status == "completed"

    # A fresh service instance proves the authoritative message frame survived restart.
    resumed_service = LLMInteractionService(
        llm_provider=provider,
        tool_bridge=_Bridge(),
        capability_policy=CapabilityPolicyService(store=store),
        computer_agent_model="computer-model",
        computer_enabled=True,
    )
    resumed = asyncio.run(
        coordinator.process_next(
            model="main-model", llm_service=resumed_service, personalization_context=""
        )
    )
    assert resumed["outcome"] == "delegation_continued"
    messages = chat_store.list_messages(session["id"])
    assistant_messages = [message for message in messages if message["role"] == "assistant"]
    assert len(assistant_messages) == 1
    assert assistant_messages[0]["id"] == assistant_id
    assert assistant_messages[0]["message_kind"] == "delegated_result"
    assert "document was saved" in assistant_messages[0]["content"]
    resumed_call = provider.calls[-1]
    assert resumed_call["messages"][-1]["role"] == "tool"
    assert resumed_call["messages"][-1]["tool_call_id"] == "request-1"
    assert store.get_delegated_task(delegation.delegation_id).status == "continued"


def test_post_control_failure_is_not_automatically_retried(tmp_path):
    store = SQLiteAutonomyAdapter(str(tmp_path / "autonomy.db"))
    request_service = LLMInteractionService(
        llm_provider=_DelegationProvider(), tool_bridge=_Bridge(),
        capability_policy=CapabilityPolicyService(store=store),
        computer_agent_model="computer-model", computer_enabled=True,
    )
    _suspend_computer(request_service)
    approval = store.list_approvals(status="pending")[0]
    delegation = store.get_delegated_task_by_approval(approval.approval_id)
    store.decide_approval(approval.approval_id, approved=True, approver="local_user")
    store.update_delegated_task(delegation.delegation_id, status="approved")
    now = datetime.now(timezone.utc).isoformat()
    store.enqueue_event(
        AmbientEvent(
            event_id="failure-approval-event", event_type="approval_granted",
            source_kind="local_approval", source_ref=approval.approval_id,
            occurred_at=now, payload_json=approval.constraints_json, confidence=1.0,
            privacy_label="private", fingerprint="failure-approval-fingerprint",
            priority=1.0, available_at=now,
        )
    )
    coordinator = AutonomyCoordinatorService(
        store=store, judgment=_Judgment(), policy=CapabilityPolicyService(store=store),
        mode="active",
    )
    failing = _FailAfterControlService()

    result = asyncio.run(
        coordinator.process_next(model="main-model", llm_service=failing, personalization_context="")
    )

    assert result["outcome"] == "delegation_failed"
    assert failing.calls == 1
    failed = store.get_delegated_task(delegation.delegation_id)
    assert failed.status == "failed"
    assert failed.control_started_at is not None
    assert store.event_counts()["pending"] == 1
