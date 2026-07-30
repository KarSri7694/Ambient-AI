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
