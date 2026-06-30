import asyncio
import queue
import sys
import tempfile
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

audio_agent_stub = types.ModuleType("audio_agent")


class _StubAudioAgentService:
    def __init__(self, *args, **kwargs):
        pass

    def get_transcription_queue(self):
        return queue.Queue()

    def start_service(self):
        return None


audio_agent_stub.AudioAgentService = _StubAudioAgentService
sys.modules.setdefault("audio_agent", audio_agent_stub)

sqlite_voice_stub = types.ModuleType("infrastructure.adapter.SQLiteVoiceAdapter")


class _StubSQLiteVoiceAdapter:
    def __init__(self, *args, **kwargs):
        pass


sqlite_voice_stub.SQLiteVoiceAdapter = _StubSQLiteVoiceAdapter
sys.modules.setdefault("infrastructure.adapter.SQLiteVoiceAdapter", sqlite_voice_stub)

from app import AmbientRuntime
from application.services.llm_interaction_service import LLMInteractionService
from infrastructure.adapter.MCPToolAdapter import MCPToolAdapter


class _FakeLLMProvider:
    def __init__(self):
        self.calls = []
        self.reports = []

    async def chat_completion_stream(self, *args, **kwargs):
        self.calls.append(
            {
                "model": kwargs.get("model") or (args[0] if args else None),
                "messages": kwargs.get("messages") or (args[1] if len(args) > 1 else None),
            }
        )

        class _Delta:
            def __init__(self, content):
                self.content = content
                self.reasoning_content = None
                self.tool_calls = None

        class _Choice:
            def __init__(self, delta):
                self.delta = delta

        class _Chunk:
            def __init__(self, content):
                self.choices = [_Choice(_Delta(content))]

        response = (
            '{"title": "Reminder Created", "summary": "Created a reminder.", "detailed_report": "A very detailed report about the reminder creation."}'
            if len(self.calls) > 1
            else "created reminder successfully"
        )

        async def _gen():
            yield _Chunk(response)

        return _gen()

    def attach_report(self, interaction_run_id, report):
        self.reports.append((interaction_run_id, report))


class _FakeToolBridge:
    def __init__(self):
        self.start_calls = 0
        self.get_calls = 0
        self.cleanup_calls = 0

    async def start_servers(self, config_path: str) -> None:
        self.start_calls += 1

    async def get_all_tools(self):
        self.get_calls += 1
        return [{"type": "function", "function": {"name": "demo", "parameters": {"type": "object", "properties": {}}}}]

    async def execute_tool(self, tool_name, tool_args):
        return "ok"

    async def cleanup(self):
        self.cleanup_calls += 1


class _FakeModelManager:
    def __init__(self):
        self.load_calls = 0
        self.unload_calls = 0
        self.loaded_models = []

    async def load_model(self, model_name: str) -> None:
        self.load_calls += 1
        self.loaded_models.append(model_name)

    async def unload_model(self) -> None:
        self.unload_calls += 1


def test_llm_interaction_service_caches_tool_definitions():
    bridge = _FakeToolBridge()
    service = LLMInteractionService(
        llm_provider=_FakeLLMProvider(),
        tool_bridge=bridge,
    )

    asyncio.run(service.initialize_tools())
    asyncio.run(service.initialize_tools())

    assert bridge.get_calls == 1


def test_mcp_tool_adapter_start_servers_is_idempotent():
    adapter = MCPToolAdapter()
    sentinel_stack = object()
    adapter._exit_stack = sentinel_stack
    adapter._sessions = {"existing": object()}

    asyncio.run(adapter.start_servers("does-not-matter.json"))

    assert adapter._exit_stack is sentinel_stack
    assert list(adapter._sessions.keys()) == ["existing"]


def test_ambient_runtime_separates_mcp_lifetime_from_model_lifetime():
    runtime = AmbientRuntime(transcription_queue=queue.Queue())
    llm = _FakeModelManager()
    bridge = _FakeToolBridge()
    service = LLMInteractionService(
        llm_provider=_FakeLLMProvider(),
        tool_bridge=bridge,
    )

    asyncio.run(runtime._initialize_mcp_tools(bridge, service))
    initialized = asyncio.run(
        runtime._ensure_runtime(
            llm_adapter=llm,
            services_initialized=False,
            reason="first load",
            model_name="model-a",
        )
    )
    still_initialized = asyncio.run(
        runtime._release_runtime(
            llm_adapter=llm,
            services_initialized=initialized,
            reason="idle unload",
        )
    )

    assert bridge.start_calls == 1
    assert bridge.get_calls == 1
    assert bridge.cleanup_calls == 0
    assert llm.load_calls == 1
    assert llm.unload_calls == 1
    assert llm.loaded_models == ["model-a"]
    assert still_initialized is False


def test_ambient_runtime_stop_only_signals_shutdown():
    runtime = AmbientRuntime(transcription_queue=queue.Queue())

    runtime.stop_service()

    assert runtime.stop_event.is_set()


def test_llm_interaction_service_uses_reporter_model_override():
    provider = _FakeLLMProvider()
    with tempfile.TemporaryDirectory() as tmpdir:
        service = LLMInteractionService(
            llm_provider=provider,
            tool_bridge=_FakeToolBridge(),
            reporter_model="reporter-model",
            artifact_root=tmpdir,
        )

        result = asyncio.run(
            service.run_interaction(
                user_input="set a reminder for tomorrow",
                system_prompt="Do the task.",
                model="execution-model",
                report_policy="auto_surface",
            )
        )

        assert result == "created reminder successfully"
        assert [call["model"] for call in provider.calls] == ["execution-model", "reporter-model"]
        assert provider.reports
        report = provider.reports[0][1]
        assert report["title"] == "Reminder Created"
        assert report["summary"] == "Created a reminder."
        artifact_path = Path(report["artifact_path"])
        assert artifact_path.exists()
        assert "A very detailed report" in artifact_path.read_text(encoding="utf-8")
        second_messages = provider.calls[1]["messages"]
        user_payload = next(msg["content"] for msg in second_messages if msg["role"] == "user")
        assert "interaction_history" in user_payload


def test_llm_interaction_service_same_model_report_uses_compact_payload():
    provider = _FakeLLMProvider()
    with tempfile.TemporaryDirectory() as tmpdir:
        service = LLMInteractionService(
            llm_provider=provider,
            tool_bridge=_FakeToolBridge(),
            reporter_model="execution-model",
            artifact_root=tmpdir,
        )
        asyncio.run(
            service.run_interaction(
                user_input="check the latest update",
                system_prompt="Do the task.",
                model="execution-model",
                report_policy="auto_surface",
            )
        )

        assert [call["model"] for call in provider.calls] == ["execution-model", "execution-model"]
        second_messages = provider.calls[1]["messages"]
        user_payload = next(msg["content"] for msg in second_messages if msg["role"] == "user")
        assert "task_brief" in user_payload
        assert "interaction_history" not in user_payload
