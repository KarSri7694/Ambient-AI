import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from application.services.llm_interaction_service import LLMInteractionService
from infrastructure.adapter.BrowserMCPToolAdapter import (
    BrowserMCPToolAdapter,
    BrowserMCPToolSession,
)
from infrastructure.adapter.MCPToolAdapter import MCPToolAdapter


def _tool(name: str, description: str = "") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
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


class _BrowserFlowProvider:
    def __init__(self, *, fail_child: bool = False, exit_browser: bool = True):
        self.current_model = "main-model"
        self.parent_model = "main-model"
        self.fail_child = fail_child
        self.exit_browser = exit_browser
        self.calls = []
        self.events = []
        self.browser_turn = 0

    def get_current_model(self):
        return self.current_model

    async def save_and_unload(self, messages):
        self.events.append("save_and_unload_parent")
        self.current_model = None
        return Path("saved-parent.bin")

    async def load_model(self, model_name):
        self.events.append(f"load:{model_name}")
        self.current_model = model_name

    async def unload_model(self):
        self.events.append(f"unload:{self.current_model}")
        self.current_model = None

    async def load_and_restore(self):
        self.events.append("restore_parent")
        self.current_model = self.parent_model
        return Path("saved-parent.bin")

    async def chat_completion_stream(self, *, model, messages, tools, image="", **kwargs):
        self.calls.append({"model": model, "messages": messages, "tools": tools})
        if self.fail_child:
            raise RuntimeError("child model failed")

        self.browser_turn += 1
        if self.browser_turn == 1:
            tool_call = SimpleNamespace(
                index=0,
                id="browser-call-1",
                function=SimpleNamespace(
                    name="browser_navigate",
                    arguments=json.dumps({"url": "https://example.com"}),
                ),
            )

            async def _tool_stream():
                yield _Chunk(tool_calls=[tool_call])

            return _tool_stream()

        finish_call = SimpleNamespace(
            index=0,
            id="finish-browser-task-1",
            function=SimpleNamespace(
                name="finish_browser_task",
                arguments=json.dumps(
                    {
                        "exit_browser": self.exit_browser,
                        "status": "completed",
                        "summary": "Opened example.com.",
                    }
                ),
            ),
        )

        async def _result_stream():
            yield _Chunk(tool_calls=[finish_call])

        return _result_stream()


class _SlowBrowserProvider(_BrowserFlowProvider):
    async def chat_completion_stream(self, *, model, messages, tools, image="", **kwargs):
        await asyncio.sleep(1)


class _MainToolBridge:
    async def start_servers(self, config_path):
        return None

    async def get_all_tools(self):
        return [_tool("use_browser"), _tool("demo")]

    async def execute_tool(self, tool_name, tool_args):
        return "main tool result"

    async def cleanup(self):
        return None


class _BrowserSession:
    def __init__(self):
        self.executions = []
        self.cleaned = False

    async def get_all_tools(self):
        return [_tool("browser_navigate"), _tool("browser_click")]

    async def execute_tool(self, tool_name, tool_args):
        self.executions.append((tool_name, tool_args))
        return "navigated"

    async def cleanup(self):
        self.cleaned = True


class _BrowserBridge:
    def __init__(self):
        self.headless_calls = []
        self.sessions = []

    async def open_session(self, *, headless):
        self.headless_calls.append(headless)
        session = _BrowserSession()
        self.sessions.append(session)
        return session


def test_use_browser_isolates_tools_and_restores_parent():
    provider = _BrowserFlowProvider()
    main_bridge = _MainToolBridge()
    browser_bridge = _BrowserBridge()
    service = LLMInteractionService(
        llm_provider=provider,
        tool_bridge=main_bridge,
        browser_tool_bridge=browser_bridge,
        browser_agent_model="browser-model",
        browser_headless=True,
    )

    asyncio.run(service.initialize_tools())
    assert [tool["function"]["name"] for tool in service._tools_for_agent_depth(0)] == [
        "use_browser",
        "demo",
    ]

    result = asyncio.run(
        service._execute_tool_calls(
            [
                {
                    "id": "use-browser-1",
                    "type": "function",
                    "function": {
                        "name": "use_browser",
                        "arguments": json.dumps(
                            {
                                "task": "Open example.com and report its title",
                                "headless": False,
                            }
                        ),
                    },
                }
            ],
            agent_depth=0,
        )
    )

    assert result == [
        (
            "use_browser",
            json.dumps(
                {
                    "status": "completed",
                    "summary": "Opened example.com.",
                    "browser_exited": True,
                }
            ),
        )
    ]
    assert browser_bridge.headless_calls == [True]
    assert browser_bridge.sessions[0].executions == [
        ("browser_navigate", {"url": "https://example.com"})
    ]
    assert browser_bridge.sessions[0].cleaned is True
    assert provider.current_model == "main-model"
    assert provider.events == [
        "save_and_unload_parent",
        "load:browser-model",
        "unload:browser-model",
        "restore_parent",
    ]
    assert all(
        [tool["function"]["name"] for tool in call["tools"]]
        == ["browser_navigate", "browser_click", "finish_browser_task"]
        for call in provider.calls
    )
    finish_schema = provider.calls[0]["tools"][-1]["function"]["parameters"]
    assert finish_schema["required"] == ["exit_browser", "status", "summary"]
    assert finish_schema["additionalProperties"] is False


def test_finish_browser_task_can_return_without_closing_browser():
    provider = _BrowserFlowProvider(exit_browser=False)
    browser_bridge = _BrowserBridge()
    service = LLMInteractionService(
        llm_provider=provider,
        tool_bridge=_MainToolBridge(),
        browser_tool_bridge=browser_bridge,
        browser_agent_model="browser-model",
        browser_headless=False,
    )

    result = asyncio.run(
        service._run_browser_agent(
            task="Open example.com and leave it open",
            agent_depth=0,
        )
    )

    assert json.loads(result) == {
        "status": "completed",
        "summary": "Opened example.com.",
        "browser_exited": False,
    }
    assert browser_bridge.sessions[0].cleaned is False
    assert service._retained_browser_sessions == [browser_bridge.sessions[0]]

    asyncio.run(service.cleanup_browser_sessions())

    assert browser_bridge.sessions[0].cleaned is True
    assert service._retained_browser_sessions == []


def test_browser_failure_still_cleans_up_and_restores_parent():
    provider = _BrowserFlowProvider(fail_child=True)
    browser_bridge = _BrowserBridge()
    service = LLMInteractionService(
        llm_provider=provider,
        tool_bridge=_MainToolBridge(),
        browser_tool_bridge=browser_bridge,
        browser_agent_model="browser-model",
    )

    with pytest.raises(RuntimeError, match="child model failed"):
        asyncio.run(
            service._run_browser_agent(
                task="Open example.com",
                agent_depth=0,
            )
        )

    assert browser_bridge.sessions[0].cleaned is True
    assert provider.current_model == "main-model"
    assert provider.events[-2:] == ["unload:browser-model", "restore_parent"]


def test_browser_timeout_still_cleans_up_and_restores_parent():
    provider = _SlowBrowserProvider()
    browser_bridge = _BrowserBridge()
    service = LLMInteractionService(
        llm_provider=provider,
        tool_bridge=_MainToolBridge(),
        browser_tool_bridge=browser_bridge,
        browser_agent_model="browser-model",
        browser_task_timeout_seconds=0.001,
        browser_headless=True,
    )

    with pytest.raises(TimeoutError):
        asyncio.run(
            service._run_browser_agent(
                task="Open example.com",
                agent_depth=0,
            )
        )

    assert browser_bridge.sessions[0].cleaned is True
    assert provider.current_model == "main-model"
    assert provider.events[-2:] == ["unload:browser-model", "restore_parent"]


def test_browser_agent_cannot_delegate_again():
    service = LLMInteractionService(
        llm_provider=_BrowserFlowProvider(),
        tool_bridge=_MainToolBridge(),
        browser_tool_bridge=_BrowserBridge(),
        browser_agent_model="browser-model",
    )
    service._tools = [_tool("use_browser"), _tool("browser_navigate")]

    assert [
        tool["function"]["name"] for tool in service._tools_for_agent_depth(1)
    ] == ["browser_navigate"]


def test_browser_session_builds_visible_and_headless_arguments(tmp_path):
    base_config = {"command": "npx", "args": ["-y", "@playwright/mcp@latest"]}
    visible = BrowserMCPToolSession(
        server_name="playwright",
        server_config=base_config,
        headless=False,
        profile_dir=tmp_path / "profile",
        denied_tool_names=set(),
    )
    headless = BrowserMCPToolSession(
        server_name="playwright",
        server_config=base_config,
        headless=True,
        profile_dir=tmp_path / "profile",
        denied_tool_names=set(),
    )

    visible_args = visible.launch_args()
    headless_args = headless.launch_args()

    assert "--user-data-dir" in visible_args
    assert str((tmp_path / "profile").resolve()) in visible_args
    assert "--headless" not in visible_args
    assert headless_args[-2:] == ["--headless", "--isolated"]
    assert "--user-data-dir" not in headless_args


class _AdvertisedBrowserSession:
    async def list_tools(self):
        return SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name="browser_navigate",
                    description="Navigate to a URL",
                    inputSchema={"properties": {}, "required": []},
                ),
                SimpleNamespace(
                    name="browser_run_code_unsafe",
                    description="RCE-equivalent arbitrary code execution",
                    inputSchema={"properties": {}, "required": []},
                ),
            ]
        )

    async def call_tool(self, tool_name, tool_args):
        return SimpleNamespace(content=[SimpleNamespace(text="ok")])


def test_browser_session_hides_and_rejects_unsafe_tools(tmp_path):
    session = BrowserMCPToolSession(
        server_name="playwright",
        server_config={"command": "npx", "args": []},
        headless=True,
        profile_dir=tmp_path / "profile",
        denied_tool_names={"browser_run_code"},
    )
    session._session = _AdvertisedBrowserSession()

    tools = asyncio.run(session.get_all_tools())
    assert [tool["function"]["name"] for tool in tools] == ["browser_navigate"]
    assert asyncio.run(session.execute_tool("browser_navigate", {})) == "ok"
    with pytest.raises(ValueError, match="not allowed"):
        asyncio.run(session.execute_tool("browser_run_code_unsafe", {}))


def test_browser_adapter_requires_browser_only_exposure(tmp_path):
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "playwright": {
                        "command": "npx",
                        "args": ["-y", "@playwright/mcp@latest"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    adapter = BrowserMCPToolAdapter(
        config_path=str(config_path),
        server_name="playwright",
        profile_dir=str(tmp_path / "profile"),
    )

    with pytest.raises(ValueError, match="exposure"):
        adapter._load_server_config()


def test_main_mcp_adapter_skips_browser_agent_servers(tmp_path):
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "playwright": {
                        "command": "command-that-must-not-run",
                        "args": [],
                        "exposure": "browser_agent",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    adapter = MCPToolAdapter()

    asyncio.run(adapter.start_servers(str(config_path)))
    tools = asyncio.run(adapter.get_all_tools())
    asyncio.run(adapter.cleanup())

    assert tools == []
