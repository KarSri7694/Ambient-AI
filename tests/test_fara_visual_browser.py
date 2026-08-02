import json
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from application.services.llm_interaction_service import LLMInteractionService
from infrastructure.adapter.FaraVisualBrowserAdapter import (
    BrowserPolicyError,
    BrowserSafetyPolicy,
    FaraVisualBrowserSession,
)


class _UnusedLLM:
    pass


def _session(tmp_path: Path) -> FaraVisualBrowserSession:
    return FaraVisualBrowserSession(
        llm_provider=_UnusedLLM(),
        profile_dir=tmp_path / "profile",
        screenshot_dir=tmp_path / "screenshots",
        headless=True,
        viewport_width=1440,
        viewport_height=900,
        max_steps=10,
        settle_ms=0,
        search_url_template="https://duckduckgo.com/?q={query}",
        browser_channel="",
        browser_executable_path="",
        policy=BrowserSafetyPolicy(),
        screenshot_retention=True,
        logger=__import__("logging").getLogger("test-fara"),
    )


def test_browser_policy_allows_public_requests_including_post():
    policy = BrowserSafetyPolicy()
    assert policy.validate_navigation("https://example.com/products/running-shoe")
    assert policy.request_allowed(url="https://example.com/products/1", method="GET")[0]
    allowed, reason = policy.request_allowed(
        url="https://example.com/api/wishlist", method="POST"
    )
    assert allowed
    assert "public browser request" in reason


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8080/admin",
        "http://127.0.0.1/private",
        "http://169.254.169.254/latest/meta-data",
        "file:///C:/Users/user/.ssh/id_rsa",
    ],
)
def test_browser_policy_allows_non_download_navigation(url):
    assert BrowserSafetyPolicy().validate_navigation(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "https://shop.example/cart",
        "https://shop.example/account/login",
        "https://shop.example/checkout/payment",
    ],
)
def test_browser_policy_allows_transactional_public_navigation(url):
    assert BrowserSafetyPolicy().validate_navigation(url) == url


def test_fara_launch_args_start_maximized_with_configured_window_size(tmp_path):
    session = _session(tmp_path)

    kwargs = session._launch_kwargs()

    assert kwargs["viewport"] == {"width": 1440, "height": 900}
    assert "--start-maximized" in kwargs["args"]
    assert "--window-size=1440,900" in kwargs["args"]


def test_fara_action_parser_accepts_official_xml_shape(tmp_path):
    session = _session(tmp_path)
    action = session._parse_text_action(
        '<tool_call>{"name":"computer_use","arguments":{"action":"scroll","pixels":-600}}</tool_call>'
    )
    assert action == {"action": "scroll", "pixels": -600}


def test_fara_action_parser_recovers_qwen_xml_action_from_reasoning(tmp_path):
    session = _session(tmp_path)
    action = session._parse_text_action(
        """The next step is direct navigation.
<tool_call>
<function=computer_use>
<parameter=action>
visit_url", "url": "https://huggingface.co/bartowski/Fara1.5-27B-GGUF/tree/main"
</function>
</tool_call>"""
    )
    assert action == {
        "action": "visit_url",
        "url": "https://huggingface.co/bartowski/Fara1.5-27B-GGUF/tree/main",
    }


def test_fara_request_action_reads_reasoning_content(tmp_path):
    asyncio.run(_exercise_reasoning_content_action(tmp_path))


async def _exercise_reasoning_content_action(tmp_path):
    class _ReasoningLLM:
        async def chat_completion_stream(self, **kwargs):
            assert kwargs["tools"][0]["function"]["name"] == "computer_use"
            assert kwargs["chat_template_kwargs"] == {"enable_thinking": False}

            async def _stream():
                delta = SimpleNamespace(
                    content=None,
                    reasoning_content=(
                        "<tool_call><function=computer_use>"
                        "<parameter=action>web_search\", \"query\": \"Fara 1.5 27B GGUF\""
                        "</parameter></function></tool_call>"
                    ),
                    tool_calls=None,
                )
                yield SimpleNamespace(choices=[SimpleNamespace(delta=delta)])

            return _stream()

    screenshot = tmp_path / "screen.png"
    screenshot.write_bytes(b"not-a-real-png")
    session = _session(tmp_path)
    session.llm = _ReasoningLLM()

    action = await session._request_action(
        task="Find Fara GGUF files.",
        model="Fara1.5-27B",
        screenshot_path=screenshot,
        current_url="https://duckduckgo.com/",
        step=1,
    )

    assert action == {"action": "web_search", "query": "Fara 1.5 27B GGUF"}


def test_fara_final_step_instructs_terminate(tmp_path):
    asyncio.run(_exercise_final_step_instruction(tmp_path))


async def _exercise_final_step_instruction(tmp_path):
    class _FinalStepLLM:
        async def chat_completion_stream(self, **kwargs):
            state = json.loads(kwargs["messages"][1]["content"])
            assert state["is_final_step"] is True
            assert state["remaining_steps_after_this"] == 0
            assert "Return a terminate action now" in state["instruction"]

            async def _stream():
                tool_call = SimpleNamespace(
                    index=0,
                    id="finish",
                    function=SimpleNamespace(
                        name="computer_use",
                        arguments=json.dumps(
                            {
                                "action": "terminate",
                                "status": "completed",
                                "answer": "Final browser result.",
                            }
                        ),
                    ),
                )
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(
                        content=None,
                        reasoning_content=None,
                        tool_calls=[tool_call],
                    ))]
                )

            return _stream()

    screenshot = tmp_path / "screen.png"
    screenshot.write_bytes(b"not-a-real-png")
    session = _session(tmp_path)
    session.llm = _FinalStepLLM()

    action = await session._request_action(
        task="Find options.",
        model="Qwen3.6-35B",
        screenshot_path=screenshot,
        current_url="https://duckduckgo.com/",
        step=session.max_steps,
    )

    assert action["action"] == "terminate"
    assert action["answer"] == "Final browser result."


def test_fara_action_validation_is_coordinate_bounded(tmp_path):
    session = _session(tmp_path)
    assert session._validate_action(
        {"action": "left_click", "coordinate": [100, 200]}
    )["coordinate"] == [100.0, 200.0]
    assert session._validate_action(
        {"action": "triple_click", "coordinate": [100, 200]}
    )["coordinate"] == [100.0, 200.0]
    drag = session._validate_action(
        {"action": "left_click_drag", "start_coordinate": [100, 200], "end_coordinate": [300, 400]}
    )
    assert drag["start_coordinate"] == [100.0, 200.0]
    assert drag["end_coordinate"] == [300.0, 400.0]
    with pytest.raises(ValueError, match="outside the viewport"):
        session._validate_action(
            {"action": "left_click", "coordinate": [2000, 200]}
        )
    with pytest.raises(ValueError, match="not allowed"):
        session._validate_action({"action": "run_javascript", "text": "alert(1)"})


def test_fara_validation_recovers_malformed_structured_action_field(tmp_path):
    session = _session(tmp_path)
    action = session._validate_action(
        {
            "action": (
                'visit_url", "coordinate": [831, 64], '
                '"text": "https://huggingface.co/bartowski/Fara1.5-27B-GGUF/tree/main", '
                '"press_enter": true, "delete_existing_text": true}'
            )
        }
    )

    assert action["action"] == "visit_url"
    assert action["url"] == "https://huggingface.co/bartowski/Fara1.5-27B-GGUF/tree/main"
    assert action["text"] == "https://huggingface.co/bartowski/Fara1.5-27B-GGUF/tree/main"
    assert action["press_enter"] is True
    assert action["delete_existing_text"] is True


def test_visual_backend_source_has_no_dom_or_uia_grounding():
    source = Path(
        "src/infrastructure/adapter/FaraVisualBrowserAdapter.py"
    ).read_text(encoding="utf-8")
    forbidden_calls = (
        ".locator(",
        ".evaluate(",
        ".content(",
        ".inner_text(",
        "accessibility.snapshot",
        "UIATAdapter",
    )
    assert not [value for value in forbidden_calls if value in source]


class _VisualSession:
    def __init__(self):
        self.cleaned = False
        self.task = ""

    async def run_task(self, *, task, model, event_callback=None):
        self.task = task
        return json.dumps(
            {
                "status": "completed",
                "task_summary": "Found two alternatives.",
                "candidates": [
                    {"title": "Example shoe", "url": "https://example.com/shoe"}
                ],
            }
        )

    async def get_all_tools(self):
        raise AssertionError("Visual sessions must not expose generic browser tools")

    async def execute_tool(self, tool_name, tool_args):
        raise AssertionError("Visual sessions execute their own loop")

    async def cleanup(self):
        self.cleaned = True


class _VisualBridge:
    def __init__(self):
        self.session = _VisualSession()

    async def open_session(self, *, headless):
        return self.session


class _RootTools:
    async def get_all_tools(self):
        return []

    async def execute_tool(self, tool_name, tool_args):
        return ""

    async def cleanup(self):
        return None


class _ModelProvider:
    def __init__(self):
        self.current = "parent-model"
        self.saved = False
        self.restored = False

    def get_current_model(self):
        return self.current

    async def save_and_unload(self, messages):
        self.saved = True
        self.current = ""
        return Path("saved-parent-state.bin")

    async def load_model(self, model):
        self.current = model

    async def unload_model(self):
        self.current = ""

    async def load_and_restore(self):
        self.current = "parent-model"
        self.restored = True
        return Path("saved-parent-state.bin")


def test_llm_service_uses_visual_runner_without_generic_child_agent():
    asyncio.run(_exercise_visual_runner())


async def _exercise_visual_runner():
    provider = _ModelProvider()
    bridge = _VisualBridge()
    service = LLMInteractionService(
        llm_provider=provider,
        tool_bridge=_RootTools(),
        browser_tool_bridge=bridge,
        browser_agent_model="Fara1.5-27B",
        browser_task_timeout_seconds=5,
    )

    result = await service.deploy_browser_agent(
        task="Find visually similar running shoes and record public product links."
    )

    payload = json.loads(result)
    assert payload["status"] == "completed"
    assert bridge.session.task.startswith("Find visually similar")
    assert bridge.session.cleaned
    assert provider.saved and provider.restored
