import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from application.services.capability_policy_service import CapabilityPolicyService, CapabilityRegistry
from application.services.proactive_sweep_service import ProactiveSweepService
from infrastructure.adapter.SQLiteAutonomyAdapter import SQLiteAutonomyAdapter
from local_control.computer import ComputerControlSession


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }


class _LlmService:
    def __init__(self):
        self.allowed_tool_names = None

    def available_tool_definitions(self):
        return [
            _tool("gmail_search"),
            _tool("gmail_send"),
            _tool("calendar_list_events"),
            _tool("calendar_create_event"),
            _tool("search_web_ddgs"),
        ]

    def reset_context(self):
        return None

    async def run_interaction(self, **kwargs):
        self.allowed_tool_names = set(kwargs["allowed_tool_names"])
        return json.dumps(
            {
                "findings": [
                    {
                        "source": "gmail",
                        "title": "Important email needs review",
                        "summary": "A recent unread message appears to need a reply.",
                        "importance": "high",
                        "evidence": ["Unread message found by Gmail search."],
                        "suggested_next_step": "Review the message and decide whether to reply.",
                        "requires_user_action": True,
                        "confidence": 0.9,
                        "sensitive": True,
                    }
                ]
            }
        )


def test_proactive_sweep_does_not_run_without_global_grant(tmp_path):
    store = SQLiteAutonomyAdapter(str(tmp_path / "autonomy.db"))
    service = ProactiveSweepService(
        autonomy_store=store,
        llm_service=_LlmService(),
        capability_policy=CapabilityPolicyService(store=store),
        model="model",
        enabled=True,
        global_grant=False,
    )

    assert service.is_due() is False
    assert asyncio.run(service.run_if_due())["ran"] is False


def test_proactive_sweep_filters_read_tools_and_surfaces_finding(tmp_path):
    store = SQLiteAutonomyAdapter(str(tmp_path / "autonomy.db"))
    llm_service = _LlmService()
    service = ProactiveSweepService(
        autonomy_store=store,
        llm_service=llm_service,
        capability_policy=CapabilityPolicyService(store=store),
        model="model",
        enabled=True,
        global_grant=True,
        enabled_sources=["gmail"],
        minimum_importance="medium",
    )

    result = asyncio.run(service.run_if_due())

    assert result["ran"] is True
    assert result["created_findings"] == 1
    assert "gmail_search" in llm_service.allowed_tool_names
    assert "gmail_send" not in llm_service.allowed_tool_names
    inbox = store.list_inbox_items()
    assert len(inbox) == 1
    assert inbox[0].title == "Important email needs review"


def test_proactive_sweep_prompt_allows_multi_turn_tool_exploration():
    prompt = ProactiveSweepService.SYSTEM_PROMPT

    assert "multiple tools across multiple" in prompt
    assert "Only after you are done exploring" in prompt
    assert "Do not finalize on the first" in prompt and "turn unless" in prompt
    assert "Return JSON only:" not in prompt


def test_workspace_tool_names_classify_common_read_variants():
    registry = CapabilityRegistry()

    assert registry.describe("gmail_search").capability == "communication.read"
    assert registry.describe("gmail_send").capability == "communication.send"
    assert registry.describe("calendar_list_events").capability == "assistance.calendar.read"
    assert registry.describe("calendar_create_event").capability == "communication.calendar.write"


def test_computer_read_only_session_hides_typing_tools():
    session = ComputerControlSession(read_only=True)
    try:
        names = {tool["function"]["name"] for tool in asyncio.run(session.get_all_tools())}
    finally:
        asyncio.run(session.cleanup())

    assert "computer_inspect" in names
    assert "computer_click" in names
    assert "computer_type_text" not in names
    assert "computer_press_key" not in names
