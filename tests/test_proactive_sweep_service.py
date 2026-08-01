import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from application.services.capability_policy_service import CapabilityPolicyService, CapabilityRegistry
from application.services.proactive_sweep_service import ProactiveSweepService
from core.models import OpportunityCandidate, ProactiveInboxItem
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


class _Memory:
    def __init__(self):
        self.observations = []

    def append_visual_observation(self, observation):
        self.observations.append(observation)


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
    memory = _Memory()
    service = ProactiveSweepService(
        autonomy_store=store,
        llm_service=llm_service,
        capability_policy=CapabilityPolicyService(store=store),
        memory=memory,
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
    assert len(memory.observations) == 1
    assert memory.observations[0].observation_type == "proactive_finding"
    assert memory.observations[0].biodata_sent_at is None
    assert memory.observations[0].followup_sent_at
    assert "Important email needs review" in memory.observations[0].summary


def test_proactive_sweep_prompt_allows_multi_turn_tool_exploration():
    prompt = ProactiveSweepService.SYSTEM_PROMPT

    assert "multiple tools across multiple" in prompt
    assert "Only after you are done exploring" in prompt
    assert "Do not finalize on the first" in prompt and "turn unless" in prompt
    assert "Return JSON only:" not in prompt


def test_proactive_gmail_task_explains_search_then_batch_workflow():
    task = ProactiveSweepService.SOURCE_TASKS["gmail"]

    assert "first call search_gmail_messages" in task
    assert "get_gmail_messages_content_batch" in task
    assert "Never call get_gmail_messages_content_batch with an empty" in task
    assert "never use 'me' as a message_id" in task


def test_proactive_inbox_lists_newest_created_items_first(tmp_path):
    store = SQLiteAutonomyAdapter(str(tmp_path / "autonomy.db"))
    for opportunity_id in ("opp-old", "opp-new"):
        store.upsert_opportunity(
            OpportunityCandidate(
                opportunity_id=opportunity_id,
                fingerprint=opportunity_id,
                title=opportunity_id,
                goal="test",
                rationale="test",
                source_event_ids=[],
                expected_value=0.5,
                urgency=0.5,
                confidence=0.8,
                cost_of_wrong=0.1,
                personalization_benefit=0.5,
                evidence_gaps=[],
                status="surfaced",
                created_at="2026-08-01T07:00:00+00:00",
                updated_at="2026-08-01T07:00:00+00:00",
            )
        )
    old_item = ProactiveInboxItem(
        inbox_id="old", opportunity_id="opp-old", title="Old", summary="old",
        detailed_report="", status="ready", confidence=0.8, why_now="old",
        sources_json="[]", personalization_json="{}", actions_json="[]",
        created_at="2026-08-01T08:00:00+00:00",
        updated_at="2026-08-01T12:00:00+00:00",
    )
    new_item = ProactiveInboxItem(
        inbox_id="new", opportunity_id="opp-new", title="New", summary="new",
        detailed_report="", status="ready", confidence=0.8, why_now="new",
        sources_json="[]", personalization_json="{}", actions_json="[]",
        created_at="2026-08-01T11:00:00+00:00",
        updated_at="2026-08-01T11:00:00+00:00",
    )
    store.add_inbox_item(old_item)
    store.add_inbox_item(new_item)

    assert [item.inbox_id for item in store.list_inbox_items()] == ["new", "old"]


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
