import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from application.services.interaction_trace import interaction_trace
from infrastructure.adapter.LoggingLLMProvider import LoggingLLMProvider
from infrastructure.adapter.SQLiteInteractionLogAdapter import SQLiteInteractionLogAdapter


class _FakeDelta:
    def __init__(self, content=None, reasoning_content=None, tool_calls=None):
        self.content = content
        self.reasoning_content = reasoning_content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, delta):
        self.delta = delta


class _FakeChunk:
    def __init__(self, content=None, reasoning_content=None, tool_calls=None):
        self.choices = [_FakeChoice(_FakeDelta(content=content, reasoning_content=reasoning_content, tool_calls=tool_calls))]


class FakeLLMProvider:
    def generate_response(self, prompt: str, image: str = "") -> str:
        return "generated-response"

    async def chat_completion_stream(self, model, messages, tools=None, image="", temperature=0.7, top_p=0.95, top_k=0):
        async def _gen():
            yield _FakeChunk(content="hello ")
            yield _FakeChunk(content="world")
        return _gen()

    async def load_model(self, model_name: str):
        return None

    async def save_and_unload(self, messages):
        return None

    async def load_and_restore(self):
        return None


class InteractionLoggingTests(unittest.TestCase):
    def test_streamed_interaction_is_persisted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "interaction_logs.db"
            store = SQLiteInteractionLogAdapter(str(db_path))
            provider = LoggingLLMProvider(FakeLLMProvider(), store)

            async def _run():
                with interaction_trace("unit_test_source", {"kind": "test"}):
                    stream = await provider.chat_completion_stream(
                        model="test-model",
                        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
                        tools=None,
                    )
                    text = []
                    async for chunk in stream:
                        text.append(chunk.choices[0].delta.content or "")
                    return "".join(text)

            result = asyncio.run(_run())
            self.assertEqual(result, "hello world")

            rows = store.list_recent(limit=5)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].source, "unit_test_source")
            self.assertEqual(rows[0].model, "test-model")
            self.assertIn('"role": "user"', rows[0].messages_json)
            self.assertEqual(rows[0].response_text, "hello world")
            self.assertEqual(json.loads(rows[0].metadata_json)["kind"], "test")
            self.assertIsNotNone(rows[0].interaction_run_id)

    def test_attach_report_persists_and_updates_markdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "interaction_logs.db"
            response_path = Path(tmpdir) / "current_llm_response.md"
            store = SQLiteInteractionLogAdapter(str(db_path))
            provider = LoggingLLMProvider(
                FakeLLMProvider(),
                store,
                current_response_path=str(response_path),
            )

            async def _run():
                with interaction_trace("unit_test_source", {"kind": "test"}):
                    stream = await provider.chat_completion_stream(
                        model="test-model",
                        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
                        tools=None,
                    )
                    async for _ in stream:
                        pass
                    rows = store.list_recent(limit=1)
                    run_id = rows[0].interaction_run_id
                    provider.attach_report(
                        run_id,
                        {
                            "title": "Latest Search Result",
                            "summary": "Searched and found the latest result.",
                            "artifact_path": str(Path(tmpdir) / "artifacts" / "Latest_Search_Result.md"),
                            "tools_used": ["web_search"],
                            "status": "completed",
                        },
                    )

            asyncio.run(_run())

            reports = store.list_recent_reports(limit=5)
            self.assertEqual(len(reports), 1)
            report_payload = json.loads(reports[0].report_json)
            self.assertEqual(report_payload["title"], "Latest Search Result")
            self.assertIn("Searched and found", report_payload["summary"])

            markdown = response_path.read_text(encoding="utf-8")
            self.assertIn("## Report Title", markdown)
            self.assertIn("Latest Search Result", markdown)
            self.assertIn("## Report Summary", markdown)
            self.assertIn("Searched and found the latest result.", markdown)


if __name__ == "__main__":
    unittest.main()
