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

from benchmarking.case_loader import build_inline_case, load_cases, load_suite
from benchmarking.provider import BenchmarkingLLMProvider
from benchmarking.services import BenchmarkExecution, score_case
from core.models import BenchmarkResult, BenchmarkRun
from infrastructure.adapter.SQLiteBenchmarkAdapter import SQLiteBenchmarkAdapter


class _FakeDelta:
    def __init__(self, content=None):
        self.content = content
        self.reasoning_content = None
        self.tool_calls = None


class _FakeChoice:
    def __init__(self, delta):
        self.delta = delta


class _FakeChunk:
    def __init__(self, content=None, usage=None):
        self.choices = [_FakeChoice(_FakeDelta(content=content))]
        self.usage = usage


class _FakeUsage:
    def __init__(self, prompt_tokens=None, completion_tokens=None, total_tokens=None):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class FakeBenchmarkProvider:
    def count_message_tokens(self, messages, image=""):
        return 12

    def count_text_tokens(self, text):
        return 5

    def generate_response(self, prompt: str, image: str = "") -> str:
        return "done"

    async def chat_completion_stream(self, model, messages, tools=None, image="", temperature=0.7, top_p=0.95, top_k=0):
        async def _gen():
            yield _FakeChunk(content="hello ")
            yield _FakeChunk(content="world")
            yield _FakeChunk(usage=_FakeUsage(prompt_tokens=21, completion_tokens=7, total_tokens=28))

        return _gen()

    async def load_model(self, model_name: str):
        return None

    async def save_and_unload(self, messages):
        return None

    async def load_and_restore(self):
        return None


class BenchmarkingTests(unittest.TestCase):
    def test_sqlite_benchmark_adapter_persists_results_and_reviews(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "benchmarking.db"
            store = SQLiteBenchmarkAdapter(str(db_path))
            run = BenchmarkRun(
                run_id="run-1",
                created_at="2026-06-28T00:00:00",
                completed_at=None,
                service_name="reflection_service",
                model_names_json='["model-a"]',
                case_ids_json='["case-1"]',
                status="running",
                notes=None,
            )
            store.insert_run(run)
            result = BenchmarkResult(
                result_id="result-1",
                run_id="run-1",
                created_at="2026-06-28T00:00:01",
                completed_at="2026-06-28T00:00:02",
                service_name="reflection_service",
                case_id="case-1",
                case_title="Case 1",
                model_name="model-a",
                response_text="output",
                auto_score=0.75,
                status="completed",
            )
            store.insert_result(result)
            review = store.upsert_manual_review(
                result_id="result-1",
                reviewer="tester",
                score=0.9,
                notes="Looks good.",
                created_at="2026-06-28T00:00:03",
                updated_at="2026-06-28T00:00:03",
            )

            fetched_results = store.list_results(run_id="run-1")
            self.assertEqual(len(fetched_results), 1)
            self.assertEqual(fetched_results[0].auto_score, 0.75)
            self.assertEqual(store.get_manual_review("result-1").notes, "Looks good.")
            self.assertEqual(review.reviewer, "tester")

    def test_benchmarking_provider_collects_stream_metrics(self):
        provider = BenchmarkingLLMProvider(FakeBenchmarkProvider())

        async def _run():
            stream = await provider.chat_completion_stream(
                model="model-a",
                messages=[{"role": "user", "content": "hello"}],
            )
            output = []
            async for chunk in stream:
                output.append(chunk.choices[0].delta.content or "")
            return "".join(output)

        result = asyncio.run(_run())
        metrics = provider.benchmark_metrics()

        self.assertEqual(result, "hello world")
        self.assertEqual(metrics.prompt_tokens, 21)
        self.assertEqual(metrics.completion_tokens, 7)
        self.assertEqual(metrics.total_tokens, 28)
        self.assertEqual(metrics.call_count, 1)
        self.assertEqual(metrics.token_count_method, "server_usage")

    def test_case_loader_and_hybrid_scoring_work_for_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            transcript = root / "sample.txt"
            transcript.write_text("hello", encoding="utf-8")
            case_path = root / "case.json"
            case_path.write_text(
                json.dumps(
                    {
                        "case_id": "case-1",
                        "service": "reflection_service",
                        "title": "Sample",
                        "inputs": {"transcript_path": "sample.txt"},
                        "expected": {"contains": ["completed"], "json_fields": ["ran"]},
                    }
                ),
                encoding="utf-8",
            )

            cases = load_cases(str(root))
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].inputs["transcript_path"], str(transcript.resolve()))

            score, details = score_case(
                cases[0],
                BenchmarkExecution(
                    response_text='{"ran": true, "status": "completed"}',
                    structured_output_json='{"ran": true, "status": "completed"}',
                ),
            )
            self.assertEqual(score, 1.0)
            self.assertEqual(details["passed_checks"], 2)

    def test_case_loader_accepts_minimal_passive_observer_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            screenshot = root / "screen.png"
            screenshot.write_bytes(b"png")
            case_path = root / "observer_case.json"
            case_path.write_text(
                json.dumps(
                    {
                        "case_id": "observer-1",
                        "service": "passive_observer",
                        "screenshot_path": "screen.png",
                    }
                ),
                encoding="utf-8",
            )

            cases = load_cases(str(root), service="passive_observer")

            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].inputs["screenshot_path"], str(screenshot.resolve()))
            self.assertIn("screen.png", cases[0].title)

    def test_inline_case_builder_supports_direct_passive_observer_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            screenshot = Path(tmpdir) / "capture.png"
            screenshot.write_bytes(b"png")

            case = build_inline_case(
                service="passive_observer",
                screenshot_path=str(screenshot),
            )

            self.assertEqual(case.service, "passive_observer")
            self.assertEqual(case.case_id, "passive_observer_inline")
            self.assertEqual(case.inputs["screenshot_path"], str(screenshot.resolve()))

    def test_suite_loader_expands_models_and_screenshots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shots = root / "shots"
            shots.mkdir()
            first = shots / "a.png"
            second = shots / "b.png"
            first.write_bytes(b"png")
            second.write_bytes(b"png")
            suite_path = root / "suite.json"
            suite_path.write_text(
                json.dumps(
                    {
                        "service": "passive_observer",
                        "models": ["model-a", "model-b"],
                        "screenshots": ["shots/a.png", "shots/b.png"],
                    }
                ),
                encoding="utf-8",
            )

            suite = load_suite(str(suite_path))

            self.assertEqual(suite.service, "passive_observer")
            self.assertEqual(suite.models, ["model-a", "model-b"])
            self.assertEqual(len(suite.cases), 2)
            self.assertEqual(suite.cases[0].inputs["screenshot_path"], str(first.resolve()))
            self.assertEqual(suite.cases[1].inputs["screenshot_path"], str(second.resolve()))


if __name__ == "__main__":
    unittest.main()
