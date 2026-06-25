import sys
import tempfile
import unittest
import uuid
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from application.services.memory_context_builder import MemoryContextBuilder
from application.services.semantic_memory_service import SemanticMemoryService
from core.models import MemoryEvent
from infrastructure.adapter.LlamaCppSemanticAdapter import _TemporaryLlamaModel
from infrastructure.adapter.SQLiteMemoryAdapter import SQLiteMemoryAdapter


class FakeEmbedder:
    async def embed_texts(self, texts):
        embeddings = []
        for text in texts:
            lowered = text.lower()
            embeddings.append([
                1.0 if "coffee" in lowered else 0.0,
                1.0 if "travel" in lowered else 0.0,
                1.0 if "mom" in lowered else 0.0,
            ])
        return embeddings


class FakeReranker:
    async def rerank(self, query, documents):
        lowered_query = query.lower()
        scores = []
        for document in documents:
            lowered_document = document.lower()
            score = 0.0
            for token in lowered_query.split():
                if token.strip(".,:;!?") in lowered_document:
                    score += 1.0
            scores.append(score)
        return scores


class FakeLlama:
    def __init__(self):
        self.current = "main-model"
        self.calls = []

    def get_current_model(self):
        self.calls.append("get_current_model")
        return self.current

    async def save_and_unload(self, messages):
        self.calls.append(("save_and_unload", messages))
        self.current = None
        return Path("main-model.kv")

    async def load_model(self, model_name):
        self.calls.append(("load_model", model_name))
        self.current = model_name

    async def unload_model(self):
        self.calls.append(("unload_model", self.current))
        self.current = None

    async def load_and_restore(self):
        self.calls.append("load_and_restore")
        self.current = "main-model"
        return Path("main-model.kv")


class SemanticMemoryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.prompts_dir = self.temp_path / "prompts"
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        (self.prompts_dir / "AGENT.md").write_text("Agent prompt", encoding="utf-8")
        (self.prompts_dir / "USER.md").write_text("User prompt", encoding="utf-8")
        self.memory = SQLiteMemoryAdapter(
            db_path=str(self.temp_path / "memory.db"),
            memory_root=str(self.temp_path / "memory"),
        )
        self.speaker = self.memory.upsert_speaker("Alice", "Alice")

    def tearDown(self):
        self.temp_dir.cleanup()

    async def test_memory_write_creates_semantic_chunk_and_retrieves_it(self):
        self.memory.append_event(
            MemoryEvent(
                event_id=uuid.uuid4().hex,
                speaker_id=self.speaker.speaker_id,
                source_type="transcript",
                source_ref="sample.txt",
                event_kind="preference",
                content="Alice likes black coffee before meetings.",
                confidence=0.8,
                status="candidate",
                created_at=datetime.now().isoformat(),
            )
        )

        pending = self.memory.get_chunks_missing_embeddings(limit=10)
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].source_type, "memory_event")

        service = SemanticMemoryService(
            memory=self.memory,
            embedder=FakeEmbedder(),
            reranker=FakeReranker(),
        )
        results = await service.retrieve("coffee preference", speaker_ids=[self.speaker.speaker_id])

        self.assertEqual(len(results), 1)
        self.assertIn("black coffee", results[0].chunk.content)
        self.assertIsNotNone(results[0].rerank_score)

    async def test_context_builder_includes_semantic_section(self):
        self.memory.upsert_semantic_chunk(
            source_type="memory_fact",
            source_id="fact-1",
            source_ref="manual",
            speaker_id=self.speaker.speaker_id,
            content="Alice prefers async status updates.",
        )
        service = SemanticMemoryService(
            memory=self.memory,
            embedder=FakeEmbedder(),
            reranker=None,
        )
        results = await service.retrieve("status updates", speaker_ids=[self.speaker.speaker_id])
        section = service.format_prompt_section(results)

        builder = MemoryContextBuilder(self.memory, str(self.prompts_dir))
        prompt = builder.build_prompt(
            base_prompt_filename="AGENT.md",
            skills_summary="Skill summary",
            participants=[],
            semantic_memory_section=section,
        )

        self.assertIn("Relevant Semantic Memory", prompt)
        self.assertIn("async status updates", prompt)

    async def test_temporary_model_preserves_and_restores_main_model(self):
        fake_llm = FakeLlama()
        temporary = _TemporaryLlamaModel(
            llm=fake_llm,
            model_path="embedding-model.gguf",
            messages_provider=lambda: [{"role": "user", "content": "current task"}],
        )

        async def operation():
            self.assertEqual(fake_llm.current, "embedding-model.gguf")
            return "embedded"

        result = await temporary.run(operation)

        self.assertEqual(result, "embedded")
        self.assertEqual(fake_llm.current, "main-model")
        self.assertIn(("save_and_unload", [{"role": "user", "content": "current task"}]), fake_llm.calls)
        self.assertIn(("load_model", "embedding-model.gguf"), fake_llm.calls)
        self.assertIn(("unload_model", "embedding-model.gguf"), fake_llm.calls)
        self.assertIn("load_and_restore", fake_llm.calls)


if __name__ == "__main__":
    unittest.main()
