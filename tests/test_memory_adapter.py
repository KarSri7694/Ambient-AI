import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from infrastructure.adapter.SQLiteMemoryAdapter import SQLiteMemoryAdapter


def test_append_working_memory_skips_exact_duplicate(tmp_path):
    memory = SQLiteMemoryAdapter(
        db_path=str(tmp_path / "memory.db"),
        memory_root=str(tmp_path / "memory"),
    )

    assert memory.append_working_memory("- **preference:** prefers local inference") is True
    assert memory.append_working_memory("- **preference:** prefers local inference") is False
    assert memory.get_working_memory() == "- **preference:** prefers local inference\n"
