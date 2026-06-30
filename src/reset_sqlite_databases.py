import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from config import CONFIG
import night_mode
from infrastructure.adapter.SQLiteBenchmarkAdapter import SQLiteBenchmarkAdapter
from infrastructure.adapter.SQLiteInteractionLogAdapter import SQLiteInteractionLogAdapter
from infrastructure.adapter.SQLiteMemoryAdapter import SQLiteMemoryAdapter
from infrastructure.adapter.SQLiteTrainingDataAdapter import SQLiteTrainingDataAdapter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
USER_DATA_DIR = Path(CONFIG.get_str("runtime", "user_data_dir", "D:\\USER_DATA"))
MEMORY_ROOT = USER_DATA_DIR / "memory"
MEMORY_DB_PATH = USER_DATA_DIR / "database" / "memory.db"
INTERACTION_LOG_DB_PATH = USER_DATA_DIR / "database" / "interaction_logs.db"
BENCHMARK_DB_PATH = Path(
    CONFIG.get_str("benchmarking", "db_path", str(PROJECT_ROOT / "database" / "benchmarking.db"))
)
TRAINING_DATA_ROOT = Path(CONFIG.get_str("training_data", "root", "D:\\TRAINING_DATA"))
TRAINING_DATA_DB_PATH = Path(
    CONFIG.get_str("training_data", "db_path", str(TRAINING_DATA_ROOT / "database" / "training_data.db"))
)
VOICE_DB_PATH = USER_DATA_DIR / "database" / "voice_database.db"
NIGHT_QUEUE_DB_PATH = Path(night_mode.DB_FILE)
FACTS_DB_PATH = PROJECT_ROOT / "database" / "facts.db"
FINANCE_DB_PATH = PROJECT_ROOT / "database" / "finance.db"


@dataclass(frozen=True)
class DatabaseSpec:
    name: str
    path: Path
    clear_statements: tuple[str, ...]
    init_func: Callable[[], None]
    description: str


def _init_memory_db() -> None:
    SQLiteMemoryAdapter(db_path=str(MEMORY_DB_PATH), memory_root=str(MEMORY_ROOT))


def _init_interaction_logs_db() -> None:
    SQLiteInteractionLogAdapter(db_path=str(INTERACTION_LOG_DB_PATH))


def _init_benchmark_db() -> None:
    SQLiteBenchmarkAdapter(db_path=str(BENCHMARK_DB_PATH))


def _init_training_data_db() -> None:
    SQLiteTrainingDataAdapter(db_path=str(TRAINING_DATA_DB_PATH))


def _init_voice_db() -> None:
    VOICE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(VOICE_DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS VOICE_EMBEDDINGS (
                uid INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                embedding BLOB NOT NULL,
                embedding_hash TEXT NOT NULL UNIQUE,
                audio BLOB
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _init_night_queue_db() -> None:
    night_mode.init_db()


def _init_facts_db() -> None:
    FACTS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(FACTS_DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
                fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                fact TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _init_finance_db() -> None:
    FINANCE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(FINANCE_DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY,
                amount REAL NOT NULL,
                category TEXT NOT NULL,
                category_id INTEGER,
                account_id INTEGER,
                date TEXT NOT NULL,
                description TEXT,
                type TEXT NOT NULL CHECK(type IN ('income', 'expense'))
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS categories (
                category_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                type TEXT NOT NULL CHECK(type IN ('Income', 'Expense'))
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS accounts (
                account_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                type TEXT NOT NULL,
                initial_balance REAL NOT NULL DEFAULT 0
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def build_database_specs() -> dict[str, DatabaseSpec]:
    return {
        "memory": DatabaseSpec(
            name="memory",
            path=MEMORY_DB_PATH,
            clear_statements=(
                "DELETE FROM semantic_memory_embeddings",
                "DELETE FROM semantic_memory_chunks",
                "DELETE FROM semantic_memory_config",
                "DELETE FROM semantic_dedupe_items",
                "DELETE FROM fused_context_episodes",
                "DELETE FROM visual_user_facts",
                "DELETE FROM visual_sessions",
                "DELETE FROM visual_observations",
                "DELETE FROM user_profile_facets",
                "DELETE FROM open_loops",
                "DELETE FROM conversation_sessions",
                "DELETE FROM transcript_evidence",
                "DELETE FROM memory_reflections",
                "DELETE FROM memory_facts",
                "DELETE FROM memory_events",
                "DELETE FROM speakers",
            ),
            init_func=_init_memory_db,
            description="Ambient memory, semantic memory, passive-observer state, and dedupe registry.",
        ),
        "interaction_logs": DatabaseSpec(
            name="interaction_logs",
            path=INTERACTION_LOG_DB_PATH,
            clear_statements=("DELETE FROM interaction_logs",),
            init_func=_init_interaction_logs_db,
            description="Persisted LLM interaction and report logs.",
        ),
        "benchmark": DatabaseSpec(
            name="benchmark",
            path=BENCHMARK_DB_PATH,
            clear_statements=(
                "DELETE FROM benchmark_manual_reviews",
                "DELETE FROM benchmark_results",
                "DELETE FROM benchmark_runs",
            ),
            init_func=_init_benchmark_db,
            description="Benchmark runs, results, and manual reviews.",
        ),
        "training_data": DatabaseSpec(
            name="training_data",
            path=TRAINING_DATA_DB_PATH,
            clear_statements=(
                "DELETE FROM training_llm_reviews",
                "DELETE FROM training_llm_records",
                "DELETE FROM training_asr_reviews",
                "DELETE FROM training_asr_records",
                "DELETE FROM training_dataset_exports",
            ),
            init_func=_init_training_data_db,
            description="Training-review records and export history.",
        ),
        "voice": DatabaseSpec(
            name="voice",
            path=VOICE_DB_PATH,
            clear_statements=("DELETE FROM VOICE_EMBEDDINGS",),
            init_func=_init_voice_db,
            description="Stored speaker embeddings.",
        ),
        "night_queue": DatabaseSpec(
            name="night_queue",
            path=NIGHT_QUEUE_DB_PATH,
            clear_statements=(
                "DELETE FROM night_queue",
                "DELETE FROM system_notifications",
            ),
            init_func=_init_night_queue_db,
            description="Queued background tasks and system notifications.",
        ),
        "facts": DatabaseSpec(
            name="facts",
            path=FACTS_DB_PATH,
            clear_statements=("DELETE FROM facts",),
            init_func=_init_facts_db,
            description="Project-local facts database used by MCP facts tools.",
        ),
        "finance": DatabaseSpec(
            name="finance",
            path=FINANCE_DB_PATH,
            clear_statements=(
                "DELETE FROM transactions",
                "DELETE FROM categories",
                "DELETE FROM accounts",
            ),
            init_func=_init_finance_db,
            description="Project-local finance transactions, categories, and accounts.",
        ),
    }


def _remove_sqlite_sidecars(db_path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(db_path) + suffix)
        if candidate.exists():
            candidate.unlink()


def _execute_clear_statements(db_path: Path, statements: Iterable[str]) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        for statement in statements:
            try:
                conn.execute(statement)
            except sqlite3.OperationalError as exc:
                if "no such table" in str(exc).lower():
                    continue
                raise
        try:
            conn.execute("DELETE FROM sqlite_sequence")
        except sqlite3.OperationalError as exc:
            if "no such table" not in str(exc).lower():
                raise
        conn.commit()
        conn.execute("VACUUM")
    finally:
        conn.close()


def soft_reset_database(spec: DatabaseSpec) -> None:
    spec.init_func()
    _execute_clear_statements(spec.path, spec.clear_statements)


def hard_reset_database(spec: DatabaseSpec) -> None:
    if spec.path.exists():
        _remove_sqlite_sidecars(spec.path)
    else:
        spec.path.parent.mkdir(parents=True, exist_ok=True)
    spec.init_func()


def parse_args() -> argparse.Namespace:
    specs = build_database_specs()
    parser = argparse.ArgumentParser(
        description="Reset one or more Ambient AI SQLite databases.",
    )
    parser.add_argument(
        "targets",
        nargs="*",
        choices=sorted([*specs.keys(), "all"]),
        default=["all"],
        help="Database targets to reset. Defaults to all.",
    )
    parser.add_argument(
        "--hard-reset",
        action="store_true",
        help="Delete and recreate each selected database file instead of just clearing table contents.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available database targets and exit.",
    )
    return parser.parse_args()


def _resolve_targets(requested: list[str], specs: dict[str, DatabaseSpec]) -> list[DatabaseSpec]:
    normalized = requested or ["all"]
    if "all" in normalized:
        return list(specs.values())
    return [specs[name] for name in normalized]


def main() -> int:
    args = parse_args()
    specs = build_database_specs()

    if args.list:
        print("Available database targets:")
        for name, spec in specs.items():
            print(f"- {name}: {spec.description} [{spec.path}]")
        return 0

    targets = _resolve_targets(args.targets, specs)
    action = "hard-reset" if args.hard_reset else "reset"

    for spec in targets:
        if args.hard_reset:
            hard_reset_database(spec)
        else:
            soft_reset_database(spec)
        print(f"{action}: {spec.name} -> {spec.path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
