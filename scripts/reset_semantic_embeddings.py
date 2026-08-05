"""Reset Ambient semantic embedding vectors after changing embedding models.

This preserves source memory:
- USER_INFO.md and MEMORY.md files are untouched.
- semantic_memory_chunks are untouched.
- artifacts/observations/other memory tables are untouched.

Only the generated vector index and stored embedding dimension are removed.
The next SemanticMemoryService sync will rebuild vectors using the configured
embedding model.
"""

from __future__ import annotations

import argparse
import configparser
import sqlite3
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.ini"


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    try:
        import sqlite_vec  # type: ignore
    except Exception:
        return
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
    except Exception:
        return
    finally:
        try:
            conn.enable_load_extension(False)
        except Exception:
            pass


def _resolve_memory_db(config_path: Path) -> Path:
    parser = configparser.ConfigParser()
    parser.read(config_path, encoding="utf-8")
    user_data_dir = Path(parser.get("runtime", "user_data_dir", fallback=str(Path.home() / "AmbientAI" / "data")))
    return user_data_dir / "database" / "memory.db"


def _table_count(conn: sqlite3.Connection, table_name: str) -> int | None:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual table') AND name = ?",
        (table_name,),
    ).fetchone()
    if row is None:
        return None
    try:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])
    except sqlite3.Error:
        return None


def reset_semantic_embeddings(db_path: Path, *, dry_run: bool = False) -> dict[str, int | str | bool | None]:
    if not db_path.exists():
        raise FileNotFoundError(f"Memory database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        _load_sqlite_vec(conn)
        before_embeddings = _table_count(conn, "semantic_memory_embeddings")
        before_chunks = _table_count(conn, "semantic_memory_chunks")
        before_dimension = conn.execute(
            "SELECT config_value FROM semantic_memory_config WHERE config_key = 'embedding_dimension'"
        ).fetchone()

        if not dry_run:
            conn.execute("DROP TABLE IF EXISTS semantic_memory_embeddings")
            conn.execute("DELETE FROM semantic_memory_config WHERE config_key = 'embedding_dimension'")
            conn.commit()

        after_embeddings = _table_count(conn, "semantic_memory_embeddings")
        after_chunks = _table_count(conn, "semantic_memory_chunks")
        after_dimension = conn.execute(
            "SELECT config_value FROM semantic_memory_config WHERE config_key = 'embedding_dimension'"
        ).fetchone()

        return {
            "dry_run": dry_run,
            "db_path": str(db_path),
            "semantic_embeddings_before": before_embeddings,
            "semantic_embeddings_after": after_embeddings,
            "semantic_chunks_before": before_chunks,
            "semantic_chunks_after": after_chunks,
            "embedding_dimension_before": before_dimension[0] if before_dimension else None,
            "embedding_dimension_after": after_dimension[0] if after_dimension else None,
        }
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Drop only Ambient semantic embeddings and embedding dimension metadata."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Optional explicit memory.db path. Defaults to [runtime] user_data_dir/database/memory.db.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Inspect counts without modifying the database.")
    args = parser.parse_args()

    db_path = args.db or _resolve_memory_db(CONFIG_PATH)
    result = reset_semantic_embeddings(db_path, dry_run=args.dry_run)

    print("Semantic embedding reset result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    if not args.dry_run:
        print("Preserved semantic_memory_chunks; only generated vectors/dimension metadata were reset.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
