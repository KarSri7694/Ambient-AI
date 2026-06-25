import sqlite3
from pathlib import Path

import night_mode
from infrastructure.adapter.SQLiteAmbientAgendaAdapter import SQLiteAmbientAgendaAdapter
from infrastructure.adapter.SQLiteActivityLedgerAdapter import SQLiteActivityLedgerAdapter
from infrastructure.adapter.SQLiteInteractionLogAdapter import SQLiteInteractionLogAdapter
from infrastructure.adapter.SQLiteMemoryAdapter import SQLiteMemoryAdapter
from infrastructure.adapter.SQLiteProactiveTopicQueueAdapter import SQLiteProactiveTopicQueueAdapter


def ensure_runtime_databases(
    *,
    memory_db_path: str,
    memory_root: str,
    ambient_agenda_db_path: str,
    interaction_log_db_path: str,
    proactive_topics_db_path: str,
    activity_ledger_db_path: str,
    voice_db_path: str,
    finance_db_path: str,
    facts_db_path: str,
) -> None:
    Path(memory_db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(ambient_agenda_db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(interaction_log_db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(proactive_topics_db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(activity_ledger_db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(voice_db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(finance_db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(facts_db_path).parent.mkdir(parents=True, exist_ok=True)

    SQLiteMemoryAdapter(db_path=memory_db_path, memory_root=memory_root)
    SQLiteAmbientAgendaAdapter(db_path=ambient_agenda_db_path)
    SQLiteInteractionLogAdapter(db_path=interaction_log_db_path)
    SQLiteActivityLedgerAdapter(
        db_path=activity_ledger_db_path,
        interaction_log_db_path=interaction_log_db_path,
    )
    SQLiteProactiveTopicQueueAdapter(db_path=proactive_topics_db_path)
    _ensure_voice_db(voice_db_path)
    night_mode.init_db()
    _ensure_finance_db(finance_db_path)
    _ensure_facts_db(facts_db_path)


def _ensure_voice_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS VOICE_EMBEDDINGS (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_name TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _ensure_finance_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                amount REAL NOT NULL,
                category TEXT,
                description TEXT,
                type TEXT NOT NULL,
                account TEXT,
                created_at TEXT,
                transaction_date TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                type TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS accounts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                account_type TEXT,
                balance REAL DEFAULT 0
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _ensure_facts_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact TEXT NOT NULL,
                source TEXT,
                created_at TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()
