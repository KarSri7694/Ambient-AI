import json
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from application.ports.autonomy_port import AutonomyStorePort
from core.models import (
    ActivityRun,
    AmbientEvent,
    ApprovalGrant,
    OpportunityCandidate,
    ProactiveInboxItem,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _utciso(value: datetime | None = None) -> str:
    return (value or _utcnow()).isoformat()


class SQLiteAutonomyAdapter(AutonomyStorePort):
    """SQLite-backed autonomy control plane with atomic leasing and deduplication."""

    SCHEMA_VERSION = 1

    def __init__(self, db_path: str):
        self.db_path = str(Path(db_path))
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS autonomy_schema (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ambient_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    source_kind TEXT NOT NULL,
                    source_ref TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    privacy_label TEXT NOT NULL,
                    fingerprint TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    priority REAL NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    available_at TEXT NOT NULL,
                    leased_at TEXT,
                    lease_expires_at TEXT,
                    processed_at TEXT,
                    error_text TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_ambient_events_claim
                    ON ambient_events(status, available_at, priority DESC, occurred_at);
                CREATE TABLE IF NOT EXISTS opportunities (
                    opportunity_id TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    source_event_ids_json TEXT NOT NULL,
                    expected_value REAL NOT NULL,
                    urgency REAL NOT NULL,
                    confidence REAL NOT NULL,
                    cost_of_wrong REAL NOT NULL,
                    personalization_benefit REAL NOT NULL,
                    evidence_gaps_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT,
                    metadata_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_opportunities_updated
                    ON opportunities(status, updated_at DESC);
                CREATE TABLE IF NOT EXISTS inbox_items (
                    inbox_id TEXT PRIMARY KEY,
                    opportunity_id TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    detailed_report TEXT NOT NULL,
                    status TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    why_now TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    personalization_json TEXT NOT NULL,
                    actions_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    feedback TEXT,
                    FOREIGN KEY(opportunity_id) REFERENCES opportunities(opportunity_id)
                );
                CREATE INDEX IF NOT EXISTS idx_inbox_items_updated
                    ON inbox_items(status, updated_at DESC);
                CREATE TABLE IF NOT EXISTS capability_policies (
                    capability TEXT PRIMARY KEY,
                    decision TEXT NOT NULL,
                    constraints_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS approvals (
                    approval_id TEXT PRIMARY KEY,
                    capability TEXT NOT NULL,
                    action_fingerprint TEXT NOT NULL,
                    constraints_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    approver TEXT NOT NULL,
                    reusable INTEGER NOT NULL DEFAULT 0,
                    used_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_approvals_lookup
                    ON approvals(capability, action_fingerprint, status, expires_at);
                CREATE TABLE IF NOT EXISTS action_idempotency (
                    idempotency_key TEXT PRIMARY KEY,
                    capability TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_text TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS activity_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT NOT NULL,
                    source_kind TEXT NOT NULL,
                    trigger_kind TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    output_text TEXT NOT NULL,
                    model TEXT NOT NULL,
                    session_id TEXT,
                    parent_run_id TEXT,
                    priority TEXT NOT NULL,
                    error_text TEXT,
                    metadata_json TEXT
                );
                CREATE TABLE IF NOT EXISTS activity_links (
                    link_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    metadata_json TEXT,
                    FOREIGN KEY(run_id) REFERENCES activity_runs(run_id)
                );
                CREATE TABLE IF NOT EXISTS autonomy_audit (
                    audit_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    action TEXT NOT NULL,
                    target TEXT NOT NULL,
                    details_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS calibration_outcomes (
                    outcome_id TEXT PRIMARY KEY,
                    capability TEXT NOT NULL,
                    correct INTEGER NOT NULL,
                    source_ref TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_calibration_capability
                    ON calibration_outcomes(capability, created_at);
                """
            )
            conn.execute(
                "INSERT OR IGNORE INTO autonomy_schema(version, applied_at) VALUES (?, ?)",
                (self.SCHEMA_VERSION, _utciso()),
            )

    def enqueue_event(self, event: AmbientEvent) -> AmbientEvent:
        available_at = event.available_at or event.occurred_at or _utciso()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO ambient_events (
                    event_id, event_type, source_kind, source_ref, occurred_at,
                    payload_json, confidence, privacy_label, fingerprint, status,
                    priority, attempt_count, available_at, leased_at,
                    lease_expires_at, processed_at, error_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id, event.event_type, event.source_kind, event.source_ref,
                    event.occurred_at, event.payload_json, event.confidence,
                    event.privacy_label, event.fingerprint, event.status, event.priority,
                    event.attempt_count, available_at, event.leased_at,
                    event.lease_expires_at, event.processed_at, event.error_text,
                ),
            )
            row = conn.execute(
                "SELECT * FROM ambient_events WHERE fingerprint=?", (event.fingerprint,)
            ).fetchone()
        return self._event_from_row(row)

    def claim_next_event(
        self,
        *,
        lease_seconds: int = 180,
        event_types: Optional[list[str]] = None,
    ) -> Optional[AmbientEvent]:
        now = _utcnow()
        now_iso = _utciso(now)
        lease_expires = _utciso(now + timedelta(seconds=max(1, lease_seconds)))
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                UPDATE ambient_events SET status='pending', leased_at=NULL, lease_expires_at=NULL
                WHERE status='leased' AND lease_expires_at < ?
                """,
                (now_iso,),
            )
            params: list[Any] = [now_iso]
            type_clause = "AND event_type != 'audio_capture_pending'"
            if event_types:
                normalized_types = [str(value) for value in event_types if str(value)]
                placeholders = ",".join("?" for _ in normalized_types)
                type_clause = f"AND event_type IN ({placeholders})"
                params.extend(normalized_types)
            row = conn.execute(
                f"""
                SELECT event_id FROM ambient_events
                WHERE status IN ('pending', 'resource_deferred') AND available_at <= ?
                {type_clause}
                ORDER BY priority DESC, occurred_at ASC LIMIT 1
                """,
                params,
            ).fetchone()
            if row is None:
                conn.commit()
                return None
            event_id = row["event_id"]
            conn.execute(
                """
                UPDATE ambient_events
                SET status='leased', leased_at=?, lease_expires_at=?, attempt_count=attempt_count+1
                WHERE event_id=? AND status IN ('pending', 'resource_deferred')
                """,
                (now_iso, lease_expires, event_id),
            )
            claimed = conn.execute("SELECT * FROM ambient_events WHERE event_id=?", (event_id,)).fetchone()
            conn.commit()
        return self._event_from_row(claimed) if claimed else None

    def complete_event(self, event_id: str, *, status: str = "processed", error_text: str | None = None) -> None:
        if status not in {"processed", "ignored", "dead_letter"}:
            raise ValueError("invalid terminal event status")
        with self._lock, self._connect() as conn:
            conn.execute(
                """UPDATE ambient_events SET status=?, processed_at=?, error_text=?,
                   leased_at=NULL, lease_expires_at=NULL WHERE event_id=?""",
                (status, _utciso(), error_text, event_id),
            )

    def retry_event(self, event_id: str, *, error_text: str, delay_seconds: int = 30, max_attempts: int = 3) -> None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT attempt_count FROM ambient_events WHERE event_id=?", (event_id,)).fetchone()
            if row is None:
                return
            if int(row["attempt_count"]) >= max_attempts:
                conn.execute(
                    """UPDATE ambient_events SET status='dead_letter', processed_at=?, error_text=?,
                       leased_at=NULL, lease_expires_at=NULL WHERE event_id=?""",
                    (_utciso(), error_text, event_id),
                )
            else:
                available = _utciso(_utcnow() + timedelta(seconds=max(0, delay_seconds)))
                conn.execute(
                    """UPDATE ambient_events SET status='pending', available_at=?, error_text=?,
                       leased_at=NULL, lease_expires_at=NULL WHERE event_id=?""",
                    (available, error_text, event_id),
                )

    def defer_event(self, event_id: str, *, reason: str, delay_seconds: int = 30) -> None:
        available = _utciso(_utcnow() + timedelta(seconds=max(1, delay_seconds)))
        with self._lock, self._connect() as conn:
            conn.execute(
                """UPDATE ambient_events
                   SET status='resource_deferred', available_at=?, error_text=?,
                       attempt_count=MAX(0, attempt_count-1), leased_at=NULL,
                       lease_expires_at=NULL
                   WHERE event_id=? AND status='leased'""",
                (available, reason, event_id),
            )

    def event_counts(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS count FROM ambient_events GROUP BY status"
            ).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}

    def has_ready_events(self) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT 1 FROM ambient_events
                   WHERE status IN ('pending', 'resource_deferred')
                     AND available_at<=? AND event_type!='audio_capture_pending'
                   LIMIT 1""",
                (_utciso(),),
            ).fetchone()
        return row is not None

    def upsert_opportunity(self, candidate: OpportunityCandidate) -> OpportunityCandidate:
        now = _utciso()
        created_at = candidate.created_at or now
        updated_at = candidate.updated_at or now
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT opportunity_id, source_event_ids_json, created_at FROM opportunities WHERE fingerprint=?",
                (candidate.fingerprint,),
            ).fetchone()
            source_event_ids = list(candidate.source_event_ids)
            opportunity_id = candidate.opportunity_id
            if existing is not None:
                opportunity_id = existing["opportunity_id"]
                created_at = existing["created_at"]
                source_event_ids = list(
                    dict.fromkeys(
                        [*json.loads(existing["source_event_ids_json"] or "[]"), *source_event_ids]
                    )
                )
            conn.execute(
                """
                INSERT INTO opportunities (
                    opportunity_id, fingerprint, title, goal, rationale,
                    source_event_ids_json, expected_value, urgency, confidence,
                    cost_of_wrong, personalization_benefit, evidence_gaps_json,
                    status, created_at, updated_at, expires_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(fingerprint) DO UPDATE SET
                    title=excluded.title, goal=excluded.goal, rationale=excluded.rationale,
                    source_event_ids_json=excluded.source_event_ids_json,
                    expected_value=MAX(opportunities.expected_value, excluded.expected_value),
                    urgency=MAX(opportunities.urgency, excluded.urgency),
                    confidence=MAX(opportunities.confidence, excluded.confidence),
                    cost_of_wrong=MIN(opportunities.cost_of_wrong, excluded.cost_of_wrong),
                    personalization_benefit=MAX(opportunities.personalization_benefit, excluded.personalization_benefit),
                    evidence_gaps_json=excluded.evidence_gaps_json,
                    status=excluded.status, updated_at=excluded.updated_at,
                    expires_at=excluded.expires_at, metadata_json=excluded.metadata_json
                """,
                (
                    opportunity_id, candidate.fingerprint, candidate.title,
                    candidate.goal, candidate.rationale, json.dumps(source_event_ids),
                    candidate.expected_value, candidate.urgency, candidate.confidence,
                    candidate.cost_of_wrong, candidate.personalization_benefit,
                    json.dumps(candidate.evidence_gaps), candidate.status, created_at,
                    updated_at, candidate.expires_at, candidate.metadata_json,
                ),
            )
            row = conn.execute(
                "SELECT * FROM opportunities WHERE fingerprint=?", (candidate.fingerprint,)
            ).fetchone()
        return self._opportunity_from_row(row)

    def list_opportunities(self, *, limit: int = 50, status: str | None = None) -> list[OpportunityCandidate]:
        query = "SELECT * FROM opportunities"
        params: list[Any] = []
        if status:
            query += " WHERE status=?"
            params.append(status)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, min(int(limit), 500)))
        with self._connect() as conn:
            return [self._opportunity_from_row(row) for row in conn.execute(query, params).fetchall()]

    def update_opportunity_status(self, opportunity_id: str, status: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE opportunities SET status=?, updated_at=? WHERE opportunity_id=?",
                (status, _utciso(), opportunity_id),
            )

    def add_inbox_item(self, item: ProactiveInboxItem) -> ProactiveInboxItem:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO inbox_items (
                    inbox_id, opportunity_id, title, summary, detailed_report, status,
                    confidence, why_now, sources_json, personalization_json,
                    actions_json, created_at, updated_at, feedback
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(opportunity_id) DO UPDATE SET
                    title=excluded.title, summary=excluded.summary,
                    detailed_report=excluded.detailed_report, status=excluded.status,
                    confidence=excluded.confidence, why_now=excluded.why_now,
                    sources_json=excluded.sources_json,
                    personalization_json=excluded.personalization_json,
                    actions_json=excluded.actions_json, updated_at=excluded.updated_at
                """,
                (
                    item.inbox_id, item.opportunity_id, item.title, item.summary,
                    item.detailed_report, item.status, item.confidence, item.why_now,
                    item.sources_json, item.personalization_json, item.actions_json,
                    item.created_at, item.updated_at, item.feedback,
                ),
            )
            row = conn.execute(
                "SELECT * FROM inbox_items WHERE opportunity_id=?", (item.opportunity_id,)
            ).fetchone()
        return self._inbox_from_row(row)

    def list_inbox_items(self, *, limit: int = 50, status: str | None = None) -> list[ProactiveInboxItem]:
        query = "SELECT * FROM inbox_items"
        params: list[Any] = []
        if status:
            query += " WHERE status=?"
            params.append(status)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, min(int(limit), 500)))
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._inbox_from_row(row) for row in rows]

    def count_inbox_since(self, since_iso: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS count FROM inbox_items WHERE created_at>=?", (since_iso,)
            ).fetchone()
        return int(row["count"] if row else 0)

    def get_inbox_item(self, inbox_id: str) -> Optional[ProactiveInboxItem]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM inbox_items WHERE inbox_id=?", (inbox_id,)).fetchone()
        return self._inbox_from_row(row) if row else None

    def get_inbox_for_opportunity(self, opportunity_id: str) -> Optional[ProactiveInboxItem]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM inbox_items WHERE opportunity_id=?", (opportunity_id,)
            ).fetchone()
        return self._inbox_from_row(row) if row else None

    def record_feedback(self, inbox_id: str, feedback: str) -> bool:
        allowed = {"useful", "not_useful", "wrong_inference", "too_intrusive"}
        if feedback not in allowed:
            raise ValueError("invalid feedback")
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "UPDATE inbox_items SET feedback=?, updated_at=? WHERE inbox_id=?",
                (feedback, _utciso(), inbox_id),
            )
            return cursor.rowcount == 1

    def get_policy(self, capability: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM capability_policies WHERE capability=?", (capability,)).fetchone()
        if row is None:
            return None
        return {
            "capability": row["capability"],
            "decision": row["decision"],
            "constraints": json.loads(row["constraints_json"] or "{}"),
            "updated_at": row["updated_at"],
        }

    def list_policies(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM capability_policies ORDER BY capability").fetchall()
        return [
            {
                "capability": row["capability"],
                "decision": row["decision"],
                "constraints": json.loads(row["constraints_json"] or "{}"),
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def set_policy(self, capability: str, decision: str, constraints: dict[str, Any] | None = None) -> dict[str, Any]:
        if decision not in {"deny", "ask", "auto_reversible", "trusted_bounded"}:
            raise ValueError("invalid policy decision")
        now = _utciso()
        if constraints is None:
            existing = self.get_policy(capability)
            constraints = (existing or {}).get("constraints") or {}
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO capability_policies(capability, decision, constraints_json, updated_at)
                   VALUES (?, ?, ?, ?) ON CONFLICT(capability) DO UPDATE SET
                   decision=excluded.decision, constraints_json=excluded.constraints_json,
                   updated_at=excluded.updated_at""",
                (capability, decision, json.dumps(constraints), now),
            )
        self.audit("admin", "policy.updated", capability, {"decision": decision, "constraints": constraints})
        return self.get_policy(capability) or {}

    def record_calibration_outcome(self, capability: str, *, correct: bool, source_ref: str | None = None) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO calibration_outcomes VALUES (?, ?, ?, ?, ?)",
                (uuid.uuid4().hex, capability, int(correct), source_ref, _utciso()),
            )

    def calibration_stats(self, capability: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT COUNT(*) AS samples, COALESCE(SUM(correct), 0) AS correct
                   FROM calibration_outcomes WHERE capability=?""",
                (capability,),
            ).fetchone()
        samples = int(row["samples"] if row else 0)
        correct = int(row["correct"] if row else 0)
        return {
            "capability": capability,
            "samples": samples,
            "correct": correct,
            "precision": (correct / samples) if samples else 0.0,
        }

    def create_approval(self, approval: ApprovalGrant) -> ApprovalGrant:
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO approvals(approval_id, capability, action_fingerprint,
                   constraints_json, status, created_at, expires_at, approver, reusable, used_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    approval.approval_id, approval.capability, approval.action_fingerprint,
                    approval.constraints_json, approval.status, approval.created_at,
                    approval.expires_at, approval.approver, int(approval.reusable), approval.used_at,
                ),
            )
        return approval

    def list_approvals(self, *, status: str | None = None, limit: int = 100) -> list[ApprovalGrant]:
        query = "SELECT * FROM approvals"
        params: list[Any] = []
        if status:
            query += " WHERE status=?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, min(int(limit), 500)))
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._approval_from_row(row) for row in rows]

    def decide_approval(self, approval_id: str, *, approved: bool, approver: str = "admin") -> bool:
        status = "approved" if approved else "denied"
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "UPDATE approvals SET status=?, approver=? WHERE approval_id=? AND status='pending'",
                (status, approver, approval_id),
            )
        if cursor.rowcount:
            self.audit(approver, f"approval.{status}", approval_id, {})
        return cursor.rowcount == 1

    def find_valid_approval(self, capability: str, action_fingerprint: str) -> Optional[ApprovalGrant]:
        now = _utciso()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """SELECT * FROM approvals WHERE capability=? AND action_fingerprint=?
                   AND status='approved' AND expires_at>? ORDER BY created_at DESC LIMIT 1""",
                (capability, action_fingerprint, now),
            ).fetchone()
            if row is None:
                return None
            approval = self._approval_from_row(row)
            if not approval.reusable:
                conn.execute(
                    "UPDATE approvals SET status='used', used_at=? WHERE approval_id=?",
                    (now, approval.approval_id),
                )
        return approval

    def begin_action(self, idempotency_key: str, capability: str, tool_name: str) -> tuple[bool, Optional[str]]:
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT status, result_text FROM action_idempotency WHERE idempotency_key=?",
                (idempotency_key,),
            ).fetchone()
            if existing:
                return False, existing["result_text"]
            conn.execute(
                """INSERT INTO action_idempotency(idempotency_key, capability, tool_name,
                   status, created_at) VALUES (?, ?, ?, 'running', ?)""",
                (idempotency_key, capability, tool_name, _utciso()),
            )
        return True, None

    def finish_action(self, idempotency_key: str, result_text: str, *, status: str = "completed") -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """UPDATE action_idempotency SET status=?, result_text=?, completed_at=?
                   WHERE idempotency_key=?""",
                (status, result_text, _utciso(), idempotency_key),
            )

    def queue_run(self, *, title: str, source_kind: str, trigger_kind: str, priority: str = "medium", metadata: dict | None = None) -> ActivityRun:
        run = ActivityRun(
            run_id=uuid.uuid4().hex, created_at=_utciso(), completed_at=None,
            status="queued", source_kind=source_kind, trigger_kind=trigger_kind,
            title=title, priority=priority, metadata_json=json.dumps(metadata or {}),
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO activity_runs(run_id, created_at, completed_at, status,
                   source_kind, trigger_kind, title, summary, output_text, model,
                   session_id, parent_run_id, priority, error_text, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run.run_id, run.created_at, run.completed_at, run.status, run.source_kind,
                    run.trigger_kind, run.title, run.summary, run.output_text, run.model,
                    run.session_id, run.parent_run_id, run.priority, run.error_text, run.metadata_json,
                ),
            )
        return run

    def complete_run(self, run_id: str, *, summary: str, output_text: str, status: str = "completed", error_text: str | None = None) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """UPDATE activity_runs SET status=?, summary=?, output_text=?, error_text=?,
                   completed_at=? WHERE run_id=?""",
                (status, summary, output_text, error_text, _utciso(), run_id),
            )

    def link_entity(self, *, run_id: str, entity_type: str, entity_id: str, relation: str, metadata: dict | None = None) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO activity_links(link_id, run_id, entity_type, entity_id,
                   relation, metadata_json) VALUES (?, ?, ?, ?, ?, ?)""",
                (uuid.uuid4().hex, run_id, entity_type, entity_id, relation, json.dumps(metadata or {})),
            )

    def audit(self, actor: str, action: str, target: str, details: dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO autonomy_audit VALUES (?, ?, ?, ?, ?, ?)",
                (uuid.uuid4().hex, _utciso(), actor, action, target, json.dumps(details, ensure_ascii=False)),
            )

    def _event_from_row(self, row: sqlite3.Row) -> AmbientEvent:
        return AmbientEvent(**dict(row))

    def _opportunity_from_row(self, row: sqlite3.Row) -> OpportunityCandidate:
        return OpportunityCandidate(
            opportunity_id=row["opportunity_id"], fingerprint=row["fingerprint"],
            title=row["title"], goal=row["goal"], rationale=row["rationale"],
            source_event_ids=json.loads(row["source_event_ids_json"] or "[]"),
            expected_value=row["expected_value"], urgency=row["urgency"],
            confidence=row["confidence"], cost_of_wrong=row["cost_of_wrong"],
            personalization_benefit=row["personalization_benefit"],
            evidence_gaps=json.loads(row["evidence_gaps_json"] or "[]"),
            status=row["status"], created_at=row["created_at"], updated_at=row["updated_at"],
            expires_at=row["expires_at"], metadata_json=row["metadata_json"],
        )

    def _inbox_from_row(self, row: sqlite3.Row) -> ProactiveInboxItem:
        return ProactiveInboxItem(**dict(row))

    def _approval_from_row(self, row: sqlite3.Row) -> ApprovalGrant:
        return ApprovalGrant(
            approval_id=row["approval_id"], capability=row["capability"],
            action_fingerprint=row["action_fingerprint"], constraints_json=row["constraints_json"],
            status=row["status"], created_at=row["created_at"], expires_at=row["expires_at"],
            approver=row["approver"], reusable=bool(row["reusable"]), used_at=row["used_at"],
        )
