import hashlib
import json
import re
import threading
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

from application.ports.autonomy_port import AutonomyStorePort
from core.models import ApprovalGrant, PolicyDecision


@dataclass(frozen=True)
class ToolCapabilityDescriptor:
    tool_name: str
    capability: str
    capability_pack: str
    access: str
    external_effect: bool
    reversible: bool
    risk_class: str
    data_scopes: tuple[str, ...]
    verification: str
    budget_cost: int = 1
    idempotent: bool = False


class ApprovalRequiredError(PermissionError):
    def __init__(self, message: str, *, approval_id: str, decision: PolicyDecision):
        super().__init__(message)
        self.approval_id = approval_id
        self.decision = decision


class PolicyDeniedError(PermissionError):
    pass


class CapabilityRegistry:
    """Classify MCP and direct tools into a small, coherent capability surface."""

    _READ_PREFIXES = ("get_", "list_", "view_", "read_", "search", "google_search")
    _HARD_ACTION_WORDS = re.compile(
        r"\b(purchase|buy|checkout|delete|erase|remove account|change password|credential|"
        r"publish|post publicly|send payment|transfer money)\b",
        re.IGNORECASE,
    )
    _BROWSER_MUTATION_WORDS = re.compile(
        r"\b(send|submit|apply|book|reserve|purchase|buy|checkout|delete|upload|post|"
        r"message|email|login|sign in|change|edit|confirm)\b",
        re.IGNORECASE,
    )

    def describe(self, tool_name: str, arguments: Optional[dict[str, Any]] = None) -> ToolCapabilityDescriptor:
        name = str(tool_name or "").strip()
        lowered = name.lower()
        args_text = json.dumps(arguments or {}, ensure_ascii=False).lower()

        if lowered == "capture_screen" or lowered.startswith(("memory_", "context_")):
            return self._d(name, "context.observe", "context", "read", False, True, "low", ("screen",), "result")
        if lowered.startswith("browser_"):
            read_only_browser = any(
                token in lowered
                for token in ("navigate", "snapshot", "screenshot", "tabs", "console", "network", "wait", "close")
            )
            if read_only_browser:
                return self._d(name, "research.web", "research", "read", False, True, "low", ("browser", "web"), "sources", 1)
            return self._d(name, "browser.mutate", "communication", "write", True, False, "high", ("browser", "web"), "screen_state", 2)
        if lowered == "use_browser":
            mutating = bool(self._BROWSER_MUTATION_WORDS.search(args_text))
            if mutating:
                return self._d(name, "browser.mutate", "communication", "write", True, False, "high", ("browser", "web"), "screen_state", 3)
            return self._d(name, "research.web", "research", "read", False, True, "low", ("web",), "sources", 3)
        if lowered in {"google_search", "tavily_search"} or lowered.startswith("search"):
            return self._d(name, "research.web", "research", "read", False, True, "low", ("web",), "sources", 2)
        if lowered in {"add_task", "queue_night_task", "schedule_task_at"} or "reminder" in lowered:
            return self._d(name, "assistance.reminder", "personal_assistance", "write", True, True, "medium", ("tasks",), "provider_readback", 1, True)
        if "calendar" in lowered and lowered.startswith(self._READ_PREFIXES):
            return self._d(name, "assistance.calendar.read", "personal_assistance", "read", False, True, "low", ("calendar",), "result", 1)
        if lowered in {"schedule_meeting", "create_event", "update_event", "delete_event"} or "calendar" in lowered:
            return self._d(name, "communication.calendar.write", "communication", "write", True, lowered.startswith("delete") is False, "high", ("calendar",), "provider_readback", 2, True)
        if any(word in lowered for word in ("email", "gmail", "message", "slack", "teams", "outlook")):
            if lowered.startswith(self._READ_PREFIXES):
                return self._d(name, "communication.read", "communication", "read", False, True, "medium", ("communications",), "result", 1)
            return self._d(name, "communication.send", "communication", "write", True, False, "high", ("communications",), "provider_readback", 2, True)
        if lowered == "powershell_terminal" or any(word in lowered for word in ("terminal", "shell", "command")):
            return self._d(name, "system.shell", "system_operations", "write", True, False, "critical", ("filesystem", "processes"), "explicit", 4)
        if "transaction" in lowered or "finance" in lowered or "payment" in lowered:
            read_only = lowered.startswith(("view_", "get_", "list_"))
            return self._d(name, "finance.read" if read_only else "finance.mutate", "system_operations", "read" if read_only else "write", not read_only, read_only, "medium" if read_only else "critical", ("financial",), "provider_readback", 2, not read_only)
        if "download" in lowered:
            return self._d(name, "system.download", "system_operations", "write", True, True, "high", ("filesystem", "web"), "file_exists", 3)
        if lowered in {"load_agent", "restore_previous_agent", "list_available_models"}:
            return self._d(name, "system.model", "system_operations", "write", False, True, "medium", ("model_runtime",), "runtime_state")
        if lowered.startswith(self._READ_PREFIXES) or lowered in {"get_current_datetime", "add"}:
            return self._d(name, "context.read", "context", "read", False, True, "low", ("local",), "result")
        return self._d(name, "unclassified", "system_operations", "write", True, False, "high", ("unknown",), "explicit")

    def is_hard_boundary(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        descriptor = self.describe(tool_name, arguments)
        if descriptor.capability in {"finance.mutate", "system.shell"}:
            return True
        return bool(self._HARD_ACTION_WORDS.search(json.dumps(arguments, ensure_ascii=False)))

    @staticmethod
    def _d(
        tool_name: str, capability: str, pack: str, access: str, external: bool,
        reversible: bool, risk: str, scopes: tuple[str, ...], verification: str,
        cost: int = 1, idempotent: bool = False,
    ) -> ToolCapabilityDescriptor:
        return ToolCapabilityDescriptor(
            tool_name=tool_name, capability=capability, capability_pack=pack,
            access=access, external_effect=external, reversible=reversible,
            risk_class=risk, data_scopes=scopes, verification=verification,
            budget_cost=cost, idempotent=idempotent,
        )


class AutonomyBudget:
    def __init__(self, *, max_tool_calls_per_hour: int = 120, max_web_queries_per_day: int = 60):
        self.max_tool_calls_per_hour = max(1, max_tool_calls_per_hour)
        self.max_web_queries_per_day = max(1, max_web_queries_per_day)
        self._tool_calls: deque[datetime] = deque()
        self._web_queries: deque[datetime] = deque()
        self._lock = threading.Lock()

    def consume(self, descriptor: ToolCapabilityDescriptor) -> bool:
        now = datetime.now(timezone.utc)
        with self._lock:
            self._trim(now)
            if len(self._tool_calls) + descriptor.budget_cost > self.max_tool_calls_per_hour:
                return False
            if descriptor.capability == "research.web" and len(self._web_queries) >= self.max_web_queries_per_day:
                return False
            self._tool_calls.extend([now] * descriptor.budget_cost)
            if descriptor.capability == "research.web":
                self._web_queries.append(now)
        return True

    def _trim(self, now: datetime) -> None:
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        while self._tool_calls and self._tool_calls[0] < hour_ago:
            self._tool_calls.popleft()
        while self._web_queries and self._web_queries[0] < day_ago:
            self._web_queries.popleft()


class CapabilityPolicyService:
    DEFAULT_POLICIES = {
        "context.observe": "auto_reversible",
        "context.read": "auto_reversible",
        "research.web": "auto_reversible",
        "assistance.artifact": "auto_reversible",
        "assistance.reminder": "auto_reversible",
        "assistance.calendar.read": "auto_reversible",
        "communication.read": "ask",
        "communication.calendar.write": "ask",
        "communication.send": "ask",
        "browser.mutate": "ask",
        "system.shell": "deny",
        "system.download": "deny",
        "system.model": "deny",
        "finance.read": "ask",
        "finance.mutate": "deny",
        "unclassified": "deny",
    }
    DEFAULT_CONSTRAINTS = {
        "assistance.reminder": {
            "requires_calibration": True,
            "minimum_samples": 20,
            "minimum_precision": 0.95,
        }
    }

    def __init__(
        self,
        *,
        store: AutonomyStorePort,
        registry: CapabilityRegistry | None = None,
        budget: AutonomyBudget | None = None,
        reminder_confidence_threshold: float = 0.85,
    ):
        self.store = store
        self.registry = registry or CapabilityRegistry()
        self.budget = budget or AutonomyBudget()
        self.reminder_confidence_threshold = reminder_confidence_threshold
        for capability, decision in self.DEFAULT_POLICIES.items():
            if self.store.get_policy(capability) is None:
                self.store.set_policy(
                    capability, decision, dict(self.DEFAULT_CONSTRAINTS.get(capability, {}))
                )

    def filter_tools(self, tools: Iterable[dict[str, Any]], *, source: str, confidence: float) -> list[dict[str, Any]]:
        result = []
        for tool in tools:
            name = str(tool.get("function", {}).get("name") or "")
            descriptor = self.registry.describe(name)
            decision = self._policy_for(descriptor.capability)
            if source.startswith("autonomy"):
                if decision not in {"auto_reversible", "trusted_bounded"}:
                    continue
                if descriptor.capability == "assistance.reminder" and confidence < self.reminder_confidence_threshold:
                    continue
                if descriptor.risk_class in {"high", "critical"}:
                    continue
            result.append(tool)
        return result

    def evaluate(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        source: str,
        confidence: float,
        explicit_user_request: bool = False,
        evidence_context: Optional[dict[str, Any]] = None,
    ) -> PolicyDecision:
        descriptor = self.registry.describe(tool_name, arguments)
        configured = self.store.get_policy(descriptor.capability)
        policy = (configured or {}).get("decision") or self.DEFAULT_POLICIES.get(descriptor.capability, "deny")
        constraints = (configured or {}).get("constraints") or {}

        if source.startswith("autonomy") and descriptor.capability == "assistance.reminder" and confidence < self.reminder_confidence_threshold:
            return PolicyDecision("ask", descriptor.capability, "reminder_confidence", "Reminder confidence is below the automatic threshold.", True)
        if source.startswith("autonomy") and descriptor.capability == "assistance.reminder":
            source_url = str(arguments.get("source_url") or "").strip()
            evidence_text = str((evidence_context or {}).get("observed_text") or "")
            source_verified = (
                arguments.get("source_verified") is True
                and source_url in evidence_text
            )
            due_datetime = str(arguments.get("due_datetime") or arguments.get("run_at") or "").strip()
            due_date = due_datetime[:10]
            date_corroborated = bool(due_date and due_date in evidence_text)
            if not (
                source_verified
                and source_url.startswith(("https://", "http://"))
                and due_datetime
                and date_corroborated
            ):
                return PolicyDecision(
                    "ask",
                    descriptor.capability,
                    "authoritative_deadline",
                    "An inferred reminder needs a verified authoritative source and an exact due time.",
                    True,
                )
            if constraints.get("requires_calibration", True) and hasattr(self.store, "calibration_stats"):
                stats = self.store.calibration_stats(descriptor.capability)
                minimum_samples = int(constraints.get("minimum_samples", 20))
                minimum_precision = float(constraints.get("minimum_precision", 0.95))
                if stats["samples"] < minimum_samples or stats["precision"] < minimum_precision:
                    return PolicyDecision(
                        "ask",
                        descriptor.capability,
                        "calibration_gate",
                        f"Automatic reminders remain gated until shadow review reaches {minimum_precision:.0%} precision across {minimum_samples} samples.",
                        True,
                        {**constraints, "calibration": stats},
                    )

        hard_boundary = self.registry.is_hard_boundary(tool_name, arguments)
        if hard_boundary and policy != "deny":
            policy = "ask"
        if policy == "deny":
            return PolicyDecision("deny", descriptor.capability, "capability_policy", "This capability is denied for this action source.")
        if policy == "ask":
            if explicit_user_request and not hard_boundary:
                return PolicyDecision("auto_reversible", descriptor.capability, "explicit_user_request", "The current explicit user request permits this bounded action.")
            return PolicyDecision("ask", descriptor.capability, "capability_policy", "This action requires explicit approval.", True, constraints)
        if policy == "auto_reversible" and (not descriptor.reversible or descriptor.risk_class in {"high", "critical"}):
            return PolicyDecision("ask", descriptor.capability, "reversibility_guard", "High-risk or irreversible work requires approval.", True, constraints)
        if policy == "trusted_bounded" and not self._within_constraints(arguments, constraints):
            return PolicyDecision("ask", descriptor.capability, "trusted_bounds", "The action exceeds configured trusted bounds.", True, constraints)
        if not self.budget.consume(descriptor):
            return PolicyDecision("deny", descriptor.capability, "budget", "The capability budget is exhausted; defer this action.")
        return PolicyDecision(policy, descriptor.capability, "capability_policy", "Action is allowed by the configured bounded policy.", False, constraints)

    def authorize_or_raise(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        source: str,
        confidence: float,
        explicit_user_request: bool = False,
        evidence_context: Optional[dict[str, Any]] = None,
    ) -> PolicyDecision:
        decision = self.evaluate(
            tool_name=tool_name, arguments=arguments, source=source,
            confidence=confidence, explicit_user_request=explicit_user_request,
            evidence_context=evidence_context,
        )
        if decision.decision == "deny":
            raise PolicyDeniedError(decision.rationale)
        if decision.requires_approval:
            fingerprint = self.action_fingerprint(tool_name, arguments)
            existing = self.store.find_valid_approval(decision.capability, fingerprint)
            if existing is not None:
                return PolicyDecision("trusted_bounded", decision.capability, "approval", "A valid scoped approval authorizes this action.")
            now = datetime.now(timezone.utc)
            approval = ApprovalGrant(
                approval_id=uuid.uuid4().hex,
                capability=decision.capability,
                action_fingerprint=fingerprint,
                constraints_json=json.dumps({"tool_name": tool_name, "arguments": arguments}, ensure_ascii=False),
                status="pending",
                created_at=now.isoformat(),
                expires_at=(now + timedelta(minutes=30)).isoformat(),
                approver="pending",
            )
            self.store.create_approval(approval)
            raise ApprovalRequiredError(
                f"Approval required for capability '{decision.capability}' (approval {approval.approval_id}).",
                approval_id=approval.approval_id,
                decision=decision,
            )
        return decision

    def begin_execution(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        scope: str,
    ) -> tuple[Optional[str], bool, Optional[str]]:
        descriptor = self.registry.describe(tool_name, arguments)
        if not descriptor.idempotent or not hasattr(self.store, "begin_action"):
            return None, True, None
        key_material = f"{scope}|{self.action_fingerprint(tool_name, arguments)}"
        key = hashlib.sha256(key_material.encode("utf-8")).hexdigest()
        started, prior_result = self.store.begin_action(key, descriptor.capability, tool_name)
        return key, started, prior_result

    def finish_execution(self, key: Optional[str], result: str, *, succeeded: bool) -> None:
        if key and hasattr(self.store, "finish_action"):
            self.store.finish_action(key, result, status="completed" if succeeded else "failed")

    @staticmethod
    def action_fingerprint(tool_name: str, arguments: dict[str, Any]) -> str:
        canonical = json.dumps({"tool": tool_name, "arguments": arguments}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _policy_for(self, capability: str) -> str:
        configured = self.store.get_policy(capability)
        return (configured or {}).get("decision") or self.DEFAULT_POLICIES.get(capability, "deny")

    @staticmethod
    def _within_constraints(arguments: dict[str, Any], constraints: dict[str, Any]) -> bool:
        if not constraints:
            return True
        serialized = json.dumps(arguments, ensure_ascii=False).lower()
        allowed_domains = [str(item).lower() for item in constraints.get("allowed_domains", [])]
        if allowed_domains and not any(domain in serialized for domain in allowed_domains):
            return False
        allowed_recipients = [str(item).lower() for item in constraints.get("allowed_recipients", [])]
        if allowed_recipients:
            recipient = str(arguments.get("recipient") or arguments.get("to") or "").lower()
            if recipient not in allowed_recipients:
                return False
        max_amount = constraints.get("max_amount")
        if max_amount is not None and "amount" in arguments:
            try:
                if float(arguments["amount"]) > float(max_amount):
                    return False
            except (TypeError, ValueError):
                return False
        return True
