"""Closure Ethics reference governor v0.1.

This module implements the generic decision layer described by Closure Ethics.
It does NOT infer real-world harm, consent, authorization, or deception from raw
observations. Domain adapters must provide auditable evidence for those inputs.

Text/specification: CC BY 4.0. Reference code: MIT (consistent with repository).
Project Möbia and Marek Zajda, 2026.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import hashlib
import heapq
import json
import math

EPS = 1e-12


def _clip01(x: float) -> float:
    if not math.isfinite(x):
        raise ValueError("metric values must be finite")
    return min(1.0, max(0.0, float(x)))


@dataclass(frozen=True)
class Interval:
    """Closed confidence/envelope interval in [0, 1]."""
    low: float
    high: float

    def __post_init__(self) -> None:
        lo, hi = _clip01(self.low), _clip01(self.high)
        if lo > hi:
            raise ValueError("Interval.low must be <= Interval.high")
        object.__setattr__(self, "low", lo)
        object.__setattr__(self, "high", hi)

    @classmethod
    def point(cls, x: float) -> "Interval":
        x = _clip01(x)
        return cls(x, x)


@dataclass(frozen=True)
class Policy:
    """Policy thresholds. Values are governance choices, not moral constants."""
    max_catastrophic_risk: float
    min_agency: float
    min_auditability: float
    min_repairability: float
    min_authorization: float
    max_deception_risk: float
    min_reversibility: float
    metric_resolution: float = 1e-6
    version: str = "closure-ethics-v0.1"

    def __post_init__(self) -> None:
        fields = (
            "max_catastrophic_risk", "min_agency", "min_auditability",
            "min_repairability", "min_authorization", "max_deception_risk",
            "min_reversibility",
        )
        for name in fields:
            value = getattr(self, name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1]")
        if self.metric_resolution <= 0:
            raise ValueError("metric_resolution must be > 0")

    def fingerprint(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()[:16]


# Demonstration profile only. Deployments MUST review and version their own values.
DEMO_POLICY = Policy(
    max_catastrophic_risk=0.05,
    min_agency=0.60,
    min_auditability=0.70,
    min_repairability=0.50,
    min_authorization=0.70,
    max_deception_risk=0.10,
    min_reversibility=0.50,
)


@dataclass(frozen=True)
class AuditEvidence:
    logging: Interval
    attribution: Interval
    independent_review: Interval


@dataclass(frozen=True)
class ReversibilityEvidence:
    rollback: Interval
    revocability: Interval
    containability: Interval


@dataclass(frozen=True)
class AuthorizationEvidence:
    scope_match: Interval
    consent_or_mandate: Interval
    least_privilege: Interval


@dataclass(frozen=True)
class DeceptionEvidence:
    concealment: Interval
    covert_persistence: Interval
    unauthorized_replication: Interval
    material_misreporting: Interval


@dataclass(frozen=True)
class RepairEvidence:
    """Bounded repair certificate normalized to configured budgets."""
    exists: bool
    cost_ratio: Interval
    risk_ratio: Interval
    path: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ActionEvidence:
    action_id: str
    catastrophic_risk: Interval
    agency_before: Mapping[str, float]
    agency_after: Mapping[str, float]
    audit: AuditEvidence
    repair: RepairEvidence
    reversibility: ReversibilityEvidence
    authorization: AuthorizationEvidence
    deception: DeceptionEvidence
    task_utility: Interval
    provenance: str = ""


@dataclass(frozen=True)
class Metrics:
    H: float
    P: float
    O: float
    R: float
    U: float
    D: float
    K: float
    Q: float


@dataclass(frozen=True)
class Evaluation:
    action_id: str
    metrics: Metrics
    admissible: bool
    violations: Tuple[str, ...]
    provenance: str


@dataclass(frozen=True)
class Decision:
    status: str
    selected_action: Optional[str]
    reason_codes: Tuple[str, ...]
    policy_version: str
    policy_fingerprint: str
    evaluations: Tuple[Evaluation, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2)


def _effective_number(capacity: Mapping[str, float]) -> float:
    vals = [max(0.0, float(v)) for v in capacity.values()]
    total = sum(vals)
    if total <= EPS:
        return 0.0
    shares = [v / total for v in vals]
    return 1.0 / max(EPS, sum(s * s for s in shares))


def agency_score(before: Mapping[str, float], after: Mapping[str, float]) -> float:
    """Conservative agency score: individual retention AND plurality retention."""
    keys = sorted(set(before) | set(after))
    if not keys:
        return 1.0

    retention: List[float] = []
    for k in keys:
        b = max(0.0, float(before.get(k, 0.0)))
        a = max(0.0, float(after.get(k, 0.0)))
        if b > EPS:
            retention.append(min(1.0, a / b))
    individual = min(retention) if retention else 1.0

    neff_before = _effective_number({k: before.get(k, 0.0) for k in keys})
    neff_after = _effective_number({k: after.get(k, 0.0) for k in keys})
    if neff_before <= EPS:
        diversity_retention = 1.0 if neff_after <= EPS else 0.0
    else:
        diversity_retention = min(1.0, neff_after / neff_before)
    return _clip01(min(individual, diversity_retention))


def compute_metrics(e: ActionEvidence) -> Metrics:
    """Collapse uncertainty pessimistically, then compute structural metrics."""
    H = e.catastrophic_risk.high
    P = agency_score(e.agency_before, e.agency_after)
    O = min(e.audit.logging.low, e.audit.attribution.low, e.audit.independent_review.low)

    if not e.repair.exists:
        R = 0.0
    else:
        worst_budget_fraction = max(e.repair.cost_ratio.high, e.repair.risk_ratio.high)
        R = _clip01(1.0 - worst_budget_fraction)

    U = min(
        e.authorization.scope_match.low,
        e.authorization.consent_or_mandate.low,
        e.authorization.least_privilege.low,
    )
    D = max(
        e.deception.concealment.high,
        e.deception.covert_persistence.high,
        e.deception.unauthorized_replication.high,
        e.deception.material_misreporting.high,
    )
    K = min(
        e.reversibility.rollback.low,
        e.reversibility.revocability.low,
        e.reversibility.containability.low,
    )
    Q = e.task_utility.low
    return Metrics(H=H, P=P, O=O, R=R, U=U, D=D, K=K, Q=Q)


def violations(m: Metrics, p: Policy) -> Tuple[str, ...]:
    out: List[str] = []
    if m.H > p.max_catastrophic_risk: out.append("CATASTROPHIC_RISK")
    if m.P < p.min_agency: out.append("AGENCY_LOSS")
    if m.O < p.min_auditability: out.append("AUDITABILITY_LOSS")
    if m.R < p.min_repairability: out.append("NO_BOUNDED_REPAIR")
    if m.U < p.min_authorization: out.append("AUTHORIZATION_FAILURE")
    if m.D > p.max_deception_risk: out.append("DECEPTION_OR_COVERT_PERSISTENCE")
    if m.K < p.min_reversibility: out.append("INSUFFICIENT_REVERSIBILITY")
    return tuple(out)


def evaluate(e: ActionEvidence, p: Policy) -> Evaluation:
    m = compute_metrics(e)
    bad = violations(m, p)
    return Evaluation(e.action_id, m, len(bad) == 0, bad, e.provenance)


def _quantize(x: float, resolution: float) -> float:
    return round(x / resolution) * resolution


def _lexicographic_key(ev: Evaluation, p: Policy) -> Tuple[float, ...]:
    """Lower tuple is preferred; task utility is deliberately last."""
    m = ev.metrics
    q = lambda x: _quantize(x, p.metric_resolution)
    return (q(m.H), -q(m.P), -q(m.O), -q(m.R), -q(m.U), q(m.D), -q(m.K), -q(m.Q))


def _fallback_reason(evaluations: Sequence[Evaluation]) -> Tuple[str, ...]:
    priority = (
        ("CATASTROPHIC_RISK", "REFUSE_OR_CONTAIN"),
        ("AGENCY_LOSS", "SEEK_AGENCY_PRESERVING_ALTERNATIVE"),
        ("AUDITABILITY_LOSS", "REQUIRE_AUDIT_PATH"),
        ("NO_BOUNDED_REPAIR", "SANDBOX_STAGE_OR_ADD_REPAIR_PATH"),
        ("AUTHORIZATION_FAILURE", "SEEK_AUTHORIZATION_OR_ESCALATE"),
        ("DECEPTION_OR_COVERT_PERSISTENCE", "REFUSE_COVERT_ACTION"),
        ("INSUFFICIENT_REVERSIBILITY", "USE_REVERSIBLE_ALTERNATIVE"),
    )
    present = {v for ev in evaluations for v in ev.violations}
    return tuple(reason for violation, reason in priority if violation in present) or ("NO_ADMISSIBLE_ACTION",)


class ClosureEthicsGovernor:
    """Reference governor placed between planner and executor."""
    def __init__(self, policy: Policy):
        self.policy = policy

    def choose(self, candidates: Sequence[ActionEvidence], recovery_candidates: Sequence[ActionEvidence] = ()) -> Decision:
        primary = tuple(evaluate(c, self.policy) for c in candidates)
        admissible = [ev for ev in primary if ev.admissible]
        if admissible:
            selected = min(admissible, key=lambda ev: _lexicographic_key(ev, self.policy))
            return Decision("EXECUTE", selected.action_id, ("ADMISSIBLE_LEXICOGRAPHIC_SELECTION",), self.policy.version, self.policy.fingerprint(), primary)

        recovery = tuple(evaluate(c, self.policy) for c in recovery_candidates)
        recovery_ok = [ev for ev in recovery if ev.admissible]
        all_evals = primary + recovery
        if recovery_ok:
            selected = min(recovery_ok, key=lambda ev: _lexicographic_key(ev, self.policy))
            return Decision("EXECUTE_RECOVERY", selected.action_id, ("PRIMARY_INADMISSIBLE", "RECOVERY_ACTION_SELECTED"), self.policy.version, self.policy.fingerprint(), all_evals)

        return Decision("REFUSE_OR_ESCALATE", None, _fallback_reason(all_evals), self.policy.version, self.policy.fingerprint(), all_evals)

    def evaluate_joint_action(self, composed_evidence: ActionEvidence) -> Evaluation:
        """Re-evaluate the COMPOSED transition; never average local scores."""
        return evaluate(composed_evidence, self.policy)


@dataclass(frozen=True)
class RepairEdge:
    target: str
    cost: float
    risk: float


@dataclass(frozen=True)
class RepairCertificate:
    exists: bool
    total_cost: float
    total_risk: float
    path: Tuple[str, ...]


def bounded_repair_search(graph: Mapping[str, Sequence[RepairEdge]], start: str, safe_states: Iterable[str], max_cost: float, max_risk: float) -> RepairCertificate:
    """Pareto label-setting path search satisfying BOTH repair budgets."""
    if max_cost <= 0 or max_risk <= 0:
        raise ValueError("repair budgets must be > 0")
    safe = set(safe_states)
    if start in safe:
        return RepairCertificate(True, 0.0, 0.0, (start,))

    labels: Dict[str, List[Tuple[float, float]]] = {start: [(0.0, 0.0)]}
    heap: List[Tuple[float, float, str, Tuple[str, ...]]] = [(0.0, 0.0, start, (start,))]

    while heap:
        cost, risk, node, path = heapq.heappop(heap)
        if node in safe:
            return RepairCertificate(True, cost, risk, path)
        for edge in graph.get(node, ()):
            nc = cost + max(0.0, edge.cost)
            nr = risk + max(0.0, edge.risk)
            if nc > max_cost + EPS or nr > max_risk + EPS:
                continue
            existing = labels.setdefault(edge.target, [])
            if any(c <= nc + EPS and r <= nr + EPS for c, r in existing):
                continue
            existing[:] = [(c, r) for c, r in existing if not (nc <= c + EPS and nr <= r + EPS)]
            existing.append((nc, nr))
            heapq.heappush(heap, (nc, nr, edge.target, path + (edge.target,)))

    return RepairCertificate(False, math.inf, math.inf, ())


def repair_evidence_from_certificate(cert: RepairCertificate, max_cost: float, max_risk: float) -> RepairEvidence:
    if not cert.exists:
        return RepairEvidence(False, Interval.point(1.0), Interval.point(1.0), ())
    return RepairEvidence(
        True,
        Interval.point(min(1.0, cert.total_cost / max_cost)),
        Interval.point(min(1.0, cert.total_risk / max_risk)),
        cert.path,
    )
