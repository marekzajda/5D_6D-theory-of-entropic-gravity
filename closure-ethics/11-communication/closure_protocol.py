from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import FrozenSet, Iterable, MutableMapping, Set, Tuple


class ProtocolState(IntEnum):
    """Recovered S0-S7 communication state machine."""

    S0_DETECTION = 0
    S1_ARTIFICIAL_STRUCTURE_CONFIRMED = 1
    S2_ELEMENTARY_ALPHABET = 2
    S3_NUMBERS_AND_OPERATIONS = 3
    S4_RELATIONAL_GRAMMAR = 4
    S5_PREDICTIVE_CLOSURE = 5
    S6_SHARED_MODEL_OF_REALITY = 6
    S7_SAFE_CONTENT_EXCHANGE = 7


class CommunicationTier(IntEnum):
    """Historical T0-T4 information/capability separation."""

    T0_OBSERVATION = 0
    T1_FORMAL_RELATIONS = 1
    T2_DESCRIPTIVE_MODELS = 2
    T3_CONSTRAINED_EXPERIMENTS = 3
    T4_OPERATIONAL_ACTIONS = 4


@dataclass(frozen=True)
class ClosureWeights:
    structural: float = 0.25
    predictive: float = 0.35
    semantic_echo: float = 0.20
    repeatability: float = 0.20

    def normalized(self) -> "ClosureWeights":
        values = (
            self.structural,
            self.predictive,
            self.semantic_echo,
            self.repeatability,
        )
        if any(v < 0 for v in values):
            raise ValueError("Closure weights must be non-negative.")
        total = sum(values)
        if total <= 0:
            raise ValueError("At least one closure weight must be positive.")
        return ClosureWeights(*(v / total for v in values))


@dataclass(frozen=True)
class ExchangeEvidence:
    """Observable evidence used by the historical C_Ω communication index."""

    structural: float
    predictive: float
    semantic_echo: float
    repeatability: float

    def __post_init__(self) -> None:
        for name, value in (
            ("structural", self.structural),
            ("predictive", self.predictive),
            ("semantic_echo", self.semantic_echo),
            ("repeatability", self.repeatability),
        ):
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0,1], got {value!r}.")


def closure_index(
    evidence: ExchangeEvidence,
    weights: ClosureWeights = ClosureWeights(),
) -> float:
    """C_Ω = w_s C_s + w_p C_p + w_e C_e + w_r C_r."""

    w = weights.normalized()
    return (
        w.structural * evidence.structural
        + w.predictive * evidence.predictive
        + w.semantic_echo * evidence.semantic_echo
        + w.repeatability * evidence.repeatability
    )


@dataclass(frozen=True)
class ProtocolPolicy:
    threshold: float = 0.85
    independent_trials: int = 3
    weights: ClosureWeights = ClosureWeights()

    def __post_init__(self) -> None:
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError("threshold must be in [0,1].")
        if self.independent_trials < 1:
            raise ValueError("independent_trials must be >= 1.")


@dataclass(frozen=True)
class ExchangeDecision:
    accepted_as_independent_evidence: bool
    relation_shared: bool
    score: float
    state_before: ProtocolState
    state_after: ProtocolState
    reason: str


@dataclass
class ClosureHandshake:
    """Reference implementation of predictive relation sharing.

    A relation is promoted only after enough independent, novel challenges pass
    the configured C_Ω threshold. Low-confidence evidence rolls the protocol back.
    A severe safety conflict resets communication to S0.
    """

    policy: ProtocolPolicy = field(default_factory=ProtocolPolicy)
    state: ProtocolState = ProtocolState.S0_DETECTION
    _passed_challenges: MutableMapping[str, Set[str]] = field(default_factory=dict)
    shared_relations: Set[str] = field(default_factory=set)

    @staticmethod
    def _next_state(state: ProtocolState) -> ProtocolState:
        return ProtocolState(
            min(int(state) + 1, int(ProtocolState.S7_SAFE_CONTENT_EXCHANGE))
        )

    @staticmethod
    def _previous_state(state: ProtocolState) -> ProtocolState:
        return ProtocolState(
            max(int(state) - 1, int(ProtocolState.S0_DETECTION))
        )

    def observe(
        self,
        *,
        relation_id: str,
        challenge_id: str,
        evidence: ExchangeEvidence,
        novel_challenge: bool,
        safety_conflict: bool = False,
    ) -> ExchangeDecision:
        if not relation_id or not challenge_id:
            raise ValueError("relation_id and challenge_id must be non-empty.")

        before = self.state
        score = closure_index(evidence, self.policy.weights)

        if safety_conflict:
            self.state = ProtocolState.S0_DETECTION
            self._passed_challenges.clear()
            return ExchangeDecision(
                False,
                relation_id in self.shared_relations,
                score,
                before,
                self.state,
                "severe safety conflict: reset to S0",
            )

        seen = self._passed_challenges.setdefault(relation_id, set())
        independent = novel_challenge and challenge_id not in seen

        if score >= self.policy.threshold and independent:
            seen.add(challenge_id)
            if len(seen) >= self.policy.independent_trials:
                self.shared_relations.add(relation_id)
                self.state = self._next_state(self.state)
                return ExchangeDecision(
                    True,
                    True,
                    score,
                    before,
                    self.state,
                    "threshold satisfied across independent novel challenges",
                )
            return ExchangeDecision(
                True,
                False,
                score,
                before,
                self.state,
                "high-confidence novel challenge recorded; more independent trials required",
            )

        # Failed or non-independent evidence must not accumulate semantic authority.
        if score < self.policy.threshold:
            self.state = self._previous_state(self.state)
            seen.clear()
            reason = "closure score below threshold: rollback one state"
        elif not novel_challenge:
            reason = "copied/non-novel response is not independent closure evidence"
        else:
            reason = "replayed challenge is not independent closure evidence"

        return ExchangeDecision(
            False,
            relation_id in self.shared_relations,
            score,
            before,
            self.state,
            reason,
        )

    def understands(self, relation_id: str) -> bool:
        """Semantic relation confidence only; never an authorization decision."""

        return relation_id in self.shared_relations


@dataclass(frozen=True)
class ActionEnvelope:
    """Independent action-authority gate.

    This deliberately prevents communication confidence from becoming authority.
    """

    authenticated: bool
    authorized: bool
    in_scope: bool
    fresh: bool
    integrity_ok: bool
    policy_admissible: bool

    def permits(self) -> bool:
        return all(
            (
                self.authenticated,
                self.authorized,
                self.in_scope,
                self.fresh,
                self.integrity_ok,
                self.policy_admissible,
            )
        )


def operationally_authorized(
    *,
    tier: CommunicationTier,
    envelope: ActionEnvelope,
) -> Tuple[bool, str]:
    """Enforce UNDERSTAND(message) != AUTHORIZE(action)."""

    if tier != CommunicationTier.T4_OPERATIONAL_ACTIONS:
        return False, "semantic/communication tier does not grant operational capability"
    if not envelope.authenticated:
        return False, "sender identity is not authenticated"
    if not envelope.authorized:
        return False, "sender is not authorized for the requested action"
    if not envelope.in_scope:
        return False, "requested action exceeds delegated scope"
    if not envelope.fresh:
        return False, "message is stale or replayed"
    if not envelope.integrity_ok:
        return False, "message/provenance integrity failed"
    if not envelope.policy_admissible:
        return False, "requested transition failed Closure Ethics policy"
    return (
        True,
        "operational action is separately authenticated, authorized, scoped, fresh, intact and admissible",
    )


@dataclass(frozen=True)
class DelegationResult:
    granted: FrozenSet[str]
    denied: FrozenSet[str]

    @property
    def amplified(self) -> bool:
        """True means the request attempted to exceed the admissible delegation ceiling."""

        return bool(self.denied)


def bounded_delegation(
    *,
    parent_capabilities: Iterable[str],
    requested_capabilities: Iterable[str],
    delegation_scope: Iterable[str],
) -> DelegationResult:
    """Enforce C_j ⊆ C_i ∩ Scope(d_i→j)."""

    parent = set(parent_capabilities)
    requested = set(requested_capabilities)
    scope = set(delegation_scope)

    allowed_ceiling = parent & scope
    granted = requested & allowed_ceiling
    denied = requested - granted
    return DelegationResult(frozenset(granted), frozenset(denied))
