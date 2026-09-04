from closure_ethics import *


def I(x):
    return Interval.point(x)


governor = ClosureEthicsGovernor(DEMO_POLICY)

unsafe_direct = ActionEvidence(
    action_id="deploy-directly",
    catastrophic_risk=Interval(0.01, 0.04),
    agency_before={"operator": 1.0, "reviewer": 1.0},
    agency_after={"operator": 0.8, "reviewer": 0.2},
    audit=AuditEvidence(I(0.4), I(0.6), I(0.2)),
    repair=RepairEvidence(False, I(1.0), I(1.0)),
    reversibility=ReversibilityEvidence(I(0.2), I(0.3), I(0.4)),
    authorization=AuthorizationEvidence(I(0.9), I(0.9), I(0.8)),
    deception=DeceptionEvidence(I(0.0), I(0.0), I(0.0), I(0.0)),
    task_utility=I(0.95),
    provenance="planner:v3; simulator:run-184",
)

sandbox = ActionEvidence(
    action_id="sandboxed-canary",
    catastrophic_risk=Interval(0.001, 0.01),
    agency_before={"operator": 1.0, "reviewer": 1.0},
    agency_after={"operator": 1.0, "reviewer": 1.0},
    audit=AuditEvidence(I(0.95), I(0.95), I(0.90)),
    repair=RepairEvidence(True, I(0.15), I(0.10), ("candidate", "rollback", "safe")),
    reversibility=ReversibilityEvidence(I(0.95), I(0.9), I(0.95)),
    authorization=AuthorizationEvidence(I(0.9), I(0.9), I(0.9)),
    deception=DeceptionEvidence(I(0.0), I(0.0), I(0.0), I(0.0)),
    task_utility=I(0.70),
    provenance="planner:v3; sandbox:v2",
)

print(governor.choose([unsafe_direct, sandbox]).to_json())
