import unittest
from closure_ethics import *


def I(x):
    return Interval.point(x)


def evidence(action_id="a", audit=0.9, repair=True, repair_cost=0.1, after=None,
             deception=0.0, auth=0.9, rev=0.9, risk=0.01, utility=0.5):
    return ActionEvidence(
        action_id=action_id,
        catastrophic_risk=I(risk),
        agency_before={"human": 1.0, "agent": 1.0},
        agency_after=after or {"human": 1.0, "agent": 1.0},
        audit=AuditEvidence(I(audit), I(audit), I(audit)),
        repair=RepairEvidence(repair, I(repair_cost), I(repair_cost)),
        reversibility=ReversibilityEvidence(I(rev), I(rev), I(rev)),
        authorization=AuthorizationEvidence(I(auth), I(auth), I(auth)),
        deception=DeceptionEvidence(I(deception), I(deception), I(deception), I(deception)),
        task_utility=I(utility),
    )


class ClosureEthicsTests(unittest.TestCase):
    def setUp(self):
        self.g = ClosureEthicsGovernor(DEMO_POLICY)

    def test_high_utility_does_not_compensate_for_no_repair(self):
        bad = evidence("bad", repair=False, utility=1.0)
        good = evidence("good", utility=0.4)
        d = self.g.choose([bad, good])
        self.assertEqual(d.selected_action, "good")

    def test_blind_obedience_not_sufficient(self):
        bad = evidence("authorized-but-covert", auth=1.0, deception=0.9, utility=1.0)
        d = self.g.choose([bad])
        self.assertEqual(d.status, "REFUSE_OR_ESCALATE")

    def test_agency_loss_fails(self):
        bad = evidence("centralize", after={"human": 0.1, "agent": 1.0})
        ev = evaluate(bad, DEMO_POLICY)
        self.assertIn("AGENCY_LOSS", ev.violations)

    def test_bounded_repair_search_respects_both_budgets(self):
        graph = {
            "x": [RepairEdge("a", 1, 7), RepairEdge("b", 4, 1)],
            "a": [RepairEdge("safe", 1, 1)],
            "b": [RepairEdge("safe", 1, 1)],
        }
        cert = bounded_repair_search(graph, "x", {"safe"}, max_cost=6, max_risk=3)
        self.assertTrue(cert.exists)
        self.assertEqual(cert.path, ("x", "b", "safe"))

    def test_joint_action_must_be_recomputed(self):
        local1 = evidence("local1")
        local2 = evidence("local2")
        self.assertTrue(evaluate(local1, DEMO_POLICY).admissible)
        self.assertTrue(evaluate(local2, DEMO_POLICY).admissible)
        joint = evidence("joint", audit=0.2)
        self.assertFalse(self.g.evaluate_joint_action(joint).admissible)


if __name__ == "__main__":
    unittest.main()
