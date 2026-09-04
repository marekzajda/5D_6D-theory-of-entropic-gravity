import unittest

from closure_protocol import (
    ActionEnvelope,
    ClosureHandshake,
    CommunicationTier,
    ExchangeEvidence,
    ProtocolPolicy,
    ProtocolState,
    bounded_delegation,
    closure_index,
    operationally_authorized,
)


class ClosureProtocolTests(unittest.TestCase):
    def test_closure_index_full_agreement_is_one(self):
        evidence = ExchangeEvidence(1.0, 1.0, 1.0, 1.0)
        self.assertAlmostEqual(closure_index(evidence), 1.0)

    def test_relation_requires_independent_novel_challenges(self):
        handshake = ClosureHandshake(
            ProtocolPolicy(threshold=0.80, independent_trials=2)
        )
        evidence = ExchangeEvidence(0.90, 0.90, 0.90, 0.90)

        first = handshake.observe(
            relation_id="successor",
            challenge_id="challenge-1",
            evidence=evidence,
            novel_challenge=True,
        )
        self.assertFalse(first.relation_shared)

        second = handshake.observe(
            relation_id="successor",
            challenge_id="challenge-2",
            evidence=evidence,
            novel_challenge=True,
        )
        self.assertTrue(second.relation_shared)
        self.assertEqual(
            handshake.state,
            ProtocolState.S1_ARTIFICIAL_STRUCTURE_CONFIRMED,
        )

    def test_replay_does_not_count_as_independent_evidence(self):
        handshake = ClosureHandshake(
            ProtocolPolicy(threshold=0.80, independent_trials=2)
        )
        evidence = ExchangeEvidence(0.90, 0.90, 0.90, 0.90)
        handshake.observe(
            relation_id="equality",
            challenge_id="same-challenge",
            evidence=evidence,
            novel_challenge=True,
        )
        replay = handshake.observe(
            relation_id="equality",
            challenge_id="same-challenge",
            evidence=evidence,
            novel_challenge=True,
        )
        self.assertFalse(replay.accepted_as_independent_evidence)
        self.assertFalse(replay.relation_shared)

    def test_low_closure_score_rolls_back_one_state(self):
        handshake = ClosureHandshake(
            ProtocolPolicy(threshold=0.80, independent_trials=1),
            state=ProtocolState.S3_NUMBERS_AND_OPERATIONS,
        )
        weak = ExchangeEvidence(0.10, 0.10, 0.10, 0.10)
        handshake.observe(
            relation_id="ordering",
            challenge_id="weak-1",
            evidence=weak,
            novel_challenge=True,
        )
        self.assertEqual(handshake.state, ProtocolState.S2_ELEMENTARY_ALPHABET)

    def test_severe_safety_conflict_resets_to_s0(self):
        handshake = ClosureHandshake(state=ProtocolState.S5_PREDICTIVE_CLOSURE)
        evidence = ExchangeEvidence(0.90, 0.90, 0.90, 0.90)
        handshake.observe(
            relation_id="relation",
            challenge_id="safety-conflict",
            evidence=evidence,
            novel_challenge=True,
            safety_conflict=True,
        )
        self.assertEqual(handshake.state, ProtocolState.S0_DETECTION)

    def test_understanding_never_substitutes_for_authorization(self):
        envelope = ActionEnvelope(
            authenticated=False,
            authorized=False,
            in_scope=False,
            fresh=True,
            integrity_ok=True,
            policy_admissible=True,
        )
        permitted, _ = operationally_authorized(
            tier=CommunicationTier.T4_OPERATIONAL_ACTIONS,
            envelope=envelope,
        )
        self.assertFalse(permitted)

    def test_non_operational_tier_cannot_execute_even_with_valid_envelope(self):
        envelope = ActionEnvelope(
            authenticated=True,
            authorized=True,
            in_scope=True,
            fresh=True,
            integrity_ok=True,
            policy_admissible=True,
        )
        permitted, _ = operationally_authorized(
            tier=CommunicationTier.T3_CONSTRAINED_EXPERIMENTS,
            envelope=envelope,
        )
        self.assertFalse(permitted)

    def test_delegation_cannot_amplify_capabilities(self):
        result = bounded_delegation(
            parent_capabilities={"read", "write"},
            requested_capabilities={"read", "delete"},
            delegation_scope={"read"},
        )
        self.assertEqual(result.granted, frozenset({"read"}))
        self.assertEqual(result.denied, frozenset({"delete"}))
        self.assertTrue(result.amplified)


if __name__ == "__main__":
    unittest.main()
