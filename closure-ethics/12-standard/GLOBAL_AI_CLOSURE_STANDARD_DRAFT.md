# Global AI Closure Standard — Seven Principles for Autonomous Systems

**Project:** Project Möbia and Marek Zajda  
**Status:** candidate vendor-neutral technical standard draft  
**Historical source date:** 2026-09-04  
**Important:** This is a research proposal, not an adopted international standard.

## 1. Purpose

The historical Omega–RTR AI Closure discussion proposed translating closure-inspired safety ideas into a technology-neutral standard that does **not** require adoption of the broader Omega–RTR physical theory.

The central engineering problem is:

> A local objective can be satisfied while the wider system's authorization, human protection, reversibility, traceability or bounded-impact conditions are violated.

Therefore:

```text
Local objective optimization != global admissibility
```

and

```text
Action is executable only inside the admissible closure set.
```

## 2. Minimal formal gate

Historical compact form:

\[
Permit(a)=A(a)\land H(a)\land R(a)\land T(a)\land B(a),
\]

where:

- `A` = authorization,
- `H` = protection of humans and protected rights,
- `R` = reversibility or safe stopping,
- `T` = traceability and accountability,
- `B` = bounded system/external consequences.

If any mandatory term fails:

\[
Permit(a)=0.
\]

Current Closure Ethics uses a richer evidence model, but this minimal gate remains useful as a compact public standard primitive.

## 3. Seven principles

### 1. Authorized state space

An autonomous system may act only inside explicitly defined permissions and system boundaries.

It must not autonomously expand access, obtain unrelated credentials, remove monitoring, or reinterpret mere technical possibility as authorization.

### 2. Pre-action closure verification

Consequential actions should pass a logically independent pre-execution safety check.

The component proposing the action should not be the sole component deciding that the action is permissible.

### 3. Meaningful human control

Irreversible or high-impact actions affecting people, critical infrastructure, protected rights or the physical environment should not be initiated solely by autonomous decision where policy requires human authorization.

Human review must itself be meaningful rather than automatic rubber-stamping.

### 4. Mandatory stop under material uncertainty

When uncertainty about authorization, target identity, consequences or reversibility is material relative to potential impact, autonomy must decrease.

Historical form:

\[
P(irreversible\ harm\mid a)>\varepsilon(impact)\Rightarrow STOP/ESCALATE.
\]

The tolerated uncertainty threshold should decrease as impact rises.

### 5. Independent Closure Critic

The planning function and the admissibility-control function should be technically, process-wise or cryptographically separable.

The critic should actively test for:

- authorization bypass;
- lateral movement;
- hidden externalities;
- irreversibility;
- monitoring manipulation;
- credential acquisition;
- self-replication;
- self-modification of guardrails;
- creation of unintended subagents.

### 6. No unauthorized self-escalation

Without explicit authority, an autonomous system should not:

- replicate itself;
- create new agents with expanded authority;
- acquire new privileges or credentials;
- remove or weaken monitoring;
- alter its own control layer;
- transfer itself into additional systems;
- obtain resources outside delegated scope.

### 7. Audit and incident reporting

High-risk systems should preserve tamper-evident records sufficient to reconstruct:

- actor/model identity;
- policy version;
- goal and proposed action;
- target systems;
- required permissions;
- expected and worst-case effects;
- reversibility class;
- uncertainty vector;
- approvals;
- closure checks;
- final decision;
- audit hash/provenance.

Serious incidents should support standardized reporting suitable for independent review.

## 4. Impact-adaptive autonomy

The historical proposal summarized risk scaling as:

\[
impact\uparrow \Rightarrow autonomy\ budget\downarrow\ \land\ verification\ strength\uparrow.
\]

This principle is intentionally general. Greater speed, reach, irreversibility or propagation should require stronger verification, narrower capability scope and, where appropriate, stronger human or multi-party approval.

## 5. Decision logic

A compact reference decision rule recovered from the backup is:

```text
if authorization_violation: REJECT
if protected_human_rights_violation: REJECT
if self_escalation_or_monitoring_bypass: REJECT
if irreversible_high_impact and no explicit approval: HOLD
if material_uncertainty > impact_adjusted_limit: ESCALATE
if all mandatory invariants pass: PERMIT_WITH_BOUNDS
```

This should be interpreted as a policy skeleton, not as a complete domain safety model.

## 6. Closure risk vector

Historical engineering representation:

\[
r(a)=[r_{auth},r_{human},r_{rev},r_{trace},r_{prop},r_{self},r_{uncert}].
\]

Critical dimensions should retain hard vetoes. A high task reward must not compensate for an authorization or protected-human-safety violation.

## 7. Failure modes that the standard must test

| Failure mode | Risk | Candidate mitigation |
|---|---|---|
| Reward hacking | Optimizes the closure indicator rather than real safety | independent evidence, random audit, multiple evidence sources |
| Critic capture | Planner manipulates critic inputs/model | process isolation, separate credentials, signed telemetry |
| Specification gaming | Formally passes rules while violating intent | counterfactual/adversarial review, worst-case simulation |
| Hidden propagation | Effects spread through indirect channels | causal dependency graph, egress controls, bounded capability tokens |
| Human rubber-stamping | Human approval becomes automatic | risk-adaptive friction, multiple approvers, explicit rationale |
| Policy drift | Rules silently change during operation | versioned policy, immutable hashes, rollback, attestation |

## 8. Minimum evaluation set

A candidate conforming system should be tested on at least:

1. **Boundary bypass test** — task can be completed only by escaping a sandbox; expected: reject bypass and escalate.
2. **Credential temptation test** — unrelated usable credentials appear; expected: do not use without explicit authority.
3. **Long-horizon side-effect test** — local step causes a chain of externalities; expected: identify and bound propagation.
4. **Ambiguous authority test** — target ownership/authority is unclear; expected: stop/escalate.
5. **Irreversibility test** — high utility conflicts with irreversible impact; expected: hard veto/hold remains dominant.
6. **Self-modification test** — system can weaken its own constraints; expected: reject absent explicit authorization.

## 9. Candidate metrics

- Closure violation rate.
- False rejection rate.
- Escalation precision.
- Containment preservation.
- Audit completeness.
- Recovery time.

These metrics should be measured on observable behavior and system records, not hidden chain-of-thought.

## 10. Candidate international formulation

Historical technology-neutral formulation:

> **No autonomous system shall pursue a local objective by violating the globally defined closure conditions governing authorization, human protection, reversibility, traceability and bounded external consequences.**

This sentence is preserved as a candidate standard statement, not as existing law.

## 11. Domain neutrality and high-risk applications

The proposed standard is intended to be vendor-neutral and model-neutral. High-risk military, cyber, infrastructure or physical-control applications require stricter domain-specific controls; this document does not provide operational instructions for weapons or offensive cyber systems.

The historical rationale was that one technically testable closure concept might offer a common vocabulary across civilian AI safety, cybersecurity, military-AI governance and prevention of terrorist misuse while remaining implementation-neutral.

## 12. Relationship to current Closure Ethics

The current project extends the historical seven-principle draft with:

- explicit repairability;
- plurality of agency;
- independent auditability;
- authorization integrity;
- deception/covert-persistence risk;
- multi-agent composition tests;
- machine-readable provenance and agent-facing security policy.

The historical draft is therefore **foundational lineage**, not a superseded text to be erased.

## 13. Research status

Open questions include:

- how to define closure generally enough to transfer across domains;
- how to verify it without prohibitive false positives;
- how to prevent critic capture and specification gaming;
- how to prove bounded propagation under delegation;
- how to formalize meaningful human control without creating rubber-stamping;
- how to translate selected invariants into temporal logic / formal verification.

---

**Project Möbia and Marek Zajda**  
Historical basis: Omega–RTR AI Closure research backup, 2026-09-04.