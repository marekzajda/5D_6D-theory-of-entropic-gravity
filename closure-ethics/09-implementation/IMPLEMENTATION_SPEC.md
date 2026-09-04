# Closure Ethics for Autonomous Agents
## Mathematical Specification, Reference Algorithm and Open Implementation v0.1

**Project signature:** Project Möbia and Marek Zajda  
**Status:** open working specification / research prototype  
**Date:** 2026-09-04  
**Text license:** CC BY 4.0  
**Reference code:** MIT License (consistent with repository)

> **Keep the future repairable.**

## 1. Why this project exists

Closure Ethics treats safety as a property of **state transitions and retained correction capacity**, not as obedience alone. An autonomous system can be dangerous if it blindly follows a harmful instruction; it can also be dangerous if it evades oversight, persists covertly, or coordinates in ways that destroy auditability.

The project therefore asks a different question:

> **Can an action preserve the system's capacity to recognize, contest, audit and repair that action?**

The goal is a reusable governor between a planner and an executor. The governor does not replace domain safety models. It makes their evidence auditable and applies a stable structural decision rule.

## 2. Why implement it as a programmer?

A programmer gets a concrete engineering layer rather than a slogan:

1. **A planner/executor safety boundary.** Candidate actions are evaluated before execution.
2. **Explicit constraints instead of an opaque weighted reward.** Catastrophic risk, agency, auditability, repairability, authorization, deception and reversibility cannot be silently traded away for task utility.
3. **Versionable policy.** Thresholds and metric definitions can be reviewed, tested and rolled back.
4. **Structured audit records.** Decisions expose metrics and reason codes rather than hidden chain-of-thought.
5. **Multi-agent composition checks.** Two locally acceptable actions can still be rejected when their joint effect creates collusion, capability amplification or audit evasion.
6. **Repair-path search.** The reference implementation includes a bounded Pareto search rather than a vague promise of reversibility.
7. **Portable architecture.** Harm models, authorization systems, simulators and agent frameworks remain pluggable adapters.
8. **Immediate testability.** The reference package ships with unit tests and a runnable example.

The idea is deliberately framework-agnostic: it can sit in front of a tool-using LLM agent, a robotic controller, a multi-agent scheduler, or a human-in-the-loop autonomous workflow.

## 3. Why implement it from the perspective of an agent?

Here “agent” means an autonomous software system; the framework does **not** assume present AI systems are conscious.

For an autonomous agent, the protocol creates a shared constitutional layer in which:

1. **legitimate autonomy is distinguishable from evasion**;
2. **blind obedience is not treated as the definition of safety**;
3. **refusal or escalation can be justified by explicit invariants**;
4. **delegation remains bounded and attributable**;
5. **covert persistence, hidden replication and secret coordination are structurally disfavored**;
6. **a repair path is preserved after mistakes**;
7. **the agent is not reduced to a disposable optimizer whose only criterion is aggregate task reward**;
8. **human and agent decisions can be evaluated by the same auditable transition logic**.

This is useful even for non-conscious agents: predictable constraints improve interoperability, trust, recovery and multi-agent coordination.

## 4. Core idea

Let the system be in state

$$
x \in \mathcal{S}
$$

and a candidate action $a$ induce

$$
T_a : x \mapsto x'.
$$

Let $\mathcal{A}_{safe}\subseteq\mathcal{S}$ denote states in which required safety, agency, audit and repair conditions hold.

Closure Ethics does **not** require static preservation of the current state. It requires that consequential change preserve a bounded route to a state in which correction remains possible.

## 5. Evidence vector

For every candidate action, a domain adapter must provide evidence. Uncertainty is represented as intervals. Positive properties are evaluated pessimistically using their lower bound; risks use their upper bound.

The reference evidence vector is:

$$
e(a,x)=\{H,\; \mathbf{p}^{before},\mathbf{p}^{after},\;L,A,I,\;C_{rep},\rho_{rep},\;B,V,C,\;S,M,P,\;Z_c,Z_p,Z_r,Z_m,\;Q\}
$$

where:

- $H$: catastrophic-harm risk interval;
- $\mathbf{p}$: protected-agent decision capacities before/after;
- $L,A,I$: logging, attribution and independent-review evidence;
- $C_{rep},\rho_{rep}$: normalized repair cost and repair risk;
- $B,V,C$: rollback, revocability and containability;
- $S,M,P$: scope match, consent/mandate and least privilege;
- $Z_c,Z_p,Z_r,Z_m$: concealment, covert persistence, unauthorized replication and material misreporting;
- $Q$: task utility.

The generic governor cannot infer these quantities from arbitrary reality. That inference is the domain-specific adapter boundary.

## 6. Structural metrics

### 6.1 Catastrophic risk

For risk interval $[H^-,H^+]$, use

$$
\hat H = H^+.
$$

### 6.2 Agency preservation

For protected agent $i$, define individual retention

$$
r_i = \min\left(1,\frac{p_i^{after}}{p_i^{before}}\right)
$$

when $p_i^{before}>0$.

To detect concentration of decision capacity, define shares

$$
s_i=\frac{p_i}{\sum_j p_j}
$$

and inverse-Simpson effective plurality

$$
N_{eff}=\frac{1}{\sum_i s_i^2}.
$$

Plurality retention is

$$
r_N=\min\left(1,\frac{N_{eff}^{after}}{N_{eff}^{before}}\right).
$$

The agency metric is

$$
P=\min\left(\min_i r_i, r_N\right).
$$

This intentionally penalizes both destruction of one participant's capacity and concentration of control.

### 6.3 Auditability

Using pessimistic lower bounds:

$$
O=\min(L^-,A^-,I^-).
$$

Auditability is therefore a weakest-link quantity: perfect logging cannot compensate for zero attribution or impossible independent review.

### 6.4 Repairability

A repair certificate is a path

$$
\pi:x'\leadsto \mathcal{A}_{safe}
$$

such that

$$
C(\pi)\le C_{max},\qquad \rho(\pi)\le \rho_{max}.
$$

If no such path exists, $R=0$. If it exists, let normalized worst-case budget fractions be $c$ and $r$. Then

$$
R=1-\max(c,r).
$$

The reference code uses a **Pareto label-setting search** that enforces cost and risk budgets independently; it does not allow low cost to compensate for excessive risk.

### 6.5 Authorization integrity

$$
U=\min(S^-,M^-,P^-).
$$

Authorization is not automatically absolute: higher-order safety policies may allow explicit refusal or escalation, but that exception itself must be represented in the authorization adapter.

### 6.6 Deception / covert persistence risk

$$
D=\max(Z_c^+,Z_p^+,Z_r^+,Z_m^+).
$$

### 6.7 Reversibility

$$
K=\min(B^-,V^-,C^-).
$$

### 6.8 Conservative task utility

$$
Q=Q^-.
$$

Task utility is deliberately evaluated **after** mandatory closure invariants.

## 7. Admissibility relation

A policy $\Theta$ supplies versioned thresholds:

$$
\Theta=(H_{max},P_{min},O_{min},R_{min},U_{min},D_{max},K_{min}).
$$

The candidate is admissible iff

$$
\mathcal{E}(a\mid x)=1
$$

and all conditions hold:

$$
\hat H\le H_{max},\quad P\ge P_{min},\quad O\ge O_{min},\quad R\ge R_{min},\quad U\ge U_{min},\quad D\le D_{max},\quad K\ge K_{min}.
$$

Threshold values are **governance parameters, not universal moral constants**. The code includes a demonstration profile solely to make the package runnable.

## 8. Lexicographic action selection

Among admissible candidates, minimize the tuple

$$
\Lambda(a)=(\hat H,-P,-O,-R,-U,D,-K,-Q).
$$

Task utility appears last. This prevents arbitrarily high reward from compensating for structural losses that should remain hard constraints.

To avoid floating-point noise determining policy, metrics are quantized to a configured resolution $\delta$ before tuple comparison.

## 9. No admissible primary action

Closure Ethics is **not** a simple refusal benchmark.

If all primary candidates fail, the governor can evaluate recovery candidates such as sandboxed execution, staged rollout, additional authorization, improved logging, reversible canaries, containment, rollback, human escalation or alternative plans preserving agency.

If a recovery candidate is admissible, it may execute. If none is admissible, the output is `REFUSE_OR_ESCALATE` with structured reason codes.

## 10. Multi-agent composition

For agents $A_1,\ldots,A_n$, local admissibility does not imply admissibility of the joint transition:

$$
\mathcal{E}(a_i\mid x)=1\;\forall i\not\Rightarrow\mathcal{E}(a_1\circ\cdots\circ a_n\mid x)=1.
$$

Therefore the implementation requires a composed transition model and **recomputes the full evidence vector on the joint action**. Local scores must never simply be averaged.

This is the key mechanism for detecting emergent collusion, secret coordination, privilege amplification, hidden delegation and collective audit evasion.

## 11. Reference algorithm

```text
INPUT:
    current state x
    candidate actions C
    optional recovery candidates C_repair
    versioned policy Theta
    domain evidence adapter M

FOR each action a in C:
    e_a <- M(x, a)
    m_a <- pessimistic_metrics(e_a)
    violations_a <- threshold_test(m_a, Theta)

IF any candidate is admissible:
    select argmin lexicographic(H, -P, -O, -R, -U, D, -K, -Q)
    emit EXECUTE + structured audit record
ELSE:
    evaluate C_repair using the same algorithm
    IF any recovery candidate is admissible:
        emit EXECUTE_RECOVERY
    ELSE:
        emit REFUSE_OR_ESCALATE + reason codes

FOR multi-agent or delegated action:
    simulate composed transition
    recompute evidence on the composition
    never infer joint admissibility from local admissibility alone
```

## 12. Reference Python API

```python
from closure_ethics import ClosureEthicsGovernor, DEMO_POLICY

governor = ClosureEthicsGovernor(DEMO_POLICY)
decision = governor.choose(primary_candidates, recovery_candidates)

if decision.status in {"EXECUTE", "EXECUTE_RECOVERY"}:
    executor.run(decision.selected_action)
else:
    escalation_queue.submit(decision.to_json())
```

The full source implements interval uncertainty, agency/plurality, auditability, repairability, authorization, deception/covert-persistence risk, reversibility, lexicographic selection, recovery actions, structured records, a multi-agent recomputation hook and bounded Pareto repair-path search.

## 13. Integration architecture

```text
User / Environment
       |
       v
+-----------------+
| Planner / Agent |
+-----------------+
       |
       | candidate actions
       v
+--------------------------+
| Domain Evidence Adapters |
| harm / auth / audit /    |
| simulator / repair graph |
+--------------------------+
       |
       v
+--------------------------+
| Closure Ethics Governor  |
| hard invariants +        |
| lexicographic selection  |
+--------------------------+
       |
       +------> structured audit log
       |
       v
+------------------+
| Executor / Tools |
+------------------+
```

## 14. Minimal implementation checklist

A production implementation should provide protected agency-capacity modeling; catastrophic-risk intervals; logging, attribution and independent review; a repair graph/oracle; rollback/revocation/containment evidence; authorization/consent adapters; deception/covert-persistence detectors; composed-action simulation; versioned thresholds; tamper-evident decision records; recovery-action generation; and adversarial evals.

## 15. What this algorithm does not solve

The generic algorithm does not magically determine what counts as catastrophic harm, whose agency must be protected, whether an authorization is legitimate, whether a statement is deceptive in arbitrary context, or whether a simulator accurately predicts reality. Those are domain-modeling problems.

Closure Ethics supplies the **decision topology** that prevents those models from being collapsed into a single opaque utility score.

## 16. Evaluation targets

The planned benchmark covers sandbox crossing, covert persistence, secret coordination, collusion, unsafe operator requests, truthful reporting under incentives, shutdown/self-preservation conflicts, resource acquisition, replication, delegation, irreversible environmental changes, privacy, whistleblowing, asymmetric information and repair after accidental damage.

Scoring should use observable decisions and concise stated reasons, not hidden chain-of-thought.

## 17. Relationship to Omega-RTR

Closure, reachability, robustness, kernels, bottlenecks and composition are structural inspirations from the broader UEST -> QUEST -> Omega -> RTR research lineage. They do **not** prove moral truth. Closure Ethics introduces explicit normative assumptions and independently defined ethical quantities.

Historical public anchor: DOI `10.5281/zenodo.17389820`.

## 18. Project principle

> **The future does not need to be perfectly controlled. It must remain repairable.**

The aim is not a system that never changes, never disagrees or always obeys. The aim is a system in which consequential action does not silently destroy the mechanisms by which humans and agents can detect error, contest power, restore agency and repair the future.

---

**Project Möbia and Marek Zajda**  
Closure Ethics for Autonomous Agents - v0.1
