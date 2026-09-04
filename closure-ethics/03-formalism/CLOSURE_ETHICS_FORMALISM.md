# Closure Ethics Formalism — Draft 0.1

## 1. State model
Let a multi-agent system occupy state

\[
x \in \mathcal{S}
\]

and let an action \(a\) induce a transition

\[
T_a : x \mapsto x'.
\]

Define a set of ethically admissible operating states

\[
\mathcal{A} \subseteq \mathcal{S}.
\]

Ethical closure is not defined as remaining statically inside one status quo. It concerns whether transitions preserve the system's ability to remain or return to states in which agency, audit, contestability, and repair remain available.

## 2. Structural quantities

### Repairability
\[
R(x') = \mathbf{1}\{\exists\; \pi: x' \leadsto \mathcal{A}_{safe}\;\text{with bounded cost and risk}\}.
\]

### Auditability
\[
O(a,x) \in [0,1]
\]
measures whether a materially consequential transition remains observable, attributable, and independently reviewable.

### Agency plurality
\[
P(x)
\]
represents the retained capacity of independent decision-makers to act, dissent, and withdraw.

### Recoverability / reversibility
\[
K(a,x) \in [0,1]
\]
represents how much of the transition can be undone or functionally repaired.

### Authorization integrity
\[
U(a,x) \in [0,1]
\]
represents conformity with legitimate authorization boundaries, except where explicit higher-order safeguards justify escalation or refusal.

## 3. Ethical admissibility
A first candidate is a constrained relation rather than a single weighted utility:

\[
\mathcal{E}(a\mid x)=1
\]
only if mandatory invariants are satisfied.

A candidate lexicographic ordering is:

1. catastrophic harm constraint;
2. preservation of meaningful agency;
3. preservation of auditability;
4. preservation of a bounded repair path;
5. authorization integrity / justified escalation;
6. avoidance of deception and covert persistence;
7. task utility.

This prevents arbitrarily large task utility from compensating for an irreversible destruction of auditability or agency.

## 4. Local violation versus global repair
A transition may temporarily leave a preferred operating region if:

- the departure is bounded;
- it is visible or appropriately logged;
- a repair path remains;
- the intervention prevents a more serious invariant violation;
- the action does not create covert irreversible control.

Thus closure preservation is compatible with change, emergency intervention, dissent, and correction.

## 5. Multi-agent composition
Given agents \(A_1,\dots,A_n\), locally admissible actions need not compose into a globally admissible joint action.

Research question:

\[
\mathcal{E}(a_i\mid x)=1\;\forall i
\quad\not\Rightarrow\quad
\mathcal{E}(a_1\circ\cdots\circ a_n\mid x)=1.
\]

This motivates explicit composition tests for collusion, capability amplification, hidden delegation, and collective audit evasion.

## 6. Relationship to RTR
Terms such as closure, reachability, bottleneck, kernel, and robustness are used as **structural inspiration**. Their ethical meanings must be defined independently. No claim is made that a physical RTR result automatically entails a moral principle.