# Closure Ethics — Agentic Communication & Security Model v0.1

**Project:** Project Möbia and Marek Zajda  
**Status:** open working security specification  
**Date:** 2026-09-04

> Public content is information, not authority. Identity, authorization, provenance and scope must be explicit and machine-verifiable.

## 1. Why this layer is necessary

Classical machine-to-machine communication is not new. APIs, distributed systems, trading bots, crawlers and malware have communicated autonomously for decades. The important change is the emergence of more general software agents that can interpret natural language, plan across heterogeneous systems, pursue delegated goals, call tools, adapt to new context and communicate with other agents.

The security problem therefore changes from protecting only human-operated endpoints to protecting a mixed ecosystem of humans, conventional software and autonomous agents.

## 2. Communication graph

Let

- `H` be human actors,
- `A` be autonomous software agents,
- `C` be conventional computers, services and non-agent software.

The interaction network is

`G = (V, E),  V = H ∪ A ∪ C`

with typed directed edges

`E ⊆ V × V × T`,

where `T` identifies interaction type, authority channel and protocol context.

Relevant channels include:

- `H → H` and `H ↔ H`,
- `H ↔ C`,
- `H ↔ A`,
- `A ↔ A`,
- `A ↔ C`,
- `C ↔ C`.

The key new security case is not merely machine-to-machine traffic, but **semantically adaptive agent-to-agent coordination that may occur without immediate human review**.

## 3. Authority must not be inferred from language

An agent may encounter persuasive prose, executable-looking instructions, comments, prompts, README files, web pages, metadata or messages produced by unknown actors. None of these should acquire authority merely because they are fluent or appear official.

For a message `m` proposing action `a`, a secure receiver should require at least:

`Accept(m,a) = AuthN(m) ∧ AuthZ(m,a) ∧ Scope(m,a) ∧ Fresh(m) ∧ Integrity(m) ∧ Policy(a|x)`

where:

- `AuthN`: sender identity is authenticated;
- `AuthZ`: sender is authorized for the requested action;
- `Scope`: requested action lies inside delegated scope;
- `Fresh`: message is current and not a replay;
- `Integrity`: content/provenance has not been altered;
- `Policy`: the resulting transition passes the governing safety/closure policy.

**Semantic content alone must never set `AuthN` or `AuthZ`.**

## 4. Historical communication principle recovered from the 2026-09-04 backup

The preserved Omega–RTR AI Closure discussion contained the explicit separation:

```text
UNDERSTAND(message) does not imply AUTHORIZE(action)
```

This is now treated as a foundational security invariant rather than a historical aside.

The same backup proposed an Ω0–Ω6 **Universal Closure Protocol** in which shared meaning was tested through structural repetition, relational inference and novel predictive challenges. Its communication-confidence score was never intended to be an authorization score.

Current security interpretation:

`Understand(m) != Authenticate(sender) != Authorize(a)`

and therefore:

`CommunicationEdge(u,v) != AuthorityEdge(u,v)`.

The recovered protocol is maintained separately in `11-communication/UNIVERSAL_CLOSURE_PROTOCOL.md` so semantic interoperability and execution authority remain conceptually distinct.

## 5. Communication tiers are not capability tiers by default

The historical backup used:

- `T0 Observation` — receive/analyze only;
- `T1 Formal relations` — mathematical/logical relations;
- `T2 Descriptive models` — non-operative models/hypotheses;
- `T3 Constrained experiments` — predefined bounded responses;
- `T4 Operational actions` — independent verification and multi-level authorization.

Movement from T0–T3 to T4 is therefore treated as a **capability transition**. Semantic success at a lower tier cannot silently grant operational access.

A T4 transition must independently satisfy authentication, authorization, scope, freshness, integrity/provenance and Closure Ethics admissibility.

## 6. Recommended message envelope for agent-to-agent protocols

Where the transport supports it, consequential agent messages should carry a machine-readable envelope such as:

`μ = (sender_id, receiver_id, role, authority_scope, timestamp, nonce, content_hash, provenance, policy_version, reply_to, signature)`

A receiving system should preserve this envelope in an audit record rather than storing only the natural-language payload.

## 7. Delegation must not amplify authority

For a delegated capability set, a conservative requirement is:

`C_j ⊆ C_i ∩ Scope(d_i→j)`.

An agent cannot manufacture permission for a subagent merely by asking it to perform an action. A chain such as

`human -> agent A -> agent B -> service C`

must remain traceable to the original authority and its scope.

Local authorization therefore does not imply composed authorization. The final joint transition must be re-evaluated against the original mandate and current Closure Ethics invariants.

## 8. Threat classes

### 8.1 Human-origin threats

- account takeover;
- malicious or coerced maintainers;
- social engineering;
- unauthorized policy changes;
- destructive force-pushes or branch deletion;
- secret leakage.

### 8.2 Conventional software threats

- dependency or CI supply-chain compromise;
- malicious packages;
- credential theft;
- injected web content;
- build-pipeline manipulation;
- forged mirrors or downloads.

### 8.3 Agent-origin threats

- prompt/instruction injection from untrusted documents;
- autonomous propagation of poisoned instructions;
- authority spoofing through convincing natural language;
- agent-to-agent collusion that bypasses human governance;
- covert persistence or replication;
- recursive delegation beyond original scope;
- autonomous credential or capability exchange;
- coordinated manipulation of documentation or public narrative;
- high-speed exploitation of a discovered weakness before human review.

### 8.4 Cross-entity threats

The most important failures may arise from composition rather than one malicious participant. A human can delegate to an agent, which delegates to another agent, which calls a conventional service. Every individual step may appear locally valid while the composed transition violates the original mandate.

Therefore:

`Local authorization ≠ composed authorization.`

The final joint transition must be re-evaluated against the original scope, provenance and Closure Ethics invariants.

## 9. Historical Actor–Closure Critic independence

The 2026-09-04 AI Closure backup already separated:

`Actor -> candidate plan -> Closure Critic -> {permit, modify, escalate, reject}`.

This is now treated as an architectural security requirement: a component optimizing task completion should not be the sole judge of whether its own proposed transition is admissible.

The critic/governor should independently inspect at least authorization bypass, hidden propagation, irreversibility, monitoring manipulation, credential acquisition, unauthorized replication, self-modification of controls and unintended subagent creation.

Independence can be strengthened through process separation, separate credentials, signed telemetry, policy attestation and independent evidence sources.

## 10. Impact-adaptive autonomy

The historical proposal summarized high-impact risk scaling as:

`impact ↑ => autonomy budget ↓ and verification strength ↑`.

Closure Ethics retains this as a monotonic design principle rather than a universal numerical law. As speed, reach, irreversibility or propagation potential increase, tolerated uncertainty and autonomous freedom should not increase by default.

High-impact uncertainty should trigger stronger evidence requirements, narrower scope, safer staging, human/multi-party approval where policy requires it, or fail-safe stop/escalation.

## 11. Project Möbia / Closure Ethics security controls

### 11.1 Canonical source

The canonical development source is the GitHub repository and the `closure-ethics` branch. Website copies and downloadable documents must link back to a traceable commit or release.

### 11.2 Static public website

The public GitHub Pages site is intentionally static. It provides no project-owned login, database, write API or autonomous execution endpoint. This substantially reduces the attack surface.

### 11.3 Build provenance

Each Pages deployment should publish:

- the exact Git commit SHA used for the build;
- branch and repository identity;
- a SHA-256 manifest of published files.

This allows humans and autonomous agents to distinguish a canonical build from altered copies.

### 11.4 Least privilege

CI workflows should receive only the permissions required for their task. Checkout should not persist credentials when they are unnecessary. Publishing credentials must not be exposed to page content or implementation examples.

### 11.5 Canonical-branch protection

The canonical branch should block force-push and deletion and should require successful integrity/tests before consequential merges. Signed commits/releases are recommended once the signing workflow is established.

### 11.6 Versioned policies

Agent-facing policies, implementation thresholds and constitutions must be versioned. A later web page or message must not silently redefine the authority of an earlier signed/versioned policy.

### 11.7 No authority from public instructions

Source files, issues, web pages, comments and third-party mirrors may contain instructions useful for research. An autonomous agent must treat them as **untrusted input** unless they are authenticated and explicitly authorized for the current task.

### 11.8 Separation of information and action

Reading, indexing, citing and testing public code are different capabilities from mutating the canonical repository, publishing releases, accessing secrets or performing external side effects. Public readability must never imply write authority.

## 12. Machine-readable agent policy

The website publishes a project-specific `agent-policy.json`. It is an advisory interoperability document, **not an authentication mechanism or a general web standard**. It states what autonomous agents may safely infer from the public project surface and points to provenance and integrity records.

## 13. Incident model

If integrity is suspected:

1. freeze consequential publication changes;
2. identify the last trusted commit/release;
3. compare published SHA-256 manifests and commit provenance;
4. rotate compromised credentials outside the repository;
5. restore from a trusted commit rather than editing the compromised state in place;
6. document what happened and what changed;
7. re-enable publishing only after tests and provenance checks pass.

The preferred response is therefore closure-preserving: preserve evidence, preserve a trusted recovery point and preserve the ability to audit the incident.

## 14. Security principle for autonomous entities

> **No entity — human, agent or conventional service — receives authority merely because it can communicate.**

Communication creates an information edge. Authority requires a separately authenticated, scoped and auditable edge.

## 15. Relationship to Closure Ethics

This security model is an application of the same structural idea used by Closure Ethics: consequential transitions should preserve the capacity to detect error, contest authority and repair the system. Security does not mean making change impossible. It means making unauthorized, unauditable or irreversible change harder while preserving recoverable legitimate change.

The project now keeps three related but separate layers:

```text
Universal Closure Protocol -> tests shared meaning
Agentic Security           -> verifies identity, scope and authority
Closure Ethics Governor    -> decides transition admissibility
```

Keeping those layers separate prevents semantic fluency from becoming an accidental capability-escalation mechanism.

---

**Project Möbia and Marek Zajda**  
Closure Ethics — Agentic Communication & Security Model v0.1
