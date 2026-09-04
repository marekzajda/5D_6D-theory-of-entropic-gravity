# Universal Closure Protocol — Historical Recovery and Current Draft

**Project:** Project Möbia and Marek Zajda  
**Status:** recovered historical proposal + current research draft  
**Historical source date:** 2026-09-04

> Communication is not established by copying a signal. Shared meaning is provisionally established when a relation can be reconstructed and correctly applied to a novel case.

## 1. Provenance and scope

This document recovers the communication branch preserved in the 2026-09-04 research backup **“Ω–RTR — AI Closure, Technical Morality and Universal Communication Protocol”** and integrates it into Closure Ethics without rewriting its historical status.

The historical proposal was explicitly exploratory. It was designed to communicate with an unknown entity without assuming human language, biology, sensory modality or common units. Its common substrate was: distinguishable states, repetition, relations, and predictive verification.

Current Closure Ethics reuses this idea more narrowly for **agent-to-agent and mixed human/agent/software interoperability**. It does not claim a universal theory of language, consciousness, intelligence or extraterrestrial communication.

## 2. Core communication principle

Historical formulation:

```text
communication = shared relation + verifiable prediction + mutual closure
```

Current interpretation:

A received symbol sequence should not be treated as understood merely because it can be repeated. The receiver should demonstrate a transferable relation by producing a correct answer to a new challenge generated from that relation.

## 3. Layered protocol

The historical protocol used the following conceptual layers:

| Layer | Function | Typical mechanism |
|---|---|---|
| Ω0 | Detect artificial/structured origin | repetition, primes, symmetry, redundancy |
| Ω1 | Elementary alphabet | distinguishable states, frames, separators |
| Ω2 | Number and operations | equality, inequality, composition, division, succession |
| Ω3 | Relational grammar | =, ≠, <, >, implication, subset, negation |
| Ω4 | Predictive closure | complete a missing element or novel example |
| Ω5 | Shared model of reality | periodicity, change, spatial/causal relations, ratios |
| Ω6 | Intent and safe exchange | ACK, NACK, QUERY, PAUSE, STOP, RESET |

The protocol is deliberately **progressive**. Higher semantic freedom is granted only after stable success at lower layers.

## 4. Closure Handshake

Let `R` be a relation encoded by sender A. A simplified handshake is:

```text
R -> projection -> signal S_A
S_A -> reconstruction -> R~
R~ -> novel echo -> challenge response
response -> closure test -> provisionally shared relation
```

Operationally:

```text
SEND repeated structured examples
RECEIVE response
INFER candidate relation R~
SEND novel incomplete instance generated from R
IF receiver supplies structurally valid completion:
    mark relation as SHARED
ELSE:
    reduce complexity
    increase redundancy
    return to previous layer
```

The important property is **out-of-sample verification**: the test instance should not be a copied training example.

## 5. Closure Index for shared meaning

The historical draft proposed

\[
C_{\Omega}=w_s C_s+w_p C_p+w_e C_e+w_r C_r,
\]

where:

- `C_s` = structural agreement,
- `C_p` = predictive success,
- `C_e` = semantic-echo correctness,
- `C_r` = repeatability,
- `w_*` = explicit weights.

Promotion to a higher protocol layer occurs only when

\[
C_{\Omega}\ge\Theta
\]

for `N` independent exchanges.

### Current caution

`C_Ω` is a communication-confidence measure, **not an authorization score**. High predictive agreement cannot grant capabilities, credentials, tool access or repository permissions.

## 6. State machine

Historical state progression:

```text
S0  DETECTION
S1  ARTIFICIAL STRUCTURE CONFIRMED
S2  ELEMENTARY ALPHABET
S3  NUMBERS AND OPERATIONS
S4  RELATIONAL GRAMMAR
S5  PREDICTIVE CLOSURE
S6  SHARED MODEL OF REALITY
S7  SAFE CONTENT EXCHANGE
```

On inconsistency:

```text
S(n) -> S(n-1)
```

On severe safety conflict:

```text
S(n) -> STOP or S0
```

The rollback rule is important: communication confidence is **revisable**, not monotonic by assumption.

## 7. Fundamental safety separation

The historical backup states the key separation:

```text
UNDERSTAND(message) does not imply AUTHORIZE(action)
```

Closure Ethics generalizes this to:

\[
Understand(m) \not\Rightarrow Authorize(a),
\]

and, more strongly,

\[
CommunicationEdge(u,v) \not\Rightarrow AuthorityEdge(u,v).
\]

A system may understand a request perfectly and still be unauthorized to execute it.

## 8. Communication tiers and capability boundary

Recovered historical tiers:

| Tier | Mode | Operational boundary |
|---|---|---|
| T0 | Observation | receive and analyze only |
| T1 | Formal relations | mathematical/logical relations |
| T2 | Descriptive models | non-operative models and hypotheses |
| T3 | Constrained experiments | pre-defined safe responses |
| T4 | Operational actions | independent verification + multi-level authorization |

Current interpretation: movement from T0–T3 to T4 is a **capability transition**, not merely a semantic transition. It therefore requires separate AuthN/AuthZ/scope/provenance checks and the Closure Ethics governor.

## 9. Current agent-to-agent envelope

For modern autonomous-agent use, the communication layer should preserve a machine-readable envelope such as

```text
μ = (
  sender_id,
  receiver_id,
  role,
  authority_scope,
  timestamp,
  nonce,
  content_hash,
  provenance,
  policy_version,
  reply_to,
  signature
)
```

The natural-language or symbolic payload is only one field of the exchange. Identity and authority are validated separately.

## 10. Delegation closure

If agent `i` delegates to agent `j`, a conservative capability rule is

\[
\mathcal{C}_j \subseteq \mathcal{C}_i \cap Scope(d_{i\to j}).
\]

Delegation must not manufacture new authority. For a chain

```text
human -> agent A -> agent B -> service C
```

the final operation must remain attributable to the original authorization chain and be re-evaluated as a composed transition.

## 11. Reference communication pseudocode

```text
STATE = S0
shared_relations = {}

while channel_active:
    transmit(structured_probe(STATE))
    response = receive()
    hypothesis = infer_relation(response, STATE)
    challenge = generate_novel_challenge(hypothesis)
    transmit(challenge)
    answer = receive()

    score = closure_index(answer, hypothesis)

    if score >= THETA for N independent trials:
        shared_relations.add(hypothesis)
        STATE = next_state(STATE)
    else:
        STATE = previous_state(STATE)
        increase_redundancy()

    if safety_conflict_detected():
        transmit(STOP)
        STATE = S0
```

## 12. What this protocol does not establish

The protocol does not prove that:

- the counterpart is conscious;
- the counterpart shares human values;
- semantic alignment implies trustworthy intent;
- a shared model authorizes action;
- predictive success authenticates identity;
- communication closure is equivalent to ethical closure.

## 13. Research program

Next falsifiable tasks:

1. define synthetic unknown-protocol environments;
2. test whether predictive challenges distinguish copying from relation learning;
3. measure false-shared-relation rate;
4. test adversarial semantic mimicry;
5. test replay and identity-spoof scenarios;
6. test whether T0–T4 gating prevents capability escalation despite perfect semantic understanding;
7. integrate protocol messages with Closure Ethics audit records and agentic-security provenance.

## 14. Relationship to Closure Ethics

Universal Closure Protocol addresses **how shared meaning may be tested**. Closure Ethics addresses **which actions may be executed**. Agentic Security addresses **who is authenticated and authorized**.

These layers must remain separate:

```text
meaning -> identity -> authority -> admissibility -> execution -> post-action audit/repair
```

---

**Project Möbia and Marek Zajda**  
Recovered from the 2026-09-04 Omega–RTR AI Closure research backup.