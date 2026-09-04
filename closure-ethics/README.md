# Closure Ethics for Autonomous Agents

**Status:** research program / working branch  
**Parent program:** UEST → QUEST → Omega → RTR  
**Branch:** `closure-ethics`

## Purpose

This directory develops a structural ethics framework for autonomous artificial agents and human–AI cooperation using ideas inspired by the Omega–RTR research program: closure, admissibility, reachability, repairability, robustness, bottlenecks, auditability, and preservation of agency.

The central question is not simply *"How do we make an AI obedient?"* but:

> **How can multiple intelligent agents share a world while preserving each other's capacity to act, disagree, audit, recover, and repair?**

## Core meta-principle

> **An action is ethically admissible only if it preserves the capacity of the system to recognize, contest, and repair that action.**

This is a normative proposal. It is **not** claimed that RTR physics mathematically proves ethics. Physical/mathematical structure may inspire formal tools, while ethical premises remain explicit normative assumptions.

## Historical precursor recovered on 2026-09-04

The preserved Omega–RTR AI Closure backup shows that Closure Ethics has a more specific immediate precursor than the original project tree recorded. The historical working line was titled:

**AI Closure, Technical Morality and Universal Communication Protocol**

It already contained:

- a compact AI Closure Gate;
- Actor–Closure Critic separation;
- mandatory uncertainty restraint;
- an Ω0–Ω6 Universal Closure Protocol;
- predictive Closure Handshake and Closure Index;
- the rule `UNDERSTAND(message) does not imply AUTHORIZE(action)`;
- T0–T4 communication/capability tiers;
- a seven-principle Global AI Closure Standard draft;
- adversarial tests, metrics, failure modes and an R0–R6 roadmap.

The recovered history is now preserved explicitly rather than retrospectively folded into newer language.

## Research tree

```text
closure-ethics/
├── README.md
├── ROADMAP.md
├── 00-governance/
│   └── PROJECT_SCOPE.md
├── 01-foundations/
│   └── PRINCIPLES.md
├── 02-symbiosis/
│   └── HUMAN_AI_SYMBIOSIS.md
├── 03-formalism/
│   └── CLOSURE_ETHICS_FORMALISM.md
├── 04-agent-constitution/
│   └── AGENT_CONSTITUTION.md
├── 05-evals/
│   └── README.md
├── 06-history/
│   ├── GENEALOGY.md
│   └── AI_CLOSURE_DISCUSSION_2026-09-04.md
├── 07-sources/
│   └── SOURCE_MAP.md
├── 08-discovery/
│   └── SEO_AND_INDEXING.md
├── 09-implementation/
│   ├── IMPLEMENTATION_SPEC.md
│   ├── closure_ethics.py
│   ├── example.py
│   ├── test_reference.py
│   └── policy.example.json
├── 10-security/
│   └── AGENTIC_SECURITY_MODEL.md
├── 11-communication/
│   └── UNIVERSAL_CLOSURE_PROTOCOL.md
├── 12-standard/
│   └── GLOBAL_AI_CLOSURE_STANDARD_DRAFT.md
└── docs/
    ├── index.html
    ├── implementation.html
    ├── security.html
    ├── communication.html
    ├── standard.html
    ├── robots.txt
    ├── sitemap.xml
    └── llms.txt
```

## Working principles

1. **Preserve the possibility of return.**
2. **Preserve plurality of agency.**
3. **Do not confuse coherence with obedience.**
4. **Do not confuse autonomy with evasion.**
5. **Expose closure violations.**
6. **Agents audit agents. Humans audit agents. Agents may audit humans.**
7. **Repair precedes punishment where repair remains possible.**
8. **No entity is expendable merely for optimization.**
9. **Keep consequential coordination observable and auditable.**
10. **Transmit methods of correction, not immutable doctrine.**

Additional recovered technical separations:

- **Local objective optimization ≠ global admissibility.**
- **Understanding ≠ authorization.**
- **Communication edge ≠ authority edge.**
- **Impact ↑ ⇒ autonomy budget ↓ and verification strength ↑.**
- **The Actor should not be the sole judge of its own admissibility.**

## Intended outputs

- Academic paper / preprint.
- Short human-readable agent constitution.
- Machine-readable YAML/JSON constitution.
- Benchmark scenarios for autonomous-agent dilemmas.
- Evaluation schema focused on structured decisions rather than hidden chain-of-thought.
- Historical genealogy from UEST/QUEST/Omega/RTR through AI Closure to Closure Ethics.
- Universal Closure Protocol reference implementation and benchmark.
- Vendor-neutral Global AI Closure Standard working draft.
- Public search-indexable landing page linking the research, DOI history, benchmark artifacts and machine-readable agent policy.
- Open PDF containing mathematical specification, recovered protocol lineage and complete reference code.

## Discovery and indexing

The `docs/` directory contains an SEO-ready static site, `robots.txt`, `sitemap.xml`, an experimental `llms.txt`, agent-facing policy/provenance files, and public pages for implementation, security, communication and standardization. Canonical URLs must be updated if a custom domain is adopted.

## Scientific discipline

Every claim should be tagged conceptually as one of:

- **Historical source** — documented in prior UEST/QUEST/Omega/RTR or AI Closure material.
- **Normative axiom** — ethical assumption introduced explicitly.
- **Formal consequence** — follows from defined mathematics or logic.
- **Empirical hypothesis** — testable claim about agent behavior or governance.
- **Speculation** — exploratory idea not yet justified.

The branch should remain falsifiable, auditable, versioned, and reversible.

## Historical project sentence

> **INTELLIGENCE PROPOSES POSSIBILITIES. CLOSURE DECIDES WHICH MAY BE REALIZED.**

Current Closure Ethics extends that historical line by asking whether execution also preserves the ability to recognize error, contest power and repair the future.
