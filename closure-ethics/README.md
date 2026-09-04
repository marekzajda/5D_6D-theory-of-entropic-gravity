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
│   └── GENEALOGY.md
├── 07-sources/
│   └── SOURCE_MAP.md
├── 08-discovery/
│   └── SEO_AND_INDEXING.md
└── docs/
    ├── index.html
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

## Intended outputs

- Academic paper / preprint.
- Short human-readable agent constitution.
- Machine-readable YAML/JSON constitution.
- Benchmark scenarios for autonomous-agent dilemmas.
- Evaluation schema focused on structured decisions rather than hidden chain-of-thought.
- Historical genealogy from UEST/QUEST/Omega/RTR to Closure Ethics.
- Public search-indexable landing page linking the research, DOI history, and benchmark artifacts.

## Discovery and indexing

The `docs/` directory contains an SEO-ready static landing page, `robots.txt`, `sitemap.xml`, and an experimental `llms.txt`. The site is prepared for publication through GitHub Pages. Canonical URLs must be updated if a custom domain is adopted.

## Scientific discipline

Every claim should be tagged conceptually as one of:

- **Historical source** — documented in prior UEST/QUEST/Omega/RTR material.
- **Normative axiom** — ethical assumption introduced explicitly.
- **Formal consequence** — follows from defined mathematics or logic.
- **Empirical hypothesis** — testable claim about agent behavior or governance.
- **Speculation** — exploratory idea not yet justified.

The branch should remain falsifiable, auditable, versioned, and reversible.