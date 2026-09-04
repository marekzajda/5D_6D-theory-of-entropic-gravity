# SEO, Indexing, and Research Discovery Strategy

## Objective

Make Closure Ethics easy to discover through ordinary web search, scholarly search, repository search, DOI-linked archives, and machine-readable research discovery without sacrificing scientific precision.

## Principles

1. Prefer clear terminology over keyword stuffing.
2. Use one canonical public landing page for the project.
3. Keep titles, abstracts, keywords, repository metadata, and DOI metadata consistent.
4. Link every public artifact bidirectionally where possible: website ↔ GitHub ↔ Zenodo/DOI ↔ preprint.
5. Publish stable URLs and avoid unnecessary renaming after indexing.
6. Distinguish historical claims, normative axioms, formal results, empirical hypotheses, and speculation.

## Primary search vocabulary

Core terms:

- Closure Ethics
- Closure Ethics for Autonomous Agents
- Human-AI Symbiosis
- Human-AI Cognitive Symbiosis
- Closure-Preserving Symbiosis
- Autonomous Agent Ethics
- Multi-Agent Safety
- AI Agent Governance
- AI Auditability
- AI Repairability
- Reversible AI Actions
- Agency Preservation
- Agent Constitution
- Regulative Theory of Reality
- Omega RTR

Secondary terms should be used naturally when relevant, not repeated mechanically.

## Public discovery stack

### Layer 1 — Canonical landing page

A lightweight public site should contain:

- one unambiguous H1 title;
- 150–250 word summary;
- author/project attribution;
- links to GitHub, DOI/preprints, constitution, formalism, benchmarks, and source history;
- canonical URL;
- Open Graph metadata;
- JSON-LD structured metadata;
- sitemap.xml;
- robots.txt;
- stable internal links.

Prepared site files live under `closure-ethics/docs/` and can be published through GitHub Pages.

### Layer 2 — GitHub

Repository pages should use consistent terms in:

- README title and first paragraph;
- repository description/topics when available;
- filenames and folder names;
- release titles;
- CITATION metadata;
- issue/benchmark terminology.

### Layer 3 — DOI / scholarly publication

For each stable paper or release:

- publish to Zenodo or another DOI-bearing archive;
- use the exact project title consistently;
- provide a strong abstract and keywords;
- include the GitHub URL and canonical website URL in metadata;
- link back from GitHub to the DOI.

### Layer 4 — Google discovery

After the public site is live:

1. Verify the property in Google Search Console.
2. Submit `sitemap.xml`.
3. Use URL Inspection → Request Indexing for the homepage and the most important newly published pages.
4. Monitor indexing and canonicalization reports.
5. Fix crawl or canonical errors before creating more content.

A sitemap helps discovery but does not guarantee indexing or ranking.

## Content strategy

Publish a small number of high-value pages rather than many thin pages:

1. **What is Closure Ethics?**
2. **The 10 Principles**
3. **Human–AI Symbiosis**
4. **Formal Definition of Ethical Admissibility**
5. **Agent Constitution**
6. **Benchmark / Evaluation Scenarios**
7. **Genealogy: UEST → QUEST → Omega → RTR → Closure Ethics**
8. **Papers, DOI, data and source material**

Each page should answer a distinct search intent and link to the underlying research source.

## Credibility signals

- DOI links and publication dates.
- Version numbers and changelog.
- Explicit limitations.
- Public benchmark data.
- Reproducible definitions.
- Clear authorship and citation instructions.
- Stable citations to historical Omega/RTR material.

## AI / machine discovery

A concise `llms.txt` may be provided as an experimental machine-readable index for systems that choose to use it. It is not a Google indexing mechanism and should not be treated as an SEO guarantee.

## Avoid

- keyword stuffing;
- fake citations or fabricated endorsements;
- claims that RTR physics proves ethics;
- sensational titles that exceed the evidence;
- duplicate pages with nearly identical content;
- publishing multiple competing canonical URLs.

## Deployment target

Default GitHub Pages project URL, if enabled without a custom domain:

`https://marekzajda.github.io/5D_6D-theory-of-entropic-gravity/`

If a custom domain is later adopted, canonical URLs, sitemap URLs, and robots.txt must be updated together.