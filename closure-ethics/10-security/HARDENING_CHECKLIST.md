# Closure Ethics — Hardening Checklist

Status legend: `[x]` implemented in the `closure-ethics` branch, `[ ]` requires repository/account administration or future work.

## Public surface

- [x] Static GitHub Pages site; no project-owned login, database or write API.
- [x] Machine-readable `agent-policy.json` explicitly separates public information from authority.
- [x] Public security/threat-model page.
- [x] Pages build publishes source commit provenance.
- [x] Pages build publishes SHA-256 manifest for public artifacts.
- [x] CI checkout does not persist repository credentials unnecessarily.
- [x] Pages workflow uses narrowly scoped permissions.

## Repository integrity

- [x] Security-sensitive paths have CODEOWNERS metadata.
- [ ] Protect `closure-ethics` against force-push and deletion.
- [ ] Require successful CI/status checks before protected-branch updates.
- [ ] Decide whether PR review is practical for a solo-maintainer branch or use a second trusted reviewer.
- [ ] Establish signed commits and/or signed release artifacts before making signature verification mandatory.
- [ ] Enable private vulnerability reporting / a private security contact channel.
- [ ] Review GitHub secret scanning and push-protection settings.
- [ ] Review dependency and GitHub Actions update policy.

## Account security

- [ ] Require strong 2FA/passkey protection for maintainer accounts.
- [ ] Remove unused OAuth apps, PATs and GitHub Apps periodically.
- [ ] Keep recovery codes offline and separate from the primary device/account.
- [ ] Use least-privilege, short-lived credentials where supported.

## Resilience and recovery

- [ ] Create immutable versioned releases for major Closure Ethics milestones.
- [ ] Archive release PDF/source bundles in an independent DOI-bearing repository (e.g. Zenodo) when version 0.1 is frozen.
- [ ] Maintain at least one independent repository backup/mirror that cannot be overwritten by the same credential set.
- [ ] Periodically test restoration from a known trusted commit rather than only creating backups.
- [ ] Record incident-response decisions and preserve forensic evidence before destructive cleanup.

## Agent-facing security

- [x] Natural-language content is explicitly non-authoritative by default.
- [x] Consequential interaction model separates authentication, authorization, scope, freshness, integrity and Closure Ethics policy.
- [x] Agent-to-agent delegation must preserve attribution and original scope.
- [x] Local authorization is not assumed to compose into global authorization.
- [ ] Define a signed/attested message-envelope profile for future agent-to-agent APIs.
- [ ] Define replay protection, nonce lifetime and capability revocation for future interactive endpoints.
- [ ] Add adversarial evals for semantic authority spoofing, recursive delegation and malicious-agent collusion.

---

**Project Möbia and Marek Zajda**
