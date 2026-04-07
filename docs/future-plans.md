# Future Plans and Improvements

This document condenses the oversized historical phase notes into the subset that still matters on `rouge-one`. The goal is to keep one planning record that is short enough to maintain and honest enough to match the branch as it exists today.

## Planning Triage

| Legacy note | Current state on `rouge-one` | Keep as future work? | Why |
| --- | --- | --- | --- |
| Sentence embeddings for the extractive stage | Not implemented | Yes | strongest remaining algorithm upgrade for semantic similarity |
| Position bias for news-style articles | Not implemented | Maybe later | useful, but lower priority than verifier and deployment work |
| Vectorized MMR refactor | Not implemented | Not now | current performance is acceptable; not a bottleneck today |
| User profiles and JSON storage | Implemented | No | already shipped |
| Article ranking | Implemented | No | already shipped |
| Feedback system | Implemented | No | already shipped and `liked` is strict-boolean validated |
| Personalized API endpoints | Implemented | No | already shipped |
| BERTScore evaluation | Implemented | No | already shipped |
| Real vLLM validation | Implemented, but operationally fragile | Keep as preflight, not roadmap | belongs in runbooks and operational checks |
| Mock vs real LLM fidelity gap | Improved, but still intentionally approximate | Keep as caveat, not a main feature track | mock mode keeps the app portable; it should not be described as parity with live inference |
| Svelte frontend rewrite | Implemented | No | already shipped |
| Flux / Kubernetes deployment scaffolding | Implemented | No | already shipped |
| Prometheus / Grafana instrumentation | Implemented | No | already shipped |
| Load testing | Not implemented | Yes | still useful for portfolio-grade throughput numbers |
| Mermaid render validation in CI | Not implemented | Yes | prevents docs and diagram drift |

## Condensed Future Plans

| Priority | Improvement | Why it matters | Scope |
| --- | --- | --- | --- |
| P1 | Shared persistence for profiles and feedback | removes the biggest multi-replica correctness gap in the current deployment story | Fundamental |
| P1 | Deployment hardening for private live inference | separates public app exposure from private vLLM access and makes auth, secrets, and network ownership explicit | Moderate |
| P1 | Reproducible live LLM comparative evaluation runner | turns one-off comparisons into repeatable evidence with locked prompts, model IDs, runtime config, and saved artifacts | Moderate |
| P2 | Verifier v2: NER plus claim-level entailment | improves trust, confidence semantics, and false-positive handling | Fundamental |
| P2 | Embeddings-assisted extractive relevance | modernizes the extractive stage without throwing away the current TextRank + centroid + MMR core | Moderate |
| P2 | Evidence-aware and structured summary outputs | makes claims, support sentences, and verifier notes easier to surface in the UI and in evaluation | Moderate |
| P3 | Clearer separation between mock-mode demos and live-inference claims | keeps onboarding honest and prevents reviewers from mistaking approximated mock behavior for real LLM behavior | Slot-in |
| P3 | Load testing and throughput reporting | adds concrete capacity numbers for demos, review, and portfolio positioning | Slot-in |
| P3 | Mermaid render checks in CI | keeps diagram docs from drifting away from the codebase | Slot-in |

## Recommended Next Phase

| Track | First move | Why |
| --- | --- | --- |
| Persistence and deployment | move profile and feedback storage behind Postgres or another shared store | fixes the biggest architecture caveat first |
| Verification | add verifier v2 before chasing more style knobs | the main limitation is trust, not more formatting variety |
| Evaluation | pin sample set, prompt version, model name, and runtime settings for live runs | makes future comparisons defensible |
| Algorithm | add optional sentence embeddings behind a feature flag | improves semantic similarity while preserving the baseline |
| Demo operations | replace ad hoc tunnel use with a named or protected access path when needed | cleaner remote demo story |

## Notes

1. The branch should not keep repeating historical phase notes as if everything is still pending.
2. The root README stays short on purpose; this file is the deeper planning summary.
3. Operational checks such as "confirm the app is not in mock mode before tunneling" belong in [operational-status.md](operational-status.md), not here.
