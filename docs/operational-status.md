# Operational Status Tables

This document keeps the table-first operational notes for the project. It is the right place for status snapshots, demo-day checks, release policy, and troubleshooting summaries that would be too heavy for the landing-page README.

## Documentation Placement

| Thing | Keep it where | Why |
| --- | --- | --- |
| Portfolio summary, architecture snapshot, quick run path | `README.md` | Keeps the landing page short and recruiter-friendly |
| Deep technical narrative, runbook, API reference, milestones | `docs/academic-readme.md` | Better fit for professors, teammates, and capstone review |
| Operational status tables, demo-day checks, versioning rules, troubleshooting summaries | `docs/operational-status.md` | Best fit for fast scanning and day-to-day project operations |

## Semantic Versioning

| Thing | State | Proof / Detail |
| --- | --- | --- |
| Versioning policy | SemVer | The project already uses semantic version formatting in `pyproject.toml` |
| Current project version | `0.2.0` | Declared in [pyproject.toml](../pyproject.toml) |
| Release posture | Pre-`1.0.0` | The project is stable enough to version, but interfaces and deployment behavior are still evolving |

| Change type | Version bump | Example |
| --- | --- | --- |
| Patch | `0.2.x` | bug fix, doc correction, CI fix, tunnel/runbook clarification |
| Minor | `0.x.0` next minor | new non-breaking feature such as metrics, startup targets, or improved verifier behavior |
| Major | `1.0.0` or next major | breaking API contract, persistence redesign with migration, or major deployment/runtime contract change |

Notes:
- Before `1.0.0`, the project should still try to behave like SemVer, but it is reasonable to treat the branch as evolving software rather than a frozen public API.
- A good threshold for `1.0.0` would be: stable persistence layer, stable startup/deployment contract, and a reproducible live-LLM evaluation path.

## Startup Paths

| Path | Command | Use when | Good proof |
| --- | --- | --- | --- |
| Mock mode | `make dev-mock` | offline development, CI-like local testing, no private GPU backend available | summaries may begin with `[Mock Summary]` in abstractive/hybrid |
| Default alias | `make dev` | backwards-compatible shortcut for mock mode | same behavior as `make dev-mock` |
| Real LLM mode | `make dev-real VLLM_BASE_URL=http://<private-host>:8000/v1 VLLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3 VLLM_API_KEY=EMPTY` | live inference against the private vLLM backend | abstractive/hybrid summaries do **not** begin with `[Mock Summary]` |
| Temporary tunnel | `make tunnel-real` | short-lived external demo after real LLM mode is already verified locally | tunnel URL opens the same app state already proven locally |

Notes:
- Only tunnel the app after confirming that one live `abstractive` or `hybrid` request does not return `[Mock Summary]`.
- `make tunnel-real` opens an unauthenticated quick tunnel. It is suitable for short demos, not for hardened deployment.

## Demo-Day Preflight

| Thing | Good state | Proof / Detail |
| --- | --- | --- |
| App process | Healthy | `GET /api/health` returns `{"status":"ok"}` |
| Startup mode | Real inference intended | running process has `USE_MOCK_LLM=0` |
| Target backend | Correct | running process has `VLLM_BASE_URL` and `VLLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3` |
| Live LLM path | Healthy | one `abstractive` or `hybrid` summarize call returns a non-mock summary |
| Verifier | Available | `hybrid` can return non-null `confidence` when the verifier is loaded and the request completes normally |
| Tunnel | Optional and temporary | only started after local real-inference verification passes |

Recommended preflight sequence:
1. Start the app with `make dev-real ...`.
2. Check `GET /api/health`.
3. Run one non-streaming `abstractive` summarize call.
4. Run one streaming `hybrid` summarize call.
5. Only then open the quick tunnel if external access is needed.

## Failure Mode Summary

| Symptom | Meaning | Most likely cause | Next check |
| --- | --- | --- | --- |
| Summary begins with `[Mock Summary]` | The app is in mock mode | started with `make dev` / `make dev-mock`, or `USE_MOCK_LLM` not set to `0` | inspect the running process environment |
| `hybrid` returns `confidence: null` and `flagged_entities: []` | Verification did not run or the request degraded before verification completed | verifier model missing, verifier unavailable, or LLM path timed out | check spaCy model availability and stream `done` payload |
| Streaming `done` payload includes `fallback: "extractive"` | The LLM path failed and the pipeline degraded gracefully | vLLM timeout, network issue, or backend overload | probe `VLLM_BASE_URL/models` directly |
| No flagged chips appear in the UI | There were no flagged entities to display | `flagged_entities` was empty | inspect the JSON response before assuming a frontend bug |
| Tunnel works but the app behaves wrong | The tunnel is exposing the current local app state, not changing it | app was already misconfigured or degraded before tunneling | verify local app behavior first, then tunnel |

## Current Diagnostic Rule

| Question | Answer | Proof / Detail |
| --- | --- | --- |
| Does a tunnel prove the app is using the real LLM? | No | The tunnel only exposes whatever local process is already running |
| Does a healthy `/api/health` prove live inference is healthy? | No | It proves the API is alive, not that vLLM is reachable or responsive |
| What is the strongest single proof of live inference? | A real `abstractive` or `hybrid` summarize response that does not start with `[Mock Summary]` | This confirms end-to-end non-mock execution |

