# QA Audit Record

This is the canonical QA and debugging record for the current stabilization pass on the `rouge-one` checkout.

## Repository Context

| Field | Value |
| --- | --- |
| Branch | `rouge-one` |
| Local `origin` | `ixxet/User-Adaptive-Summarization_COMP385-402_Group-4_Winter2026` |
| Local `upstream` | `lytekm/User-Adaptive-Summarization_COMP385-402_Group-4_Winter2026` |
| Connected GitHub identity | `IzzetAbidi` |
| Audit location | `.claude/worktrees/recursing-rubin` |

## Audit Notes

The findings below were checked against the current source tree, frontend stores, backend routes, deployment manifests, and evaluation scripts. Items are marked conservatively:

- `Open` means the issue is a real defect that should be fixed in a minimal follow-up commit.
- `Known limitation` means the behavior is intentional for this pass or not worth changing in the first wave.
- `Deferred` means the issue is real, but the fix belongs to a larger follow-up such as persistence or evaluation redesign.

## Initial Audit Snapshot

This table captures the original audit state before the first stabilization commits landed.

| Symptom | Root cause | Impact | Fix / disposition | Status | Commit |
| --- | --- | --- | --- | --- | --- |
| Unknown `/api/*` GET requests return HTML instead of API errors | The frontend catch-all route serves `index.html` for any unmatched GET path, including mistyped API paths. | Clients see `200 text/html` where they expect a 404, which hides routing mistakes and breaks error handling. | Restrict the catch-all to non-API paths or explicitly reject `/api/` prefixes before falling back to the frontend shell. | Open | pending |
| Extractive summaries render `NaN` confidence in the UI | The backend omits `confidence` and `flagged_entities` in extractive mode, while the badge component only treats `null` as unavailable. | The dashboard and summarize page show a broken confidence indicator instead of a neutral state. | Return a stable extractive shape and treat missing confidence as unavailable in the UI. | Open | pending |
| Hybrid and abstractive summaries can show `100%` confidence even when verification did not actually run | Confidence is hard-coded or falls back to `1.0` when verification is unavailable, and the verifier silently disables itself if spaCy/model loading fails. | The confidence value is not safe to read as factual assurance. | Only score confidence when verification actually ran; otherwise report `null` and surface verification as unavailable. | Open | pending |
| Signed-in users with no saved profile lose their article feed | The frontend treats any non-empty user ID as logged in, then requests the personalized feed; the backend returns 404 when the profile does not exist, and the store clears the list. | The dashboard appears empty for users who have signed in but have not created a profile yet. | Fall back to the generic feed for missing profiles and keep the missing-profile state non-fatal. | Open | pending |
| Persona and length appear inert in mock mode | The mock abstractor originally ignored persona styling and length scaling, so every offline summary looked nearly identical. | During offline/mock runs, the controls did not demonstrate the intended behavior of the summarization workspace. | Replace the fixed mock output with deterministic persona and length heuristics so the UI controls produce visibly different summaries even without a live LLM. | Open | pending |
| Profile defaults can overwrite manual persona/length selection after load | The summarize form reapplies profile defaults reactively whenever a profile exists. | The user's manual control changes can feel sticky or ignored. | Apply defaults once on profile load, then leave manual selections alone. | Open | pending |
| Feedback can be recorded incorrectly when the payload is not a real boolean | `liked` is coerced with `bool(liked)`, so non-empty strings such as `"false"` become truthy. | Adaptive ranking weights can be trained on the wrong signal. | Validate `liked` as a strict boolean in the request model. | Open | pending |
| Personalized state diverges across the two API replicas in Kubernetes | Profiles and feedback are stored in pod-local JSON files, but the deployment runs multiple replicas with no shared persistence. | A user can receive different ranking behavior depending on which pod serves the request. | Defer until shared storage or a database is introduced. | Deferred | n/a |
| The comparative live-LLM evaluation artifact is not reproducible from the checked-in runner | The committed evaluation CLI only runs the extractive baseline. The comparative `extractive/abstractive/hybrid` numbers were generated through a separate path. | The repo preserves results, but not a single self-contained replay path for those comparative numbers. | Defer until a comparative runner or scripted replay path is added. | Deferred | n/a |

## Verification Summary

The current stabilization pass has been checked against the live `rouge-one` worktree with the following commands:

- `PYTHONPATH=. ruff check api.py src/pipeline.py tests/test_api.py tests/test_summarization_pipeline.py`
- `PYTHONPATH=. pytest tests/test_api.py tests/test_summarization_pipeline.py -q`
- `python3 -m compileall -q api.py src tests`
- `npm run check`
- `npm run build`

## Current Status

| Symptom | Root cause | Impact | Fix / disposition | Status | Commit |
| --- | --- | --- | --- | --- | --- |
| Unknown `/api/*` GET requests return HTML instead of API errors | The frontend catch-all route served `index.html` for unmatched API GET paths. | Clients received HTML instead of an API 404. | API-prefixed paths now return `404 Not Found` before the frontend fallback runs. | Fixed | `bf8ef49` |
| Extractive summaries render `NaN` confidence in the UI | The backend omitted extractive confidence fields and the UI did not normalize missing values. | The dashboard and summarize page showed a broken confidence indicator. | The API now returns a stable response shape and the frontend normalizes unavailable confidence to `N/A`. | Fixed | `bf8ef49`, `29e3f0b` |
| Hybrid and abstractive summaries can show `100%` confidence even when verification did not actually run | Confidence was hard-coded or defaulted to `1.0` when verification was unavailable. | The confidence value overstated factual assurance. | The pipeline now returns `null` unless verification actually ran. | Fixed | `bf8ef49` |
| Signed-in users with no saved profile lose their article feed | The frontend treated any non-empty `userId` as personalized mode and cleared the feed on a missing-profile 404. | The dashboard appeared empty for users without saved preferences. | Feed loading now depends on a loaded profile and falls back to generic articles when the profile is missing. | Fixed | `29e3f0b` |
| Persona and length appear inert in mock mode | The mock abstractor originally ignored persona styling and length scaling, so every offline summary looked nearly identical. | Offline/mock runs did not demonstrate the intended behavior of the summarization workspace. | Mock mode now applies deterministic persona and length heuristics for visible UI feedback. It remains an approximation, not a substitute for a real LLM. | Fixed | `8e17c14` |
| Profile defaults can overwrite manual persona/length selection after load | The summarize form kept reapplying profile defaults reactively. | Manual selections could appear sticky or ignored. | Profile defaults now prefill once per profile load and stop overwriting manual user choices. | Fixed | `29e3f0b` |
| Feedback can be recorded incorrectly when the payload is not a real boolean | `liked` was coerced with `bool(liked)`. | Adaptive ranking weights could be trained on the wrong signal. | Feedback now uses strict boolean validation at the request boundary. | Fixed | `bf8ef49` |
| Personalized state diverges across the two API replicas in Kubernetes | Profiles and feedback are stored in pod-local JSON files, but the deployment runs multiple replicas with no shared persistence. | A user can receive different ranking behavior depending on which pod serves the request. | Defer until shared storage or a database is introduced. | Deferred | n/a |
| The comparative live-LLM evaluation artifact is not reproducible from the checked-in runner | The committed evaluation CLI only runs the extractive baseline. The comparative `extractive/abstractive/hybrid` numbers were generated through a separate path. | The repo preserves results, but not a single self-contained replay path for those comparative numbers. | Defer until a comparative runner or scripted replay path is added. | Deferred | n/a |

## Disposition Summary

- The first stabilization pass should stay minimal and target the contract mismatches, not the larger architecture work.
- Known limitations should stay documented as such so the repo does not overclaim mock-mode fidelity or multi-replica personalization safety.
- Deferred items are still worth tracking because they affect reproducibility and deployment realism, but they are not first-pass blockers.
