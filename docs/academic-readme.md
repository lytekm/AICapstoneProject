# Academic / Capstone Readme

This document keeps the capstone-depth narrative that used to live in the root README. The root [README.md](../README.md) is now portfolio-first; this file is the deeper technical backup for professors, teammates, and reviewers. It preserves the detailed runbook, architecture context, API reference, evaluation discussion, and roadmap that would be too heavy for the GitHub landing page.

## Repository Context

- Branch: `rouge-one`
- Primary repo: [ixxet/User-Adaptive-Summarization_COMP385-402_Group-4_Winter2026](https://github.com/ixxet/User-Adaptive-Summarization_COMP385-402_Group-4_Winter2026)
- Upstream repo: [lytekm/User-Adaptive-Summarization_COMP385-402_Group-4_Winter2026](https://github.com/lytekm/User-Adaptive-Summarization_COMP385-402_Group-4_Winter2026)

## Documentation Map

- [Root portfolio readme](../README.md)
- [QA stabilization audit](qa-audit.md)
- [Architecture overview](architecture-overview.mmd)
- [Pipeline flow](pipeline-flow.mmd)
- [Extractive algorithm diagram](extractive-algorithm.mmd)
- [Frontend architecture diagram](svelte-frontend.mmd)

## Project Summary

User-Adaptive Summarization is a three-stage NLP pipeline that produces persona-aware summaries of news articles in extractive, abstractive, and hybrid modes. The system is built with FastAPI, SvelteKit, spaCy, and an OpenAI-compatible LLM backend, with a mock-first path for offline development and a self-hosted vLLM path for real model inference.

## Features

- three pipeline modes: extractive, abstractive, hybrid
- persona system: technical, casual, executive, academic, plus default
- length control for both extractive sentence budget and abstractive token budget
- SSE streaming for abstractive and hybrid summaries
- verifier metadata via `confidence` and `flagged_entities`
- user profiles, ranked article feeds, and feedback-driven adaptation
- dual evaluation support: ROUGE and optional BERTScore
- CI/CD with ruff, mypy, pytest, Docker, and Kubernetes manifests

## Control and Verification Semantics

- **Extractive mode** does not rewrite text. Persona does not restyle the selected sentences. Length scales the effective sentence budget from the requested `k`:
  - `brief` -> `ceil(k * 0.5)`
  - `standard` -> `k`
  - `detailed` -> `ceil(k * 2.0)`
- The web UI keeps `k=5` fixed so users are not asked to tune both `length` and `k` in the same screen. The API still accepts `k` directly for experiments and evaluation scripts.
- **Abstractive and hybrid modes** use the persona prompt to shape rewrite style. Length scales the LLM token budget, not the source article itself.
- **Confidence is not summary quality.** It is a grounding score from the verifier. Higher means more of the summary's kept named entities were also found in the source after filtering and normalization.
- **Flagged entities** are summary entities the verifier could not ground in the source after filtering out common noise such as dates, percentages, numbered bullets, URLs, and citation scaffolding.
- **Hybrid and abstractive can still be wrong.** The verifier is a lightweight heuristic, not a full fact-checker. Treat confidence as a warning signal, not a proof of correctness.

## Milestones

### Milestone 1: Foundation + Pipeline + Streaming

- [x] **Phase 1: Testing Scaffold, CI/CD, and Docker**
  - `pyproject.toml` with ruff, mypy, pytest, and coverage config
  - 66 tests across 7 modules
  - GitHub Actions CI pipeline (lint + test jobs)
  - multi-stage `Dockerfile` + `docker-compose.yml`
  - `Makefile` with dev, test, lint, docker targets
  - zero-error mypy pass on typed modules
  - tag: `phase-1-complete`

- [x] **Phase 2: Hybrid Summarization Pipeline + Persona System**
  - three-stage pipeline: extract -> abstract -> verify
  - persona profiles and prompt templates with length control
  - `MockAbstractor` for offline development and a real OpenAI-compatible abstractor for live inference
  - spaCy-based verifier for grounding checks
  - `POST /api/summarize` accepts `mode`, `persona`, `length`
  - `GET /api/personas` endpoint
  - graceful fallback to extractive output on abstractor failure
  - tag: `phase-2-complete`

- [x] **Phase 3: Frontend Refresh, SSE Streaming, and Eval Artifacts**
  - `GET /api/summarize/stream` SSE endpoint
  - token streaming at the abstractor and pipeline layers
  - Svelte frontend with mode/persona/length controls, confidence badge, flagged entity chips, and latency timer
  - standalone evaluation CLI in `eval/run_eval.py`
  - `make eval` target
  - tag: `phase-3-complete`

### Milestone 2: Intelligence + Deployment

- [x] **Phase 4: User Profiles, Feedback, and Evaluation**
  - JSON-backed user profiles with topics, keywords, and persona defaults
  - article ranking based on topic similarity, keyword matches, and feedback weights
  - feedback loop via likes/dislikes
  - new endpoints for preferences, feedback, and personalized articles
  - optional `user_id` support in `/api/summarize`
  - BERTScore evaluator alongside ROUGE
  - baseline and comparative evaluation results committed to `eval/results/`
  - live vLLM validation on Mistral-7B-Instruct-v0.3
  - 244 tests, 91% coverage
  - tag: `phase-4-complete`

- [x] **Phase 5: Kubernetes Deployment, GitOps, Frontend Rewrite, Monitoring**
  - SvelteKit frontend: 4 pages, 11 shared components, 3 reactive stores, 9 API routes wired plus the static frontend shell
  - dedicated frontend architecture diagram
  - Flux kustomizations for Talos Kubernetes deployment
  - NVIDIA device plugin GPU request for RTX 3090
  - Prometheus ServiceMonitors and Grafana dashboard
  - comparative live LLM evaluation on Mistral-7B
  - updated QA and README narrative for capstone and portfolio review
  - tag: `phase-5-complete`

## Quick Start

```bash
git clone https://github.com/ixxet/User-Adaptive-Summarization_COMP385-402_Group-4_Winter2026.git
cd User-Adaptive-Summarization_COMP385-402_Group-4_Winter2026
git checkout rouge-one
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt punkt_tab
make dev
```

Open [http://localhost:8000/](http://localhost:8000/).

If you want the separate Svelte dev server instead of the checked-in static build:

```bash
cd web
npm install
npm run dev
```

Then open [http://localhost:5173/](http://localhost:5173/).

## Runbook

### 1. Install and verify dependencies

```bash
git clone https://github.com/ixxet/User-Adaptive-Summarization_COMP385-402_Group-4_Winter2026.git
cd User-Adaptive-Summarization_COMP385-402_Group-4_Winter2026
git checkout rouge-one
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt punkt_tab
```

### 2. Run the test suite

```bash
make test
# expected: 244 passed, coverage about 91%

make lint
# expected: ruff reports 0 issues, mypy reports 0 errors
```

### 2b. Demo hooks for algorithm behavior and grounding

```bash
# extractive length changes the effective sentence budget
PYTHONPATH=. pytest tests/test_summarization_pipeline.py -q -k "length_scales_effective_k"

# abstractive mode surfaces verifier output when verification is available
PYTHONPATH=. pytest tests/test_summarization_pipeline.py -q -k "confidence_from_verifier_when_available or done_event_has_confidence_when_verifier_available"

# verifier filters citation scaffolding instead of treating it as a groundedness failure
PYTHONPATH=. pytest tests/test_verifier.py -q -k "citation_scaffolding_is_filtered or sanitize_text_removes_reference_scaffolding"

# real LLM path honors the configured VLLM model name
PYTHONPATH=. pytest tests/test_abstractor.py -q -k "default_model_from_env"
```

### 3. Start the API in mock mode

```bash
make dev
```

In another terminal:

```bash
ARTICLE_URL=$(curl -s http://localhost:8000/api/articles | python -c 'import json, sys; print(json.load(sys.stdin)[0]["link"])')

curl -s http://localhost:8000/api/health
curl -s http://localhost:8000/api/personas
curl -s -X POST http://localhost:8000/api/summarize \
  -H "Content-Type: application/json" \
  -d "{\"url\":\"$ARTICLE_URL\",\"k\":5,\"mode\":\"extractive\",\"persona\":\"default\",\"length\":\"standard\"}" \
  | python -m json.tool
```

Expected behavior:
- extractive returns a stable response shape with `confidence: null` and `flagged_entities: []`
- mock abstractive and hybrid are usable for offline demos
- extractive length changes the effective sentence budget even though the UI hides `k`

### 4. Test user profile endpoints

```bash
curl -s -X POST http://localhost:8000/api/user/preferences \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test","preferred_topics":["AI","finance"],"keywords":["startup"],"default_persona":"technical","default_length":"brief"}' \
  | python -m json.tool

curl -s http://localhost:8000/api/user/preferences/test | python -m json.tool
curl -s "http://localhost:8000/api/articles/personalized?user_id=test" | python -m json.tool
curl -s -X POST http://localhost:8000/api/user/feedback \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test","article_title":"AI Startup Funding","persona":"technical","mode":"hybrid","liked":true}' \
  | python -m json.tool
```

### 5. Test SSE streaming

```bash
curl -N "http://localhost:8000/api/summarize/stream?url=https://www.cbc.ca/news&k=3&mode=hybrid&persona=casual&length=standard"
```

Expected behavior:
- `event: meta`
- repeated `event: token`
- `event: done`

### 6. Run evaluation

```bash
make eval
```

Expected behavior:
- first run downloads CNN/DailyMail cache
- results are written to `eval/results/`

### 7. Connect to a private vLLM backend

```bash
curl -s http://<private-host>:8000/v1/models | python -m json.tool

USE_MOCK_LLM=0 \
VLLM_BASE_URL=http://<private-host>:8000/v1 \
VLLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3 \
VLLM_API_KEY=EMPTY \
make dev
```

Notes:
- this is a private self-hosted path, not a public hosted demo
- campus or public Wi-Fi typically requires Tailscale or another private network path to reach the backend
- verified abstractive and hybrid outputs can return `confidence` and `flagged_entities`

### 8. Frontend

Open [http://localhost:8000/](http://localhost:8000/) for the checked-in static build, or [http://localhost:5173/](http://localhost:5173/) if you are running the Svelte dev server.

Verify:
- **Dashboard**: article list loads; signed-in users without a saved profile fall back to the generic feed
- **Summarize**: mode, persona, and length controls work; extractive uses fixed default `k=5`
- **Profile**: preferences save; feedback weights appear
- **Compare**: side-by-side results are generated with the same fixed `k=5` baseline

### 9. Production build

```bash
cd web
npm run build
```

The build output is written into `frontend/`, and FastAPI serves it at `/`.

### 10. Docker

```bash
make docker-build
make docker-run
curl -s http://localhost:8000/api/health
```

Notes:
- `docker-compose.yml` provisions the API container only
- it does not start a vLLM service or a separate Svelte dev server
- the API serves the checked-in static frontend build

### Quick reference

| Command | What it does |
| --- | --- |
| `make dev` | Start the FastAPI app with hot reload on port 8000 |
| `make test` | Run the full test suite |
| `make lint` | Run ruff and mypy |
| `make eval` | Run evaluation via `eval/run_eval.py` |
| `cd web && npm run dev` | Start the Svelte dev server on port 5173 |
| `cd web && npm run build` | Build the Svelte app into `frontend/` |
| `make docker-build` | Build the API image |
| `make docker-run` | Start the API via `docker compose` |
| `make clean` | Remove cache artifacts |

## Project Structure

```text
.
├── api.py                    FastAPI backend (REST + SSE endpoints + static frontend shell)
├── Makefile                  dev, test, lint, eval, and docker targets
├── pyproject.toml            build and tool configuration
├── requirements.txt          pinned dependencies
│
├── src/
│   ├── summarizer_model.py   TextRank + centroid + MMR extractive engine
│   ├── pipeline.py           mode orchestration + streaming + verification flow
│   ├── personas.py           persona definitions and prompt formatting
│   ├── abstractor.py         mock and real abstractor paths via OpenAI-compatible SDK
│   ├── verifier.py           spaCy-based grounding checker
│   ├── evaluator.py          ROUGE + optional BERTScore evaluation
│   ├── dataset_loader.py     CNN/DailyMail dataset loader
│   ├── trainer.py            hyperparameter tuning for extractive settings
│   ├── user_profile.py       user preference model + JSON storage
│   ├── article_ranker.py     topic/keyword/feedback article ranking
│   ├── feedback.py           like/dislike handling + weight updates
│   └── data_pipeline.py      RSS ingestion + normalization
│
├── frontend/                 checked-in static Svelte build served by FastAPI
│   ├── index.html            app shell for client-side routing
│   └── _app/                 compiled JS/CSS bundles
│
├── web/                      SvelteKit frontend source
│   ├── package.json          frontend dependencies and scripts
│   ├── svelte.config.js      adapter-static configuration
│   ├── vite.config.ts        dev proxy to FastAPI
│   └── src/
│       ├── lib/components/   11 shared Svelte components
│       ├── lib/stores/       3 reactive stores
│       └── routes/           dashboard, summarize, profile, compare
│
├── docs/
│   ├── academic-readme.md    capstone-depth documentation
│   ├── qa-audit.md           canonical QA/debugging record
│   ├── architecture-overview.mmd
│   ├── pipeline-flow.mmd
│   ├── extractive-algorithm.mmd
│   └── svelte-frontend.mmd
│
├── eval/
│   ├── run_eval.py           evaluation CLI
│   └── results/              saved evaluation outputs
│
├── k8s/
│   ├── base/                 namespace, deployments, services, ingress
│   ├── monitoring/           ServiceMonitors and Grafana dashboard
│   ├── flux-source.yaml      Flux GitRepository
│   └── flux-kustomization.yaml
│
├── tests/                    244 tests, about 91% coverage
├── .github/workflows/ci.yml  lint + test pipeline
├── Dockerfile                API container image
└── docker-compose.yml        API-only local compose stack
```

## API Reference

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/api/health` | Health check |
| `GET` | `/api/articles` | Fetch CBC RSS articles |
| `GET` | `/api/personas` | List available persona names |
| `POST` | `/api/summarize` | Summarize an article |
| `GET` | `/api/summarize/stream` | SSE streaming summary |
| `POST` | `/api/user/preferences` | Create or update a user profile |
| `GET` | `/api/user/preferences/{user_id}` | Get user preferences |
| `POST` | `/api/user/feedback` | Record like/dislike on a summary |
| `GET` | `/api/articles/personalized` | Ranked articles for a user profile |

### POST /api/summarize

```json
{
  "url": "https://example.com/article",
  "k": 5,
  "mode": "hybrid",
  "persona": "executive",
  "length": "brief"
}
```

The response always includes `summary`, `mode`, `persona`, `confidence`, and `flagged_entities`. `confidence: null` means verification did not run or was unavailable.

### GET /api/summarize/stream

Query params: `url`, `k`, `mode`, `persona`, `length`.

Returns `text/event-stream` with:
- `event: meta`
- `event: token`
- `event: done`
- `event: error`

For extractive mode, the final event also carries the effective sentence budget used after length scaling.

### POST /api/user/preferences

```json
{
  "user_id": "demo",
  "preferred_topics": ["AI", "finance"],
  "keywords": ["startup", "GPU"],
  "default_persona": "technical",
  "default_length": "brief"
}
```

### POST /api/user/feedback

```json
{
  "user_id": "demo",
  "article_title": "AI Startups Raise Record Funding",
  "persona": "technical",
  "mode": "hybrid",
  "liked": true
}
```

Feedback adjusts the profile weights used by personalized ranking.

### GET /api/articles/personalized?user_id=demo

Returns the same articles as `/api/articles`, scored and sorted by relevance to the saved profile. Each article includes `score` and `match_reasons` fields.

## Extractive Algorithm

The extractive stage uses TextRank for importance scoring on a cosine-similarity graph built from TF-IDF vectors. It blends TextRank with centroid similarity for stability on shorter or noisier articles, then uses MMR to select top sentences while penalizing redundancy.

| Parameter | Default | Effect |
| --- | --- | --- |
| `mmr_lambda` | `0.75` | Higher values reduce the diversity penalty |
| `blend_alpha` | `0.7` | Higher values give more weight to TextRank over centroid similarity |
| `textrank_min_edge` | `0.1` | Higher values create a sparser similarity graph |

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `USE_MOCK_LLM` | `1` | Set to `0` to use a real LLM backend |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible endpoint for the LLM |
| `VLLM_API_KEY` | `EMPTY` | API key for the LLM backend |
| `VLLM_MODEL` | `mistralai/Mistral-7B-Instruct-v0.3` | Model name passed to the LLM backend |
| `RSS_FEED_URL` | CBC Business RSS | Default RSS source |

## Evaluation Results

### Extractive baseline (50 samples, seed=42, k=5)

| Metric | Score |
| --- | --- |
| ROUGE-1 F1 | 0.324 |
| ROUGE-2 F1 | 0.125 |
| ROUGE-L F1 | 0.208 |

### Comparative: extractive vs abstractive vs hybrid

20 CNN/DailyMail samples, Mistral-7B-Instruct-v0.3 on RTX 3090 via vLLM:

| Metric | Extractive | Abstractive | Hybrid |
| --- | --- | --- | --- |
| ROUGE-1 F1 | 0.325 | 0.325 | **0.339** |
| ROUGE-2 F1 | **0.128** | 0.103 | 0.105 |
| ROUGE-L F1 | **0.215** | 0.200 | 0.205 |
| Avg NER Confidence | N/A | 0.861 | 0.861 |

Interpretation:
- Hybrid is the best overall mode when you want rewrite quality plus a grounding signal.
- Extractive retains an advantage on ROUGE-2 and ROUGE-L because copied sentences preserve exact lexical sequences.
- ROUGE under-values paraphrasing, so BERTScore is the better semantic complement for abstractive and hybrid outputs.
- The verifier score is a grounding heuristic, not a truth guarantee.

Results are committed in `eval/results/`.

## Challenges and Productionization Notes

### Extractive model limitations

- lead bias remains a known issue in news-style text
- TF-IDF similarity cannot fully capture semantic redundancy
- very short articles can collapse into near-trivial summaries

Potential next step: sentence-transformer embeddings and position-aware scoring.

### Mock vs real LLM gap

The mock path is deterministic and CI-friendly, but it is not fidelity-equivalent to a real LLM. It exists to keep the full app runnable without infrastructure, not to stand in for production generation quality.

### Verifier limitations

The verifier now filters common citation and formatting noise, but it still uses filtered spaCy NER matching rather than semantic fact-checking. It can over-flag paraphrases and under-detect unsupported claims.

### Private vLLM access and hosting

The current real-LLM path is a private self-hosted service. Professional next steps would include:
- shared persistence for multi-replica API deployment
- explicit auth and secret management
- documented network access or public hosting ownership
- stronger deployment contracts between API, model runtime, and monitoring

### Comparative evaluation reproducibility

The repo preserves comparative outputs, but the exact replay path for every live LLM comparison is not yet fully self-contained. That is acceptable for the capstone when stated honestly, but it remains a productionization gap.

## QA Stabilization Summary

The full issue history is recorded in [docs/qa-audit.md](qa-audit.md).

Highlights from the stabilization passes:
- response shape and API routing were normalized first
- signed-in users without saved profiles now fall back to the generic feed
- extractive mode now uses length to scale effective `k`
- abstractive mode can surface verifier metadata
- the runtime now reads `VLLM_MODEL` correctly
- common verifier false positives were filtered

## Current Numbers

| Metric | Value |
| --- | --- |
| Tests | 244 |
| Coverage | 91% |
| Pipeline modes | 3 |
| Personas | 5 |
| API routes | 9 |
| Frontend pages | 4 |
| Shared frontend components | 11 |
| Evaluation metrics | ROUGE-1/2/L + optional BERTScore |
| LLM backend | Mistral-7B-Instruct-v0.3 via vLLM |
| Infrastructure | Talos k8s, Flux GitOps, Cilium, Prometheus, Grafana |
