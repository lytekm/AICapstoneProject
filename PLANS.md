# Project Context & Upgrade Plan

---

## Project identity

- **Name**: User-Adaptive Summarization (COMP385-402 Capstone, Group 4, Winter 2026)
- **Repo root**: `User-Adaptive-Summarization_COMP385-402_Group-4_Winter2026/`
- **Current grade**: B (78/100) — technically solid extractive summarizer, but capstone-level deductions for scope gap, low test coverage, and missing eval artifacts

---

## Current codebase map

| File | Role |
|---|---|
| `src/summarizer_model.py` | Core algorithm — `TextRankMMRSummarizer` with PageRank, MMR selection, centroid blending, TF-IDF + cosine similarity graph |
| `src/data_pipeline.py` | RSS ingestion via feedparser, HTML extraction via trafilatura, regex cleaning |
| `src/trainer.py` | Hyperparameter grid search (`tune` method) over symbolic model params |
| `src/dataset_loader.py` | HuggingFace `datasets` loader for CNN/DailyMail, seeded sampling |
| `src/evaluator.py` | `RougeEvaluator` — ROUGE-1/2/L F1 evaluation |
| `api.py` | FastAPI + Uvicorn backend serving summarization endpoints |
| `frontend/index.html` | Vanilla HTML/CSS/JS dashboard with keyword trend comparison |
| `requirements.txt` | Python dependencies |
| `tests/test_pipeline.py` | Only 2 normalization tests — effectively zero coverage |
| `architecture.drawio` | Architecture diagram (claims features NOT in code) |
| `Component interaction-Diagram.drawio` | Component interaction diagram |
| `Model View Controller Architecture Diagram.drawio` | MVC diagram |

## Current tech stack (implemented)

- **Language**: Python
- **NLP**: NLTK (sentence tokenization), scikit-learn (TF-IDF + cosine similarity), NumPy (PageRank/matrix ops)
- **Evaluation**: rouge-score
- **Data**: HuggingFace datasets (CNN/DailyMail), feedparser (RSS), trafilatura (article extraction)
- **Backend**: FastAPI + Uvicorn
- **Frontend**: Vanilla HTML/CSS/JS
- **Visualization**: Graphviz (auto-generated architecture diagrams)
- **Data export**: Pandas (CSV in fetch script)

## Current techniques (implemented)

- Extractive summarization via TextRank (custom PageRank over TF-IDF cosine similarity graph)
- MMR (Maximal Marginal Relevance) selection to reduce redundancy
- Centroid relevance blending (`blend_alpha` parameter)
- Hyperparameter grid search for model tuning
- ROUGE-1/2/L F1 evaluation
- Seeded sampling for reproducibility
- RSS + HTML extraction + regex cleaning pipeline
- Client-side keyword trend comparison in frontend

---

## Critical shortfalls (from review)

### 1. Scope gap — documented vs implemented
Architecture diagrams claim these features that **DO NOT EXIST in code**:
- Domain classifier
- Grouping/clustering
- Grounding evidence module
- Persona weighting
- Word budget control
- Output formatter
- Broader datasets (XSum, MultiNews, AG News)

The project title says "User-Adaptive" but nothing in the code adapts to users.

### 2. Testing — nearly zero
- `python3 -m unittest discover -s tests -v` fails (missing feedparser dep)
- Only 2 normalization tests in `tests/test_pipeline.py`
- No API endpoint tests, no integration tests, no eval regression tests

### 3. Evaluation artifacts missing
- No saved evaluation results committed
- No comparison baselines
- No reproducible evaluation script that runs end-to-end

### 4. Engineering gaps
- No Dockerfile, no containerization
- No CI/CD pipeline
- No type hints or linting config
- Frontend is a single monolithic HTML file

---

## Upgrade plan — transform B → A+ portfolio project

### Owner's hardware
- **GPU**: NVIDIA RTX 3090 (24GB VRAM)
- **Cluster**: k3s (lightweight Kubernetes)
- **Goal**: Deploy vLLM on k3s, use it for abstractive summarization

### Phase 1: vLLM on k3s (infrastructure)

**What to build:**
- Install NVIDIA GPU Operator on k3s cluster
- Deploy vLLM as a Kubernetes Deployment (model: `mistralai/Mistral-7B-Instruct-v0.3` or `Qwen/Qwen2.5-7B-Instruct`)
- Create: PVC for model cache, HF token Secret, Service, Ingress
- Add Prometheus + Grafana for inference observability
- Write Helm chart for the entire stack

**Key k3s manifests needed:**
- `k8s/vllm-deployment.yaml` — vLLM pod with GPU resource limits
- `k8s/vllm-service.yaml` — ClusterIP service on port 8000
- `k8s/api-deployment.yaml` — FastAPI summarizer backend
- `k8s/frontend-deployment.yaml` — Next.js frontend
- `k8s/ingress.yaml` — Traefik routing rules
- `k8s/monitoring/` — Prometheus ServiceMonitor + Grafana dashboards

**RTX 3090 model fit (24GB VRAM):**
- Mistral 7B Instruct @ FP16 → ~14GB, leaves ~10GB for KV cache (good for ~50 concurrent requests)
- Qwen2.5-7B-Instruct @ FP16 → ~14GB, same headroom
- Mistral Small 3 @ Q4 → ~14-15GB, better quality but quantized
- Any 32B model @ Q4 → ~19-20GB, tight but workable for demos

### Phase 2: Hybrid summarization pipeline (the novel core)

**Refactor `src/summarizer_model.py` into a 3-stage pipeline:**

```
Stage 1: EXTRACT (keep existing TextRank+MMR)
    → Input: raw article text
    → Output: top-k salient sentences (configurable k)

Stage 2: ABSTRACT (NEW — calls vLLM)
    → Input: extracted sentences + user persona prompt
    → Output: fluent abstractive summary
    → Uses OpenAI-compatible client pointing at vLLM service

Stage 3: VERIFY (NEW — factual consistency)
    → Input: source article + generated summary
    → Output: summary + confidence score + flagged entities
    → Uses spaCy NER to compare entity sets
```

**New files to create:**
- `src/pipeline.py` — orchestrates the 3 stages
- `src/abstractor.py` — vLLM client wrapper, prompt templates, persona system
- `src/verifier.py` — NER-based factual consistency checker
- `src/personas.py` — persona definitions (technical, casual, executive, academic)

**API modes** (add `mode` param to endpoint):
- `extractive` — current behavior (backward compatible)
- `abstractive` — LLM-only summarization
- `hybrid` — extract → abstract → verify (default, flagship)

### Phase 3: Make it actually user-adaptive

**Persona system** (delivers on the project title):
- `technical` — preserves domain jargon, includes statistics, structured output
- `casual` — plain language, shorter, conversational tone
- `executive` — bullet-point conclusions, action items, key metrics
- `academic` — formal register, citations preserved, methodology noted

Each persona = a different system prompt template sent to vLLM. Trivial to implement, huge impact on the "adaptive" claim.

**Length control**: 1-sentence, paragraph (3-5 sentences), detailed (full summary). Maps to `max_tokens` on vLLM + TextRank `top_k` adjustment.

**Domain detection**: Zero-shot classification via vLLM prompt ("Classify this article as: tech, politics, science, sports, business, health"). Adjusts extraction parameters and prompt template per domain.

### Phase 4: Modern frontend with streaming

**Replace** `frontend/index.html` **with Next.js app** (or React + Vite):
- SSE (Server-Sent Events) streaming — summary writes itself token by token
- Side-by-side comparison view: extractive vs hybrid output
- Persona selector (dropdown/toggle)
- Length slider
- Inference metrics panel (tok/s, latency, GPU %, pulled from Prometheus)
- Dark/light mode, responsive, modern design

**FastAPI streaming endpoint:**
```python
@app.get("/summarize/stream")
async def stream_summary(url: str, persona: str = "casual"):
    async def generate():
        async for token in abstractor.stream(extracted_sentences, persona):
            yield f"data: {json.dumps({'token': token})}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")
```

### Phase 5: Testing & CI/CD (grade-saver)

**Testing targets (aim for >80% coverage):**
- `tests/test_pipeline.py` — expand normalization tests, add extraction tests
- `tests/test_summarizer.py` — TextRank scoring, MMR selection, edge cases
- `tests/test_abstractor.py` — mock vLLM client, test prompt formatting
- `tests/test_verifier.py` — NER entity comparison logic
- `tests/test_api.py` — FastAPI endpoint tests with `httpx.AsyncClient`
- `tests/test_personas.py` — persona prompt generation
- `tests/conftest.py` — shared fixtures, sample articles, mock vLLM responses

**CI/CD pipeline (.github/workflows/ci.yml):**
```yaml
- Lint: ruff check .
- Type check: mypy src/
- Test: pytest --cov=src --cov-fail-under=80
- Build: docker build
- ROUGE regression: run eval on held-out set, fail if ROUGE-L drops >2%
```

**Docker:**
- `Dockerfile.api` — FastAPI backend
- `Dockerfile.frontend` — Next.js app
- `docker-compose.yml` — local dev stack (API + frontend + vLLM mock)

**GitOps (stretch goal):**
- ArgoCD watching the repo, auto-deploys to k3s on merge to main

---

## Dependency upgrades

| DROP | REPLACE WITH | REASON |
|---|---|---|
| NLTK (sole tokenizer) | spaCy `en_core_web_sm` | Faster, gives NER for free (needed for verifier), better sentence segmentation |
| scikit-learn TF-IDF only | Keep TF-IDF + ADD `sentence-transformers` (`all-MiniLM-L6-v2`) | Semantic similarity for better TextRank graph edges |
| rouge-score only | ADD `bert-score`, keep rouge-score | BERTScore captures semantic similarity ROUGE misses |
| No HTTP client for LLM | `openai` Python SDK (OpenAI-compatible) | vLLM exposes OpenAI-compatible API, use official SDK |
| No containerization | Docker + Helm | Everything becomes a k3s pod |
| No linting | `ruff` + `mypy` | Fast linting + type safety |
| No testing framework | `pytest` + `pytest-cov` + `pytest-asyncio` | Modern Python testing |

## Updated requirements.txt (target)

```
# Core NLP
spacy>=3.7
scikit-learn>=1.4
numpy>=1.26
sentence-transformers>=2.7

# Evaluation
rouge-score>=0.1.2
bert-score>=0.3.13

# Data pipeline
feedparser>=6.0
trafilatura>=1.8
datasets>=2.19

# LLM client
openai>=1.30

# API
fastapi>=0.111
uvicorn[standard]>=0.29
sse-starlette>=2.0

# Testing
pytest>=8.2
pytest-cov>=5.0
pytest-asyncio>=0.23
httpx>=0.27

# Dev tools
ruff>=0.4
mypy>=1.10
```

---

## File structure (target)

```
├── PLANS.md                   ← this file
├── Dockerfile.api
├── Dockerfile.frontend
├── docker-compose.yml
├── Makefile                   ← make dev / make test / make deploy
├── pyproject.toml             ← replaces requirements.txt, configures ruff/mypy/pytest
├── src/
│   ├── __init__.py
│   ├── summarizer_model.py    ← KEEP — TextRank+MMR extractor (refactored)
│   ├── pipeline.py            ← NEW — 3-stage orchestrator
│   ├── abstractor.py          ← NEW — vLLM client + prompt templates
│   ├── verifier.py            ← NEW — NER factual consistency
│   ├── personas.py            ← NEW — persona definitions
│   ├── data_pipeline.py       ← KEEP — RSS + HTML extraction
│   ├── dataset_loader.py      ← KEEP — HF dataset loading
│   ├── evaluator.py           ← EXPAND — add BERTScore, comparative eval
│   └── trainer.py             ← KEEP — grid search
├── api.py                     ← EXPAND — add streaming endpoint, mode param
├── frontend/                  ← REPLACE — Next.js app
│   ├── package.json
│   ├── app/
│   │   ├── page.tsx
│   │   └── components/
│   └── ...
├── k8s/                       ← NEW — Kubernetes manifests
│   ├── namespace.yaml
│   ├── vllm-deployment.yaml
│   ├── vllm-service.yaml
│   ├── api-deployment.yaml
│   ├── frontend-deployment.yaml
│   ├── ingress.yaml
│   └── monitoring/
│       ├── prometheus-config.yaml
│       └── grafana-dashboard.json
├── helm/                      ← NEW — Helm chart
│   └── summarizer/
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
├── tests/
│   ├── conftest.py
│   ├── test_pipeline.py       ← EXPAND
│   ├── test_summarizer.py     ← NEW
│   ├── test_abstractor.py     ← NEW
│   ├── test_verifier.py       ← NEW
│   ├── test_api.py            ← NEW
│   └── test_personas.py       ← NEW
├── eval/                      ← NEW — evaluation scripts + results
│   ├── run_eval.py
│   ├── results/
│   └── baselines/
├── .github/
│   └── workflows/
│       └── ci.yml             ← NEW — CI pipeline
└── docs/
    ├── architecture.md        ← REPLACE drawio with living doc
    └── api-reference.md
```

---

## Quick start

```bash
# Verify code compiles
python -m compileall -q src/ api.py

# Run tests
python -m pytest tests/ -v --tb=short

# Start the API
uvicorn api:app --reload
```

---

## Priority order for implementation

1. **Testing scaffold** — write tests for existing code first (establishes baseline, catches regressions during refactor)
2. **Pipeline refactor** — split monolithic summarizer into extract/abstract/verify stages
3. **Abstractor + vLLM integration** — connect to vLLM via OpenAI SDK
4. **Persona system** — prompt templates per persona
5. **Verifier** — NER-based fact checking
6. **Streaming API** — SSE endpoint for token-by-token output
7. **Frontend** — Next.js with streaming UI
8. **Dockerfiles** — containerize API and frontend
9. **k8s manifests** — deploy everything to k3s
10. **CI/CD** — GitHub Actions pipeline
11. **Evaluation** — comparative eval (extractive vs hybrid vs pure LLM) with ROUGE + BERTScore
12. **Helm chart + monitoring** — production polish

---

## Success criteria

When done, you should be able to:
1. `git push` → GitHub Actions runs lint + tests + builds containers
2. `helm install` → deploys vLLM + API + frontend to k3s
3. Open browser → paste article URL → pick persona → watch summary stream in
4. See extractive vs hybrid comparison side-by-side
5. Show Grafana dashboard with inference metrics
6. Run `make eval` → produces comparative ROUGE + BERTScore report
7. All tests pass with >80% coverage
