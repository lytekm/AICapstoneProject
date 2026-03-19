# User-Adaptive Summarization

A three-stage NLP pipeline that produces persona-aware summaries of news articles, with extractive, abstractive, and hybrid modes. Built with FastAPI, spaCy, and an OpenAI-compatible LLM backend.

```
Raw Article Text
      |
[Stage 1] Extractive (TextRank + MMR)
      |     sentence segmentation, TF-IDF, graph ranking, diversity selection
      |
[Stage 2] Abstractive (LLM via OpenAI SDK)
      |     persona prompt + extracted sentences -> fluent rewrite
      |
[Stage 3] Verification (spaCy NER)
            compare named entities in source vs summary -> confidence score
```

## Features

- **Three pipeline modes**: extractive (TextRank+MMR only), abstractive (LLM rewrite), hybrid (all three stages)
- **Persona system**: technical, casual, executive, academic profiles shape the LLM prompt and output style
- **Length control**: brief, standard, detailed options that scale the LLM token budget
- **SSE streaming**: token-by-token delivery for abstractive/hybrid via `EventSource`
- **NER verification**: spaCy-based factual consistency check with confidence scoring and flagged entity reporting
- **ROUGE evaluation**: reproducible eval script on CNN/DailyMail test split
- **Mock-first design**: runs fully offline with `MockAbstractor`; real LLM opt-in via env vars

---

## Quick Start

```bash
# 1. clone and install
git clone https://github.com/ixxet/User-Adaptive-Summarization_COMP385-402_Group-4_Winter2026.git
cd User-Adaptive-Summarization_COMP385-402_Group-4_Winter2026
pip install -e ".[dev]"
python -m spacy download en_core_web_sm

# 2. start the dev server
make dev

# 3. open the frontend
#    http://localhost:8000/frontend/index.html
```

## Running the Project

### Development server

```bash
make dev
```

Starts `uvicorn` on port 8000 with hot-reload. The API serves both REST endpoints and the static frontend.

### Test suite

```bash
make test          # run all 168 tests with coverage report
make lint          # ruff lint + mypy type check
make typecheck     # mypy only
```

### ROUGE evaluation

```bash
make eval                              # default: 50 samples, seed=42
python -m eval.run_eval --samples 100  # custom sample count
python -m eval.run_eval --output results/my_run.json
```

Downloads CNN/DailyMail test split on first run, then caches locally.

### Docker

```bash
make docker-build   # build the container image
make docker-run     # start via docker-compose on port 8000
```

---

## Project Structure

```
.
├── api.py                    FastAPI backend (REST + SSE endpoints)
├── Makefile                  dev, test, lint, eval, docker targets
├── pyproject.toml            build config, tool settings
├── requirements.txt          pinned dependencies
│
├── src/
│   ├── summarizer_model.py   TextRank + MMR extractive engine
│   ├── pipeline.py           three-stage orchestrator + SSE streaming
│   ├── personas.py           persona definitions and prompt builder
│   ├── abstractor.py         LLM abstraction layer (mock + real)
│   ├── verifier.py           NER-based factual consistency checker
│   ├── evaluator.py          ROUGE scoring
│   ├── dataset_loader.py     CNN/DailyMail dataset loader
│   ├── trainer.py            hyperparameter tuning for extractive config
│   └── data_pipeline.py      RSS ingestion + text normalization
│
├── frontend/
│   └── index.html            vanilla JS frontend with SSE streaming
│
├── eval/
│   ├── run_eval.py           reproducible evaluation CLI
│   └── results/              saved evaluation outputs (JSON)
│
├── tests/                    168 tests, 88% coverage
│   ├── conftest.py           shared fixtures and sample data
│   ├── test_summarizer.py    TextRank scoring, MMR selection, edge cases
│   ├── test_summarization_pipeline.py   all 3 modes + streaming + errors
│   ├── test_abstractor.py    mock/real abstractor, config, streaming
│   ├── test_verifier.py      NER extraction, confidence, graceful fallback
│   ├── test_personas.py      persona definitions, prompt formatting
│   ├── test_api.py           all endpoints including SSE stream
│   ├── test_eval.py          evaluation script and CLI
│   ├── test_data_pipeline.py normalization, tokenization, RSS fetch
│   ├── test_dataset_loader.py CNN/DailyMail loader
│   └── test_trainer.py       hyperparameter tuning
│
├── .github/workflows/
│   └── ci.yml                lint + test on push (GitHub Actions)
│
├── Dockerfile                multi-stage build for the API
├── docker-compose.yml        single-service local deploy
└── PLANS.md                  project context and upgrade roadmap
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/articles` | Fetch CBC RSS articles |
| `GET` | `/api/personas` | List available persona names |
| `POST` | `/api/summarize` | Summarize an article (JSON body) |
| `GET` | `/api/summarize/stream` | SSE streaming summary (query params) |

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

Response includes `summary`, `mode`, `persona`, and for hybrid/abstractive modes: `confidence` (0-1) and `flagged_entities`.

### GET /api/summarize/stream

Query params: `url`, `k`, `mode`, `persona`, `length`.

Returns `text/event-stream` with three event types:
- `event: meta` - pipeline mode and persona
- `event: token` - individual tokens as they generate
- `event: done` - final summary with confidence and flagged entities

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_MOCK_LLM` | `1` | Set to `0` to use a real LLM backend |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible endpoint for the LLM |
| `VLLM_API_KEY` | `EMPTY` | API key for the LLM backend |
| `RSS_FEED_URL` | CBC Business RSS | Default RSS feed for article fetching |

---

## Extractive Algorithm

The extractive stage uses TextRank for importance scoring with cosine similarity on TF-IDF vectors, blended with centroid similarity for stability on short articles. MMR (Maximal Marginal Relevance) selects the final top-k sentences to balance relevance against diversity.

| Parameter | Default | Effect |
|-----------|---------|--------|
| `mmr_lambda` | 0.75 | Higher = less redundancy penalty |
| `blend_alpha` | 0.7 | Higher = more TextRank influence |
| `textrank_min_edge` | 0.1 | Higher = sparser similarity graph |

---

## License

COMP385-402 Capstone Project, Centennial College, Winter 2026.
