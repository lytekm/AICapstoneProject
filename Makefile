.PHONY: test lint typecheck dev dev-mock dev-real tunnel-real docker-build docker-run clean eval

PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
HOST ?= 127.0.0.1
PORT ?= 8000
VLLM_BASE_URL ?= http://192.168.2.205:8000/v1
VLLM_MODEL ?= mistralai/Mistral-7B-Instruct-v0.3
VLLM_API_KEY ?= EMPTY

test:
	$(PYTHON) -m pytest tests/ --cov=src --cov-report=term-missing -v

lint:
	$(PYTHON) -m ruff check src/ api.py tests/
	$(PYTHON) -m mypy src/ api.py --ignore-missing-imports

typecheck:
	$(PYTHON) -m mypy src/ api.py --ignore-missing-imports

dev:
	@$(MAKE) dev-mock

dev-mock:
	USE_MOCK_LLM=1 \
	$(PYTHON) -m uvicorn api:app --reload --host $(HOST) --port $(PORT)

dev-real:
	USE_MOCK_LLM=0 \
	VLLM_BASE_URL=$(VLLM_BASE_URL) \
	VLLM_MODEL=$(VLLM_MODEL) \
	VLLM_API_KEY=$(VLLM_API_KEY) \
	$(PYTHON) -m uvicorn api:app --reload --host $(HOST) --port $(PORT)

tunnel-real:
	@echo "Opening a quick tunnel for http://$(HOST):$(PORT)"
	@echo "Use this only after starting the app with 'make dev-real'."
	@echo "Verification step: run one abstractive or hybrid summarize call and confirm the result does not begin with '[Mock Summary]'."
	cloudflared tunnel --url http://$(HOST):$(PORT)

docker-build:
	docker build -t summarizer-api .

docker-run:
	docker compose up -d

eval:
	$(PYTHON) -m eval.run_eval

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
