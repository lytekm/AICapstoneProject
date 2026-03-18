.PHONY: test lint typecheck dev docker-build docker-run clean

test:
	python -m pytest tests/ --cov=src --cov-report=term-missing -v

lint:
	python -m ruff check src/ api.py tests/
	python -m mypy src/ api.py --ignore-missing-imports

typecheck:
	python -m mypy src/ api.py --ignore-missing-imports

dev:
	uvicorn api:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t summarizer-api .

docker-run:
	docker compose up -d

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
