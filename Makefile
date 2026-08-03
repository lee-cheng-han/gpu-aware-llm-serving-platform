.PHONY: install install-dev run test test-model lint typecheck docker-build docker-run benchmark-sanity benchmark-no-batching benchmark-dynamic-batch
install:
	python -m pip install -r requirements.txt
install-dev:
	python -m pip install -r requirements-dev.txt
run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000
test:
	pytest -q
test-model:
	RUN_MODEL_TESTS=1 pytest -m model -q
lint:
	ruff check .
typecheck:
	mypy
docker-build:
	docker build -t llm-inference-scheduler:local .
docker-run:
	docker run --rm -p 8000:8000 llm-inference-scheduler:local
benchmark-sanity:
	python benchmark/sanity_model_benchmark.py --models sshleifer/tiny-gpt2 gpt2
benchmark-no-batching:
	python benchmark/compare_schedulers.py --policy no_batching
benchmark-dynamic-batch:
	python benchmark/compare_schedulers.py --policy dynamic_batch
