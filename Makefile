.PHONY: install install-dev install-observability run test test-model test-gpu test-otel lint typecheck evaluation evaluation-token-policies docker-build docker-build-cuda docker-run deploy-kind benchmark-sanity benchmark-token-loop benchmark-no-batching benchmark-dynamic-batch benchmark-sustained-load
install:
	python -m pip install -r requirements.txt
install-dev:
	python -m pip install -r requirements-dev.txt
install-observability:
	python -m pip install -r requirements-observability.txt
run:
	uvicorn apps.gateway.main:app --host 0.0.0.0 --port 8000
test:
	pytest -q
test-model:
	RUN_MODEL_TESTS=1 pytest -m model -q
test-gpu:
	RUN_MODEL_TESTS=1 RUN_GPU_TESTS=1 pytest -m gpu -q
test-otel:
	RUN_OTEL_TESTS=1 pytest -m otel -q
lint:
	ruff check .
typecheck:
	mypy
evaluation:
	python -m benchmark.run_simulated_evaluation --check
evaluation-token-policies:
	python -m benchmark.compare_token_scheduling_policies
docker-build:
	docker build -t gpu-aware-llm-serving-platform:local .
docker-build-cuda:
	docker build -f Dockerfile.cuda -t gpu-aware-llm-serving-platform:cuda .
docker-run:
	docker run --rm -p 8000:8000 gpu-aware-llm-serving-platform:local
deploy-kind:
	./scripts/deploy_local_kind.sh
benchmark-sanity:
	python -m benchmark.sanity_model_benchmark --models HuggingFaceTB/SmolLM2-135M-Instruct
benchmark-token-loop:
	python -m benchmark.compare_token_backends
benchmark-no-batching:
	python -m benchmark.compare_schedulers --policy no_batching
benchmark-dynamic-batch:
	python -m benchmark.compare_schedulers --policy dynamic_batch
benchmark-sustained-load:
	python -m benchmark.run_sustained_load --duration-seconds 60 --concurrency 4
