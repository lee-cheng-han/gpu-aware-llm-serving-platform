.PHONY: install install-dev run test test-model lint typecheck evaluation docker-build docker-build-cuda docker-run deploy-kind benchmark-sanity benchmark-no-batching benchmark-dynamic-batch
install:
	python -m pip install -r requirements.txt
install-dev:
	python -m pip install -r requirements-dev.txt
run:
	uvicorn apps.gateway.main:app --host 0.0.0.0 --port 8000
test:
	pytest -q
test-model:
	RUN_MODEL_TESTS=1 pytest -m model -q
lint:
	ruff check .
typecheck:
	mypy
evaluation:
	python -m benchmark.run_simulated_evaluation --check
docker-build:
	docker build -t gpu-aware-llm-serving-platform:local .
docker-build-cuda:
	docker build -f Dockerfile.cuda -t gpu-aware-llm-serving-platform:cuda .
docker-run:
	docker run --rm -p 8000:8000 gpu-aware-llm-serving-platform:local
deploy-kind:
	./scripts/deploy_local_kind.sh
benchmark-sanity:
	python -m benchmark.sanity_model_benchmark --models sshleifer/tiny-gpt2 gpt2
benchmark-no-batching:
	python -m benchmark.compare_schedulers --policy no_batching
benchmark-dynamic-batch:
	python -m benchmark.compare_schedulers --policy dynamic_batch
