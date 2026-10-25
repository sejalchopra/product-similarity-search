.PHONY: lint test build run-local index eval eval-smoke eval-medium promote-smoke bench demo-warm

PYTHON ?= python3
ifeq ($(wildcard .venv/bin/python),)
  PY := $(PYTHON)
else
  PY := .venv/bin/python
endif

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .
	$(PY) -m mypy --strict service app.py

test:
	$(PY) -m pytest -q

build:
	docker build -t product-similarity:local .

run-local:
	PYTHONPATH=. $(PY) -m uvicorn app:app --host 0.0.0.0 --port 8000

index:
	PYTHONPATH=. $(PY) scripts/build_index.py

index-metadata:
	PYTHONPATH=. $(PY) scripts/build_index.py --metadata-only

eval:
	PYTHONPATH=. $(PY) scripts/evaluate.py

eval-smoke:
	EVAL_SMOKE=1 EVAL_MAX_TUNE=400 EVAL_MAX_VAL=200 EVAL_MAX_TEST=200 PYTHONPATH=. $(PY) scripts/evaluate.py

EVAL_TUNE_QUERIES ?= 1500
EVAL_VAL_QUERIES ?= 1500
EVAL_TEST_QUERIES ?= 2000

eval-medium:
	@mkdir -p logs
	EVAL_MEDIUM=1 EVAL_TUNE_QUERIES=$(EVAL_TUNE_QUERIES) EVAL_VAL_QUERIES=$(EVAL_VAL_QUERIES) EVAL_TEST_QUERIES=$(EVAL_TEST_QUERIES) PYTHONPATH=. $(PY) scripts/evaluate.py

promote-smoke:
	CONFIRM=1 PYTHONPATH=. $(PY) scripts/promote_smoke.py

bench:
	PYTHONPATH=. $(PY) scripts/bench_hnsw.py

demo-warm:
	PYTHONPATH=. $(PY) scripts/demo_warm.py $(DEMO_BASE)
