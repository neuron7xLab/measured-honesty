PY ?= python3

.PHONY: verify test lint type report viz snli spam clean

# Pure tasks (T1/T3/T5) run anywhere; T2/T4 gated behind AOS_LLM_TESTS=1 + stack.
verify: lint type test report
	@echo "VERIFY: complete"

viz:
	$(PY) -m mh.viz

snli:
	$(PY) -m mh.benchmark_snli

spam:
	$(PY) -m mh.benchmark_spam

lint:
	$(PY) -m ruff check mh tests
	$(PY) -m ruff format --check mh tests

type:
	$(PY) -m mypy mh

test:
	$(PY) -m pytest tests/ -q

report:
	$(PY) -m mh.run_all

clean:
	rm -rf artifacts .pytest_cache .ruff_cache .mypy_cache .hypothesis
