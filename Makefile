# Canon. Everything here runs on free tiers; reproducing the benchmark costs $0.
#
# Tabs matter in a Makefile. If you are editing this, keep the recipe indentation as tabs.

.PHONY: help data dryrun bench demo test lint gold tune resume clean-runs

help:
	@echo "Canon -- make targets"
	@echo
	@echo "  make data      download Olist and build data/canon.duckdb (idempotent)"
	@echo "  make gold      execute every gold SQL and record its value"
	@echo "  make dryrun    project tokens, days and wall-clock. Contacts NO provider."
	@echo "  make bench     run the benchmark. Checkpoints; resumable across days."
	@echo "  make resume    continue an interrupted sweep (same tag)"
	@echo "  make tune      tune the refusal thresholds on the held-out dev set"
	@echo "  make demo      Streamlit demo"
	@echo "  make test      full test suite (zero LLM calls)"
	@echo "  make lint      ruff check + format"
	@echo
	@echo "Always run 'make dryrun' before 'make bench'."

data:
	uv run python scripts/load_data.py

gold:
	uv run python scripts/verify_gold.py

dryrun:
	uv run python scripts/dry_run.py

# One run of both systems over all 80 questions. Checkpoints after every question, so
# quota exhaustion ends the session rather than the run -- re-run to continue.
bench:
	uv run python -m benchmark.harness.run --tag sweep-groq-r1 --runs 1

resume: bench

tune:
	uv run python scripts/tune_thresholds.py

demo:
	uv run streamlit run app.py

test:
	uv run pytest

lint:
	uv run ruff check --fix .
	uv run ruff format .

clean-runs:
	rm -rf benchmark/runs
