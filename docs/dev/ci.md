## CI Contract (V1)

Current GitHub Actions workflow runs the following checks on pushes/PRs:

1) Lint job: `ruff check finantradealgo services tests scripts` and `black --check finantradealgo services tests scripts`.
2) Typecheck job: `mypy --config-file mypy.ini finantradealgo/system finantradealgo/risk finantradealgo/live_trading`.
3) Backend job (after lint + typecheck):
   - `pytest -m "not slow" --cov=finantradealgo --cov=services --cov-report=xml --cov-report=term-missing --cov-fail-under=60`
   - `python scripts/check_config_sanity.py`
   - `python scripts/check_strategy_dependency.py`
   - `python scripts/check_research_imports.py`
   - CLI smoke tests: `pytest -q tests/test_run_*_cli_*.py tests/test_run_test_risk_overlays_rg1.py`
   - Coverage XML is uploaded as an artifact.
   - On failure, selected `outputs/**/*.json|csv|parquet|log` are uploaded for debugging.
4) Frontend job: `npm run lint --if-present` and `npm run build` in `frontend/web`.
5) DB integration job (runs only if `FT_TIMESCALE_DSN` or `FT_POSTGRES_DSN` secrets are set):
   - `alembic history -q` sanity
   - `pytest -m "db"` (db-marked tests; otherwise skipped in default CI)

All steps must pass for CI to succeed. Extend this contract and document changes here when new checks are added.

Notes:
- Security scans (e.g., `pip-audit`, `bandit`) are not yet enforced in V1; consider adding them in a future RS milestone.
- DB-marked tests are skipped in the normal backend job; the separate db-integration job runs only when DSN secrets are available.
- Branch protection: main/master should require the following status checks to pass before merge:
  - Lint (ruff + black)
  - Typecheck (mypy)
  - Backend tests + coverage
  - Config/risk guardrails + CLI smoke
