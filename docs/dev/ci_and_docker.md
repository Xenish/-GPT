# CI and Docker Guide

## Overview

This document explains the CI/CD pipeline and Docker deployment options for FinanTradeAlgo.

## CI Pipeline

### GitHub Actions Workflow

The project uses GitHub Actions for continuous integration (`.github/workflows/ci.yml`).

**Jobs:**

1. **Backend Job**
   - Python 3.11 on Ubuntu
   - Installs dependencies from `requirements.txt`
   - Runs fast tests: `pytest -q -m "not slow"`
   - Excludes slow integration tests to keep CI fast (<2 min)

2. **Frontend Job**
   - Node 20 on Ubuntu
   - Installs npm dependencies
   - Runs linter: `npm run lint`
   - Builds production bundle: `npm run build`

**Test Markers:**

- Fast tests: Run on every commit/PR (default)
- Slow tests: Marked with `@pytest.mark.slow` (3 tests)
  - `test_regression_backtests.py`
  - `test_rule_vs_ml_integration.py`
  - `test_live_replay.py`

**Running tests locally:**

```bash
# CI mode (fast tests only)
make test-backend-ci
# or
pytest -q -m "not slow"

# All tests (including slow)
make test-backend
# or
pytest -v

# Slow tests only
make test-backend-slow
# or
pytest -v -m slow
```

### CI Configuration

Python version is standardized to 3.11 across:
- `Dockerfile.api` (base image: `python:3.11-slim`)
- `.github/workflows/ci.yml` (python-version: "3.11")
- `pyproject.toml` (requires-python: ">=3.11")

## Docker Deployment

### Two-Container Setup (Default)

**Architecture:**
```
┌──────────────────────────────────────┐
│  Host Machine                        │
│  ┌────────────────┐ ┌──────────────┐│
│  │  API Container │ │  Frontend    ││
│  │  FastAPI       │ │  Container   ││
│  │  Port: 8000    │ │  Next.js     ││
│  │                │ │  Port: 3000  ││
│  └────────────────┘ └──────────────┘│
│          ▲                  ▲        │
│          │                  │        │
└──────────┼──────────────────┼────────┘
           │                  │
      localhost:8000    localhost:3000
```

**Files:**
- `docker-compose.yml` - Orchestrates both containers
- `Dockerfile.api` - Python 3.11 backend
- `Dockerfile.frontend` - Node 20 frontend

**Usage:**

```bash
# Start both services
make docker-up
# or
docker-compose up -d

# View logs
make docker-logs
# or
docker-compose logs -f

# Stop services
make docker-down
# or
docker-compose down

# Rebuild images
make docker-build
# or
docker-compose build
```

**Accessing services:**
- API: http://localhost:8000/docs (Swagger)
- Frontend: http://localhost:3000

**Volumes:**
- `./data:/app/data` - Market data persistence
- `./outputs:/app/outputs` - Backtest results, ML models, trades

**Environment variables:**
- Backend: Reads from `.env` file (not committed)
- Frontend: `NEXT_PUBLIC_API_BASE_URL=http://finantrade_api:8000`

### Single-Container Setup (Optional)

For the single-container option with Nginx + Supervisor, see `docs/dev/docker_single_container.md`.

## Environment Variables

### Backend (.env)
```bash
BINANCE_FUTURES_API_KEY=your_key
BINANCE_FUTURES_API_SECRET=your_secret
PYTHONUNBUFFERED=1
```

### Frontend (Next.js)
```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000  # Development
# or
NEXT_PUBLIC_API_BASE_URL=http://finantrade_api:8000  # Docker
```

**Important:**
- `.env` and `.env.local` are gitignored (never commit secrets)
- `.env.example` is committed as a template
- Docker Compose injects `NEXT_PUBLIC_API_BASE_URL` automatically

## Local Development

### Without Docker

```bash
# Terminal 1: Backend
python scripts/run_api.py
# or
uvicorn finantradealgo.api.server:create_app --factory --reload

# Terminal 2: Frontend
cd frontend/web
npm run dev
```

### With Docker

```bash
# Start everything
make docker-up

# Tail logs
make docker-logs

# Restart after code changes
make docker-build && make docker-up
```

## Makefile Commands

```bash
make help              # Show all available commands
make test-backend-ci   # Run CI tests
make docker-up         # Start Docker services
make docker-down       # Stop Docker services
make clean             # Remove build artifacts
```

## Troubleshooting

### CI Failures

1. **Backend tests fail:**
   - Check Python version is 3.11
   - Verify `requirements.txt` is up to date
   - Run locally: `pytest -q -m "not slow"`

2. **Frontend build fails:**
   - Check Node version is 20
   - Verify `package-lock.json` is committed
   - Run locally: `cd frontend/web && npm run build`

### Docker Issues

1. **Port already in use:**
   ```bash
   # Change ports in docker-compose.yml
   ports:
     - "8001:8000"  # API
     - "3001:3000"  # Frontend
   ```

2. **Volumes not persisting:**
   - Check `./data` and `./outputs` directories exist
   - Verify permissions on Windows (may need Docker Desktop settings)

3. **Frontend can't reach API:**
   - Check `NEXT_PUBLIC_API_BASE_URL` in docker-compose.yml
   - Verify API container is running: `docker ps`
   - Check network: `docker network inspect tradeproject_default`

## Best Practices

1. **Always run CI tests before pushing:**
   ```bash
   make test-backend-ci
   ```

2. **Keep .env.example updated** when adding new variables

3. **Use Makefile shortcuts** for consistency across team

4. **Never commit secrets** - verify with `git status` before committing

5. **Test Docker builds locally** before deploying:
   ```bash
   make docker-build
   make docker-up
   ```

## References

- [pytest markers documentation](https://docs.pytest.org/en/stable/example/markers.html)
- [Docker Compose documentation](https://docs.docker.com/compose/)
- [Next.js environment variables](https://nextjs.org/docs/pages/building-your-application/configuring/environment-variables)
