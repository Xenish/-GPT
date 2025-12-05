# Makefile for finantradealgo

.PHONY: help test-backend test-backend-ci test-backend-slow build-frontend lint-frontend docker-up docker-down docker-build clean

help:
	@echo "FinanTradeAlgo Development Commands"
	@echo "===================================="
	@echo ""
	@echo "Backend:"
	@echo "  make test-backend         - Run all backend tests (including slow)"
	@echo "  make test-backend-ci      - Run fast tests only (CI mode)"
	@echo "  make test-backend-slow    - Run slow integration tests only"
	@echo "  make lint-backend         - Run Python linting (if configured)"
	@echo ""
	@echo "Frontend:"
	@echo "  make build-frontend       - Build Next.js frontend"
	@echo "  make lint-frontend        - Lint frontend code"
	@echo "  make dev-frontend         - Run frontend in dev mode"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up            - Start docker-compose services"
	@echo "  make docker-down          - Stop docker-compose services"
	@echo "  make docker-build         - Rebuild docker images"
	@echo "  make docker-logs          - Tail docker logs"
	@echo ""
	@echo "Single Container Docker:"
	@echo "  make docker-single-build  - Build single container image"
	@echo "  make docker-single-up     - Start single container"
	@echo "  make docker-single-down   - Stop single container"
	@echo "  make docker-single-logs   - Tail single container logs"
	@echo "  make docker-single-status - Show supervisor process status"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean                - Remove build artifacts and caches"

# Backend tests
test-backend:
	pytest -v

test-backend-ci:
	pytest -q -m "not slow and not db"

test-backend-slow:
	pytest -v -m slow

db-tests:
	pytest -v -m "db"

db-migrate:
	alembic upgrade head

lint-backend:
	@echo "Python linting not configured yet"
	# Add ruff, black, mypy, etc. when ready

# Frontend commands
build-frontend:
	cd frontend/web && npm run build

lint-frontend:
	cd frontend/web && npm run lint

dev-frontend:
	cd frontend/web && npm run dev

# Docker commands
docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-build:
	docker-compose build

docker-logs:
	docker-compose logs -f

docker-restart:
	docker-compose restart

# Single container Docker commands
docker-single-build:
	docker-compose -f docker-compose.single.yml build

docker-single-up:
	docker-compose -f docker-compose.single.yml up -d

docker-single-down:
	docker-compose -f docker-compose.single.yml down

docker-single-logs:
	docker-compose -f docker-compose.single.yml logs -f

docker-single-restart:
	docker-compose -f docker-compose.single.yml restart

docker-single-status:
	docker exec -it finantrade_single supervisorctl status

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	cd frontend/web && rm -rf .next node_modules/.cache 2>/dev/null || true
	@echo "Cleaned build artifacts and caches"

# Quick dev setup
install-backend:
	pip install -e .

install-frontend:
	cd frontend/web && npm install

install: install-backend install-frontend
	@echo "All dependencies installed"
