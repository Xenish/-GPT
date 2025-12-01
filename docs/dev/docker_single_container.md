# Single Container Deployment

## Overview

The single-container deployment packages both frontend and backend into one Docker container with Nginx as a reverse proxy. This is suitable for:

- Simple deployments (single server, no load balancing)
- Resource-constrained environments
- Easier orchestration with fewer moving parts
- Deployments where exposing multiple ports is problematic

**Default setup (2 containers) is recommended for:**
- Development (hot reload, independent restarts)
- Production with separate scaling of frontend/backend
- Microservices architecture

## Architecture

```
Port 8080 → Nginx → /         → Static Frontend (HTML/CSS/JS)
                 → /api/*     → FastAPI Backend (Python)
                 → /docs      → Swagger UI
                 → /health    → Health Check
```

**Process management**: Supervisor manages nginx + uvicorn in one container

**Frontend**: Static HTML export (no Node runtime needed)

**Backend**: FastAPI on localhost:8000 (not exposed externally)

## Files

- `Dockerfile.single` - Multi-stage build (frontend + backend)
- `docker-compose.single.yml` - Single-service compose file
- `docker/single/nginx.conf` - Nginx reverse proxy config
- `docker/single/supervisord.conf` - Process manager config
- `frontend/web/next.config.mjs` - Static export config

## Building

```bash
# Build image
docker build -f Dockerfile.single -t finantrade-single:latest .

# Or use docker-compose
docker-compose -f docker-compose.single.yml build
# Or use Makefile
make docker-single-build
```

**Build time**: ~5-10 minutes (includes frontend build + Python deps)

## Running

```bash
# Start container
docker-compose -f docker-compose.single.yml up -d
# Or
make docker-single-up

# View logs
docker-compose -f docker-compose.single.yml logs -f
# Or
make docker-single-logs

# Stop container
docker-compose -f docker-compose.single.yml down
# Or
make docker-single-down
```

**Access:**
- Frontend: http://localhost:8080
- API docs: http://localhost:8080/docs
- Health check: http://localhost:8080/health

## Volumes

Three directories are mounted:

```yaml
volumes:
  - ./data:/app/data        # Market data (OHLCV, features)
  - ./outputs:/app/outputs  # Backtest results, ML models, trades
  - ./config:/app/config    # Configuration files
```

**Important**: Create these directories before first run:
```bash
mkdir -p data outputs config
```

## Environment Variables

Set in `.env` file (not committed):

```bash
BINANCE_FUTURES_API_KEY=your_key
BINANCE_FUTURES_API_SECRET=your_secret
```

Docker Compose will inject these into the container.

**Frontend API URL**: Hardcoded to `/api` (relative path) during build.

## Health Check

Container includes health check:
```bash
curl http://localhost:8080/health
# Expected: {"status":"ok"}
```

Docker Compose will mark container unhealthy if health check fails 3 times.

## Process Management

Supervisor manages two processes:

1. **Nginx** (priority 10, starts first)
   - Listens on port 8080
   - Serves static files
   - Proxies API requests

2. **Uvicorn** (priority 20, starts after nginx)
   - FastAPI backend on localhost:8000
   - Not directly accessible from outside container

**View process status:**
```bash
docker exec -it finantrade_single supervisorctl status
# Or
make docker-single-status
```

**Restart a process:**
```bash
docker exec -it finantrade_single supervisorctl restart uvicorn
docker exec -it finantrade_single supervisorctl restart nginx
```

## Logs

**View all logs:**
```bash
docker-compose -f docker-compose.single.yml logs -f
# Or
make docker-single-logs
```

**View specific process logs inside container:**
```bash
# Nginx logs
docker exec -it finantrade_single tail -f /var/log/nginx/access.log
docker exec -it finantrade_single tail -f /var/log/nginx/error.log

# Uvicorn logs
docker exec -it finantrade_single tail -f /var/log/supervisor/uvicorn.log

# Supervisor logs
docker exec -it finantrade_single tail -f /var/log/supervisor/supervisord.log
```

## Troubleshooting

### Container won't start

1. **Check logs:**
   ```bash
   docker-compose -f docker-compose.single.yml logs
   ```

2. **Verify Dockerfile builds:**
   ```bash
   docker build -f Dockerfile.single -t test .
   ```

3. **Check supervisor status:**
   ```bash
   docker exec -it finantrade_single supervisorctl status
   ```

### Frontend shows 404

- Verify static files exist: `docker exec -it finantrade_single ls /app/frontend/out`
- Check nginx config: `docker exec -it finantrade_single nginx -t`
- View nginx logs: `docker exec -it finantrade_single tail /var/log/nginx/error.log`

### API requests fail (500/502)

- Check uvicorn is running: `docker exec -it finantrade_single supervisorctl status uvicorn`
- Test backend directly: `docker exec -it finantrade_single curl http://127.0.0.1:8000/health`
- View uvicorn logs: `docker exec -it finantrade_single tail /var/log/supervisor/uvicorn.log`

### Port 8080 already in use

Change port in `docker-compose.single.yml`:
```yaml
ports:
  - "8081:8080"  # Use 8081 instead
```

## Comparison: Single vs Two Containers

| Feature | Single Container | Two Containers |
|---------|------------------|----------------|
| **Ports** | 1 (8080) | 2 (3000, 8000) |
| **Processes** | Nginx + Uvicorn | Node + Uvicorn |
| **Frontend** | Static HTML | Node runtime |
| **Restart** | Both restart together | Independent restarts |
| **Scaling** | Scale together | Scale independently |
| **Dev mode** | Harder (rebuild for frontend changes) | Easier (hot reload) |
| **Production** | Simpler deployment | More flexible |
| **Resource usage** | Lower (no Node runtime) | Higher |

**Recommendation**: Use two-container setup (default `docker-compose.yml`) for development. Use single-container for simple production deployments.

## Updating Code

### Backend changes:
```bash
make docker-single-build
make docker-single-up
```

### Frontend changes:
```bash
# Rebuild required (static export)
make docker-single-build
make docker-single-up
```

**Note**: Frontend changes require full rebuild since it's static HTML.

## Production Considerations

1. **Environment variables**: Use Docker secrets or vault instead of .env file
2. **SSL/TLS**: Add nginx SSL configuration or use reverse proxy (Traefik, Caddy)
3. **Logging**: Ship logs to centralized system (ELK, Loki, CloudWatch)
4. **Monitoring**: Add Prometheus metrics exporter
5. **Health checks**: Configure orchestrator (Kubernetes, Docker Swarm) to use `/health`
6. **Resource limits**: Set memory/CPU limits in docker-compose.yml

## Migrating from Two Containers

```bash
# Stop two-container setup
docker-compose down
# Or
make docker-down

# Start single-container setup
docker-compose -f docker-compose.single.yml up -d
# Or
make docker-single-up

# Volumes (data, outputs) are preserved
```

No data migration needed - volumes are in the same location.

## References

- [Nginx reverse proxy guide](https://docs.nginx.com/nginx/admin-guide/web-server/reverse-proxy/)
- [Supervisor documentation](http://supervisord.org/configuration.html)
- [Next.js static export](https://nextjs.org/docs/pages/building-your-application/deploying/static-exports)
- [Docker multi-stage builds](https://docs.docker.com/build/building/multi-stage/)
