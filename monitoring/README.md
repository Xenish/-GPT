# Monitoring Stack

This folder contains Docker Compose and config to run Prometheus + Grafana for FinanTrade observability.

## Run
```bash
docker-compose -f docker-compose.monitoring.yml up -d
```

## Services
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (default admin/admin)

## Dashboards
- Provisioned from `finantradealgo/monitoring/grafana_dashboards/`

## Prometheus scrape target
- Update `monitoring/prometheus.yml` to point to your FastAPI service host/port (defaults to `backend:8000/metrics`).
