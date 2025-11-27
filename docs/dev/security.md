# Security Guidelines

## API Key Management
- Store sensitive credentials in environment variables, never directly in `config/system.yml`.
- Use placeholders like `${BINANCE_API_KEY}` and `${BINANCE_SECRET_KEY}` in the YAML; the loader resolves them via `resolve_env_placeholders`.
- Keep a `.env` file (excluded from git) or export keys in your shell profile, e.g. `export BINANCE_API_KEY=...`.

## Testnet vs Mainnet
- Default configuration targets Binance Futures testnet with `dry_run=true`.
- Validate the entire pipeline (training, backtest, live) on testnet and small size before switching to mainnet endpoints.
- Monitor kill-switch and exchange risk limits when moving to production.

## Dry-Run & Risk Limits
- `exchange.dry_run=true` prevents accidental live orders; disable it only after manual review.
- Configure `exchange.max_leverage`, `max_position_notional`, and kill-switch thresholds to cap exposure.

## Infrastructure Hygiene
- If the exchange account is IP-locked or requires VPN, ensure the live machine matches the allowlist.
- Use supervisors (systemd, Supervisor, Kubernetes) alongside the heartbeat/watchdog scripts to restart/alert when the live loop stalls.
