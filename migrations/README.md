# Database Migrations

This folder contains Alembic migrations for the FinanTradeAlgo PostgreSQL schema.

## Structure

- `env.py` - Alembic environment file, configures DB connection and target metadata.
- `versions/` - Individual migration scripts (upgrade/downgrade).

## Running migrations

Typical commands (once Alembic is installed and configured):

```bash
alembic upgrade head      # Apply all migrations
alembic downgrade -1      # Roll back last migration
```

Configuration (DSN, script location, etc.) should be aligned with:

- `finantradealgo/storage/postgres_client.py` (DBConnectionConfig.dsn)
- Your deployment environment variables.
