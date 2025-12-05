# Database Migrations

This folder contains Alembic migrations for the FinanTradeAlgo PostgreSQL schema.

## Structure

- `env.py` - Alembic environment file, configures DB connection and target metadata.
- `versions/` - Individual migration scripts (upgrade/downgrade).

## Running migrations

Typical commands (once Alembic is installed and configured):

```bash
export FT_TIMESCALE_DSN="postgresql://user:pass@host:5432/dbname"
alembic upgrade head      # Apply all migrations
alembic downgrade -1      # Roll back last migration
```

`env.py` reads `sqlalchemy.url` from alembic.ini or `FT_TIMESCALE_DSN` if not set.
Ensure the DSN points to a Timescale/Postgres instance; hypertable creation is guarded with `if_not_exists`.
