from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool
import os

# this is the Alembic Config object, which provides access to the values
# within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None  # we use imperative migrations with raw SQL


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url") or os.getenv("FT_TIMESCALE_DSN")
    if not url:
        raise RuntimeError("Set sqlalchemy.url in alembic.ini or provide FT_TIMESCALE_DSN env.")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    dsn = config.get_main_option("sqlalchemy.url") or os.getenv("FT_TIMESCALE_DSN")
    if not dsn:
        raise RuntimeError("Set sqlalchemy.url in alembic.ini or provide FT_TIMESCALE_DSN env.")

    section = config.get_section(config.config_ini_section)
    section["sqlalchemy.url"] = dsn

    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
