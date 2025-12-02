import os

import pytest

TEST_DSN_ENV = "FTA_TEST_DB_DSN"

pytestmark = pytest.mark.integration

# Skip early if no Postgres client module is available
postgres_module = pytest.importorskip(
    "finantradealgo.storage.postgres_client",
    reason="Postgres client module not present; skipping DB integration scaffold.",
)

PostgresClient = getattr(postgres_module, "PostgresClient", None)
DBConnectionConfig = getattr(postgres_module, "DBConnectionConfig", None)

db_dsn = os.getenv(TEST_DSN_ENV)
if not db_dsn:
    pytest.skip(f"Skipping DB integration tests, {TEST_DSN_ENV} is not set")


def test_postgres_basic_roundtrip():
    if PostgresClient is None or DBConnectionConfig is None:
        pytest.xfail("Postgres client/config classes not implemented yet")

    conn_config = DBConnectionConfig(dsn=db_dsn)
    client = PostgresClient(conn_config=conn_config)

    try:
        client.connect()
    except NotImplementedError:
        pytest.xfail("connect() not implemented yet")
    except Exception as exc:  # pragma: no cover - defensive guard
        pytest.fail(f"Unexpected DB connection error: {exc}")
