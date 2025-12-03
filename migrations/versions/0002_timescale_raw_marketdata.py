from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0002_timescale_raw_marketdata"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Add raw market data tables and convert them to Timescale hypertables.

    Tables:
    - raw_ohlcv: core candles partitioned by (symbol, timeframe, ts)
    - raw_funding_rates: per-symbol funding rates
    - raw_open_interest: open interest snapshots
    - raw_flow: perp premium/basis/liquidations flows
    - raw_sentiment: sentiment scores/time-series
    """

    # Enable TimescaleDB extension if available (no-op if already installed)
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'
            ) THEN
                CREATE EXTENSION IF NOT EXISTS timescaledb;
            END IF;
        END$$;
        """
    )

    op.create_table(
        "raw_ohlcv",
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.String(length=64), nullable=False),
        sa.Column("timeframe", sa.String(length=16), nullable=False),
        sa.Column("open", sa.Float, nullable=False),
        sa.Column("high", sa.Float, nullable=False),
        sa.Column("low", sa.Float, nullable=False),
        sa.Column("close", sa.Float, nullable=False),
        sa.Column("volume", sa.Float, nullable=True),
        sa.Column("vwap", sa.Float, nullable=True),
        sa.PrimaryKeyConstraint("symbol", "timeframe", "ts", name="pk_raw_ohlcv"),
    )
    op.execute(
        """
        SELECT create_hypertable('raw_ohlcv', 'ts', chunk_time_interval => interval '7 days', if_not_exists => TRUE);
        SELECT add_dimension('raw_ohlcv', 'symbol', if_not_exists => TRUE);
        """
    )
    op.create_index(
        "ix_raw_ohlcv_symbol_tf_ts",
        "raw_ohlcv",
        ["symbol", "timeframe", "ts"],
    )
    op.create_index(
        "ix_raw_ohlcv_ts_desc",
        "raw_ohlcv",
        [sa.text("ts DESC")],
    )

    op.create_table(
        "raw_funding_rates",
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.String(length=64), nullable=False),
        sa.Column("timeframe", sa.String(length=16), nullable=False, server_default="8h"),
        sa.Column("funding_rate", sa.Float, nullable=False),
        sa.Column("mark_price", sa.Float, nullable=True),
        sa.Column("index_price", sa.Float, nullable=True),
        sa.Column("open_interest", sa.Float, nullable=True),
        sa.PrimaryKeyConstraint("symbol", "ts", name="pk_raw_funding_rates"),
    )
    op.execute(
        """
        SELECT create_hypertable('raw_funding_rates', 'ts', chunk_time_interval => interval '30 days', if_not_exists => TRUE);
        SELECT add_dimension('raw_funding_rates', 'symbol', if_not_exists => TRUE);
        """
    )
    op.create_index(
        "ix_raw_funding_symbol_ts",
        "raw_funding_rates",
        ["symbol", "ts"],
    )

    op.create_table(
        "raw_open_interest",
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.String(length=64), nullable=False),
        sa.Column("timeframe", sa.String(length=16), nullable=False, server_default="1h"),
        sa.Column("open_interest", sa.Float, nullable=False),
        sa.Column("volume", sa.Float, nullable=True),
        sa.Column("turnover", sa.Float, nullable=True),
        sa.PrimaryKeyConstraint("symbol", "ts", name="pk_raw_open_interest"),
    )
    op.execute(
        """
        SELECT create_hypertable('raw_open_interest', 'ts', chunk_time_interval => interval '30 days', if_not_exists => TRUE);
        SELECT add_dimension('raw_open_interest', 'symbol', if_not_exists => TRUE);
        """
    )
    op.create_index(
        "ix_raw_oi_symbol_ts",
        "raw_open_interest",
        ["symbol", "ts"],
    )

    op.create_table(
        "raw_flow",
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.String(length=64), nullable=False),
        sa.Column("timeframe", sa.String(length=16), nullable=False),
        sa.Column("perp_premium", sa.Float, nullable=True),
        sa.Column("basis", sa.Float, nullable=True),
        sa.Column("oi", sa.Float, nullable=True),
        sa.Column("oi_change", sa.Float, nullable=True),
        sa.Column("liq_up", sa.Float, nullable=True),
        sa.Column("liq_down", sa.Float, nullable=True),
        sa.PrimaryKeyConstraint("symbol", "timeframe", "ts", name="pk_raw_flow"),
    )
    op.execute(
        """
        SELECT create_hypertable('raw_flow', 'ts', chunk_time_interval => interval '30 days', if_not_exists => TRUE);
        SELECT add_dimension('raw_flow', 'symbol', if_not_exists => TRUE);
        """
    )
    op.create_index(
        "ix_raw_flow_symbol_tf_ts",
        "raw_flow",
        ["symbol", "timeframe", "ts"],
    )

    op.create_table(
        "raw_sentiment",
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.String(length=64), nullable=False),
        sa.Column("timeframe", sa.String(length=16), nullable=False),
        sa.Column("sentiment_score", sa.Float, nullable=False),
        sa.Column("volume", sa.Float, nullable=True),
        sa.Column("source", sa.String(length=64), nullable=True),
        sa.PrimaryKeyConstraint("symbol", "timeframe", "ts", name="pk_raw_sentiment"),
    )
    op.execute(
        """
        SELECT create_hypertable('raw_sentiment', 'ts', chunk_time_interval => interval '30 days', if_not_exists => TRUE);
        SELECT add_dimension('raw_sentiment', 'symbol', if_not_exists => TRUE);
        """
    )
    op.create_index(
        "ix_raw_sentiment_symbol_tf_ts",
        "raw_sentiment",
        ["symbol", "timeframe", "ts"],
    )


def downgrade() -> None:
    op.drop_index("ix_raw_sentiment_symbol_tf_ts", table_name="raw_sentiment")
    op.drop_table("raw_sentiment")

    op.drop_index("ix_raw_flow_symbol_tf_ts", table_name="raw_flow")
    op.drop_table("raw_flow")

    op.drop_index("ix_raw_oi_symbol_ts", table_name="raw_open_interest")
    op.drop_table("raw_open_interest")

    op.drop_index("ix_raw_funding_symbol_ts", table_name="raw_funding_rates")
    op.drop_table("raw_funding_rates")

    op.drop_index("ix_raw_ohlcv_ts_desc", table_name="raw_ohlcv")
    op.drop_index("ix_raw_ohlcv_symbol_tf_ts", table_name="raw_ohlcv")
    op.drop_table("raw_ohlcv")
