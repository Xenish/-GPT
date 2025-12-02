from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Initial schema for core tables:

    - trades
    - strategy_metadata
    - configs

    Timescale-specific hypertable setup for ohlcv/metrics can be added
    in a follow-up migration.
    """

    op.create_table(
        "trades",
        sa.Column("trade_id", sa.String(length=64), primary_key=True),
        sa.Column("account_id", sa.String(length=64), nullable=False),
        sa.Column("strategy_id", sa.String(length=128), nullable=True),
        sa.Column("symbol", sa.String(length=64), nullable=False),
        sa.Column("side", sa.String(length=8), nullable=False),
        sa.Column("qty", sa.Float, nullable=False),
        sa.Column("price", sa.Float, nullable=False),
        sa.Column("pnl", sa.Float, nullable=True),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata", sa.JSON, nullable=True),
    )

    op.create_table(
        "strategy_metadata",
        sa.Column("strategy_id", sa.String(length=128), primary_key=True),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("params", sa.JSON, nullable=True),
    )

    op.create_table(
        "configs",
        sa.Column("key", sa.String(length=128), nullable=False),
        sa.Column("profile", sa.String(length=64), nullable=True),
        sa.Column("payload", sa.JSON, nullable=False),
        sa.PrimaryKeyConstraint("key", "profile", name="pk_configs"),
    )

    # Optional: tables for metrics / ohlcv as plain Postgres tables;
    # Timescale hypertable conversion can be done later.
    op.create_table(
        "metrics",
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("value", sa.Float, nullable=False),
        sa.Column("tags", sa.JSON, nullable=True),
    )
    op.create_index("ix_metrics_ts_name", "metrics", ["ts", "name"])

    op.create_table(
        "ohlcv",
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.String(length=64), nullable=False),
        sa.Column("timeframe", sa.String(length=16), nullable=False),
        sa.Column("open", sa.Float, nullable=False),
        sa.Column("high", sa.Float, nullable=False),
        sa.Column("low", sa.Float, nullable=False),
        sa.Column("close", sa.Float, nullable=False),
        sa.Column("volume", sa.Float, nullable=True),
    )
    op.create_index("ix_ohlcv_symbol_timeframe_ts", "ohlcv", ["symbol", "timeframe", "ts"])


def downgrade() -> None:
    op.drop_index("ix_ohlcv_symbol_timeframe_ts", table_name="ohlcv")
    op.drop_table("ohlcv")

    op.drop_index("ix_metrics_ts_name", table_name="metrics")
    op.drop_table("metrics")

    op.drop_table("configs")
    op.drop_table("strategy_metadata")
    op.drop_table("trades")
