from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0003_ingestion_state"
down_revision = "0002_timescale_raw_marketdata"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Scheduler/ingestion state tables to support idempotent jobs and watermarks.
    """
    op.create_table(
        "ingestion_runs",
        sa.Column("run_id", sa.String(length=64), primary_key=True),
        sa.Column("job_name", sa.String(length=128), nullable=False, index=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("watermark", sa.JSON, nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.Text, nullable=True),
    )
    op.create_index("ix_ingestion_runs_job_name", "ingestion_runs", ["job_name"])

    op.create_table(
        "ingestion_watermarks",
        sa.Column("job_name", sa.String(length=128), nullable=False),
        sa.Column("scope", sa.String(length=256), nullable=False),
        sa.Column("watermark_ts", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("job_name", "scope", name="pk_ingestion_watermarks"),
    )
    op.create_index("ix_ingestion_watermarks_job_scope", "ingestion_watermarks", ["job_name", "scope"])


def downgrade() -> None:
    op.drop_index("ix_ingestion_watermarks_job_scope", table_name="ingestion_watermarks")
    op.drop_table("ingestion_watermarks")

    op.drop_index("ix_ingestion_runs_job_name", table_name="ingestion_runs")
    op.drop_table("ingestion_runs")
