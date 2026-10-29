"""baseline schema; prediction_logs.confidence becomes nullable

Databases created earlier by Base.metadata.create_all() have confidence NOT NULL, which rejects
LLM decisions (they report no confidence). This revision creates missing tables and, for an
existing prediction_logs table, relaxes that constraint. It is safe to run on either kind of DB.

Revision ID: 0001
Revises:
"""
from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    existing = set(sa.inspect(op.get_bind()).get_table_names())

    if "prediction_logs" not in existing:
        op.create_table(
            "prediction_logs",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("latency_ms", sa.Float, nullable=False),
            sa.Column("confidence", sa.Float, nullable=True),
            sa.Column("model_used", sa.String, nullable=False),
            sa.Column("prediction", sa.String, nullable=False),
            sa.Column("text_length", sa.Integer, nullable=False),
            sa.Column("timestamp", sa.DateTime),
        )
        op.create_index("ix_prediction_logs_id", "prediction_logs", ["id"])
    else:
        with op.batch_alter_table("prediction_logs") as batch:
            batch.alter_column("confidence", existing_type=sa.Float(), nullable=True)

    if "reports" not in existing:
        op.create_table(
            "reports",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("query", sa.String, nullable=False),
            sa.Column("top_keywords", sa.JSON, nullable=True),
            sa.Column("sentiment_distribution", sa.JSON, nullable=True),
            sa.Column("fallback_used", sa.Boolean),
            sa.Column("timestamp", sa.DateTime),
        )
        op.create_index("ix_reports_id", "reports", ["id"])

    if "monitoring_metrics" not in existing:
        op.create_table(
            "monitoring_metrics",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("avg_latency", sa.Float, nullable=False),
            sa.Column("avg_confidence", sa.Float, nullable=False),
            sa.Column("fallback_rate", sa.Float, nullable=False),
            sa.Column("prediction_distribution", sa.JSON, nullable=False),
            sa.Column("timestamp", sa.DateTime),
        )
        op.create_index("ix_monitoring_metrics_id", "monitoring_metrics", ["id"])


def downgrade() -> None:
    # Restoring NOT NULL would fail on rows with NULL confidence, and dropping tables would destroy
    # logged data, so this baseline is intentionally one-way.
    raise NotImplementedError("0001 is a baseline and cannot be downgraded")
