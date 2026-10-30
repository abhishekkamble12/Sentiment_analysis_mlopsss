"""agent_runs (traces) and annotation_queue (posts worth a human label)

Revision ID: 0007
Revises: 0006
"""
from alembic import op
import sqlalchemy as sa

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    existing = set(sa.inspect(op.get_bind()).get_table_names())  # create_all() may have made them already
    if "agent_runs" not in existing:
        op.create_table(
            "agent_runs",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("kind", sa.String, nullable=False),
            sa.Column("question", sa.String, nullable=False),
            sa.Column("topic_id", sa.Integer, nullable=True),
            sa.Column("status", sa.String, nullable=False),
            sa.Column("steps", sa.JSON, nullable=False),
            sa.Column("answer", sa.JSON, nullable=True),
            sa.Column("error", sa.String, nullable=True),
            sa.Column("tokens_in", sa.Integer, nullable=False, server_default="0"),
            sa.Column("tokens_out", sa.Integer, nullable=False, server_default="0"),
            sa.Column("latency_ms", sa.Float, nullable=True),
            sa.Column("provider", sa.String, nullable=True),
            sa.Column("model", sa.String, nullable=True),
            sa.Column("prompt_version", sa.String, nullable=True),
            sa.Column("created_at", sa.DateTime, nullable=True),
            sa.Column("finished_at", sa.DateTime, nullable=True),
        )
        op.create_index("ix_agent_runs_id", "agent_runs", ["id"])
        op.create_index("ix_agent_runs_topic_id", "agent_runs", ["topic_id"])
        op.create_index("ix_agent_runs_created_at", "agent_runs", ["created_at"])
    if "annotation_queue" not in existing:
        op.create_table(
            "annotation_queue",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("post_id", sa.Integer, sa.ForeignKey("posts.id"), nullable=False, unique=True),
            sa.Column("topic_id", sa.Integer, nullable=False),
            sa.Column("reason", sa.String, nullable=False),
            sa.Column("priority", sa.Float, nullable=False),
            sa.Column("status", sa.String, nullable=False, server_default="pending"),
            sa.Column("created_at", sa.DateTime, nullable=True),
        )
        op.create_index("ix_annotation_queue_id", "annotation_queue", ["id"])
        op.create_index("ix_annotation_queue_topic_id", "annotation_queue", ["topic_id"])


def downgrade() -> None:
    existing = set(sa.inspect(op.get_bind()).get_table_names())
    for table in ("annotation_queue", "agent_runs"):
        if table in existing:
            op.drop_table(table)
