"""llm_calls (cache and token ledger) and insights (themes, daily digest)

Revision ID: 0006
Revises: 0005
"""
from alembic import op
import sqlalchemy as sa

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    existing = set(sa.inspect(op.get_bind()).get_table_names())  # create_all() may have made them already
    if "llm_calls" not in existing:
        op.create_table(
            "llm_calls",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("key", sa.String, nullable=False),
            sa.Column("provider", sa.String, nullable=False),
            sa.Column("model", sa.String, nullable=False),
            sa.Column("prompt_version", sa.String, nullable=False),
            sa.Column("tokens_in", sa.Integer, nullable=False, server_default="0"),
            sa.Column("tokens_out", sa.Integer, nullable=False, server_default="0"),
            sa.Column("ok", sa.Boolean, nullable=False, server_default=sa.false()),
            sa.Column("response", sa.JSON, nullable=True),
            sa.Column("created_at", sa.DateTime, nullable=True),
        )
        op.create_index("ix_llm_calls_id", "llm_calls", ["id"])
        op.create_index("ix_llm_calls_key", "llm_calls", ["key"])
        op.create_index("ix_llm_calls_created_at", "llm_calls", ["created_at"])
    if "insights" not in existing:
        op.create_table(
            "insights",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("topic_id", sa.Integer, sa.ForeignKey("topics.id"), nullable=False),
            sa.Column("kind", sa.String, nullable=False),
            sa.Column("day", sa.String, nullable=False),
            sa.Column("payload", sa.JSON, nullable=False),
            sa.Column("created_at", sa.DateTime, nullable=True),
        )
        op.create_index("ix_insights_id", "insights", ["id"])
        op.create_index("ix_insights_topic_id", "insights", ["topic_id"])
        op.create_index("ix_insights_day", "insights", ["day"])


def downgrade() -> None:
    existing = set(sa.inspect(op.get_bind()).get_table_names())
    for table in ("insights", "llm_calls"):
        if table in existing:
            op.drop_table(table)
