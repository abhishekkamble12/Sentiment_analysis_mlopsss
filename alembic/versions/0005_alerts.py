"""alerts: statistically unusual changes on a monitored topic

Revision ID: 0005
Revises: 0004
"""
from alembic import op
import sqlalchemy as sa

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if "alerts" in set(sa.inspect(op.get_bind()).get_table_names()):
        return  # created by Base.metadata.create_all() on a fresh DB
    op.create_table(
        "alerts",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("topic_id", sa.Integer, sa.ForeignKey("topics.id"), nullable=False),
        sa.Column("kind", sa.String, nullable=False),
        sa.Column("window_hours", sa.Integer, nullable=False),
        sa.Column("baseline", sa.Float, nullable=False),
        sa.Column("observed", sa.Float, nullable=False),
        sa.Column("p_value", sa.Float, nullable=True),
        sa.Column("message", sa.String, nullable=False),
        sa.Column("acknowledged", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime, nullable=True),
    )
    op.create_index("ix_alerts_id", "alerts", ["id"])
    op.create_index("ix_alerts_topic_id", "alerts", ["topic_id"])
    op.create_index("ix_alerts_created_at", "alerts", ["created_at"])


def downgrade() -> None:
    if "alerts" in set(sa.inspect(op.get_bind()).get_table_names()):
        op.drop_table("alerts")
