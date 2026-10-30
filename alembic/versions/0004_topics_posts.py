"""topics, posts, ingestion_runs: continuous topic monitoring

Revision ID: 0004
Revises: 0003
"""
from alembic import op
import sqlalchemy as sa

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    existing = set(sa.inspect(op.get_bind()).get_table_names())  # create_all() may have made them already
    if "topics" not in existing:
        op.create_table(
            "topics",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("name", sa.String, nullable=False, unique=True),
            sa.Column("query", sa.String, nullable=False),
            sa.Column("sources", sa.JSON, nullable=False),
            sa.Column("active", sa.Boolean, nullable=False, server_default=sa.true()),
            sa.Column("created_at", sa.DateTime, nullable=True),
        )
        op.create_index("ix_topics_id", "topics", ["id"])
    if "posts" not in existing:
        op.create_table(
            "posts",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("topic_id", sa.Integer, sa.ForeignKey("topics.id"), nullable=False),
            sa.Column("source", sa.String, nullable=False),
            sa.Column("external_id", sa.String, nullable=False),
            sa.Column("content_hash", sa.String, nullable=False),
            sa.Column("url", sa.String, nullable=True),
            sa.Column("text", sa.String, nullable=False),
            sa.Column("published_at", sa.DateTime, nullable=True),
            sa.Column("fetched_at", sa.DateTime, nullable=True),
            sa.Column("prediction", sa.String, nullable=False),
            sa.Column("confidence", sa.Float, nullable=True),
            sa.Column("decision_source", sa.String, nullable=True),
            sa.UniqueConstraint("topic_id", "external_id"),
            sa.UniqueConstraint("topic_id", "content_hash"),
        )
        op.create_index("ix_posts_id", "posts", ["id"])
        op.create_index("ix_posts_topic_id", "posts", ["topic_id"])
        op.create_index("ix_posts_fetched_at", "posts", ["fetched_at"])
    if "ingestion_runs" not in existing:
        op.create_table(
            "ingestion_runs",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("topic_id", sa.Integer, sa.ForeignKey("topics.id"), nullable=False),
            sa.Column("source", sa.String, nullable=False),
            sa.Column("started_at", sa.DateTime, nullable=False),
            sa.Column("finished_at", sa.DateTime, nullable=True),
            sa.Column("fetched", sa.Integer, nullable=False, server_default="0"),
            sa.Column("new", sa.Integer, nullable=False, server_default="0"),
            sa.Column("error", sa.String, nullable=True),
        )
        op.create_index("ix_ingestion_runs_id", "ingestion_runs", ["id"])
        op.create_index("ix_ingestion_runs_topic_id", "ingestion_runs", ["topic_id"])


def downgrade() -> None:
    existing = set(sa.inspect(op.get_bind()).get_table_names())
    for table in ("ingestion_runs", "posts", "topics"):
        if table in existing:
            op.drop_table(table)
