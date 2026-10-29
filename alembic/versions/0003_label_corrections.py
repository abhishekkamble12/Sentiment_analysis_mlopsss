"""label_corrections: user reports of wrong predictions (annotation candidates, never training labels)

Revision ID: 0003
Revises: 0002
"""
from alembic import op
import sqlalchemy as sa

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if "label_corrections" in set(sa.inspect(op.get_bind()).get_table_names()):
        return  # created by Base.metadata.create_all() on a fresh DB
    op.create_table(
        "label_corrections",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("text", sa.String, nullable=False),
        sa.Column("predicted", sa.String, nullable=False),
        sa.Column("corrected", sa.String, nullable=False),
        sa.Column("decision_source", sa.String, nullable=True),
        sa.Column("reviewed", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime),
    )
    op.create_index("ix_label_corrections_id", "label_corrections", ["id"])


def downgrade() -> None:
    if "label_corrections" in set(sa.inspect(op.get_bind()).get_table_names()):
        op.drop_table("label_corrections")
