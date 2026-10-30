"""api_keys: hashed keys for the write, ingestion and agent endpoints

Revision ID: 0008
Revises: 0007
"""
from alembic import op
import sqlalchemy as sa

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if "api_keys" in set(sa.inspect(op.get_bind()).get_table_names()):
        return  # created by Base.metadata.create_all() on a fresh DB
    op.create_table(
        "api_keys",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String, nullable=False),
        sa.Column("key_hash", sa.String, nullable=False, unique=True),
        sa.Column("revoked", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime, nullable=True),
    )
    op.create_index("ix_api_keys_id", "api_keys", ["id"])


def downgrade() -> None:
    if "api_keys" in set(sa.inspect(op.get_bind()).get_table_names()):
        op.drop_table("api_keys")
