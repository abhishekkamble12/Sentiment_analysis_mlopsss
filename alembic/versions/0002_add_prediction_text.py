"""prediction_logs.text: store the analysed text so the Prediction History page can show it

The column is nullable so it is safe for databases created before this revision (their rows keep
their data and get NULL text) and for the anonymous monitoring inserts that omit it. On a fresh
database Base.metadata.create_all() already creates the column from the model; this migration only
adds it where it is missing.

Revision ID: 0002
Revises: 0001
"""
from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    if "prediction_logs" not in tables:
        return  # 0001 creates it (with the column when run on a fresh DB after this revision)

    columns = {c["name"] for c in inspector.get_columns("prediction_logs")}
    if "text" in columns:
        return  # already present (e.g. create_all() on a fresh DB built from the current model)

    with op.batch_alter_table("prediction_logs") as batch:
        batch.add_column(sa.Column("text", sa.String, nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "prediction_logs" not in set(inspector.get_table_names()):
        return
    columns = {c["name"] for c in inspector.get_columns("prediction_logs")}
    if "text" not in columns:
        return
    with op.batch_alter_table("prediction_logs") as batch:
        batch.drop_column("text")
