"""The baseline Alembic revision must work on a fresh DB and on one created by the old create_all()."""
import sqlite3
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

from tweet_sentiment_classifier.config.env_config import get_settings

REPO = Path(__file__).resolve().parents[1]

OLD_SCHEMA = """
CREATE TABLE prediction_logs (
    id INTEGER NOT NULL PRIMARY KEY, latency_ms FLOAT NOT NULL, confidence FLOAT NOT NULL,
    model_used VARCHAR NOT NULL, prediction VARCHAR NOT NULL, text_length INTEGER NOT NULL, timestamp DATETIME);
CREATE TABLE reports (id INTEGER NOT NULL PRIMARY KEY, query VARCHAR NOT NULL, top_keywords JSON,
    sentiment_distribution JSON, fallback_used BOOLEAN, timestamp DATETIME);
"""

INSERT_NULL = ("INSERT INTO prediction_logs (latency_ms, confidence, model_used, prediction, text_length) "
               "VALUES (900, NULL, 'llm:x', 'negative', 12)")


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    path = tmp_path / "mig.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{path.as_posix()}")
    get_settings.cache_clear()  # alembic/env.py reads the URL from settings
    yield path
    get_settings.cache_clear()


def upgrade():
    cfg = Config(str(REPO / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO / "alembic"))
    command.upgrade(cfg, "head")


def test_fresh_database_gets_all_tables_with_nullable_confidence(db_path):
    upgrade()
    con = sqlite3.connect(db_path)
    tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    assert {"prediction_logs", "reports", "monitoring_metrics", "label_corrections", "alembic_version"} <= tables
    con.execute(INSERT_NULL)  # would raise IntegrityError if confidence were still NOT NULL


def test_existing_database_keeps_its_rows_and_accepts_null_confidence(db_path):
    con = sqlite3.connect(db_path)
    con.executescript(OLD_SCHEMA)
    con.execute("INSERT INTO prediction_logs (latency_ms, confidence, model_used, prediction, text_length) "
                "VALUES (5, 0.8, 'LogisticRegression', '1', 10)")
    con.commit()
    with pytest.raises(sqlite3.IntegrityError):
        con.execute(INSERT_NULL)  # precondition: the old schema really rejects it
    con.close()

    upgrade()

    con = sqlite3.connect(db_path)
    assert con.execute("select count(*), min(confidence) from prediction_logs").fetchone() == (1, 0.8)
    con.execute(INSERT_NULL)
    con.commit()
    assert con.execute("select count(*) from prediction_logs").fetchone()[0] == 2
    assert {"monitoring_metrics", "label_corrections"} <= {r[0] for r in con.execute("select name from sqlite_master")}


def test_topic_tables_are_created_on_fresh_and_existing_databases(db_path):
    con = sqlite3.connect(db_path)
    con.executescript(OLD_SCHEMA)
    con.commit()
    con.close()

    upgrade()

    con = sqlite3.connect(db_path)
    tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    assert {"topics", "posts", "ingestion_runs"} <= tables
    con.execute("INSERT INTO topics (name, query, sources) VALUES ('t', 'q', '[]')")
    insert = ("INSERT INTO posts (topic_id, source, external_id, content_hash, text, prediction) "
              "VALUES (1, 's', ?, ?, 'x', 'neutral')")
    con.execute(insert, ("e1", "h1"))
    with pytest.raises(sqlite3.IntegrityError):  # same source id twice for one topic
        con.execute(insert, ("e1", "h2"))
    with pytest.raises(sqlite3.IntegrityError):  # same content twice for one topic
        con.execute(insert, ("e2", "h1"))
