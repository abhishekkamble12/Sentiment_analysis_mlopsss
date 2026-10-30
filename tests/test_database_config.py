"""A DATABASE_URL whose driver is missing must fail with a message that says what to do."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
HAS_PG_DRIVER = bool(importlib.util.find_spec("psycopg") or importlib.util.find_spec("psycopg2"))


def import_database(url):
    env = {**os.environ, "DATABASE_URL": url, "PYTHONPATH": str(SRC)}
    return subprocess.run([sys.executable, "-c", "import tweet_sentiment_classifier.config.database"],
                          env=env, capture_output=True, text=True, timeout=60)


@pytest.mark.skipif(HAS_PG_DRIVER, reason="a Postgres driver is installed here, so the import would succeed")
def test_missing_postgres_driver_gives_an_actionable_message_without_leaking_the_password():
    result = import_database("postgresql://user:SECRET-PW@db:5432/x")
    assert result.returncode != 0
    assert "driver 'psycopg' is not installed" in result.stderr
    assert "sqlite:///./monitoring.db" in result.stderr and "Remove-Item Env:DATABASE_URL" in result.stderr
    assert "SECRET-PW" not in result.stderr and "Traceback" in result.stderr  # the cause is shown, the secret is not


def test_sqlite_url_imports_cleanly(tmp_path):
    assert import_database(f"sqlite:///{(tmp_path / 'ok.db').as_posix()}").returncode == 0
