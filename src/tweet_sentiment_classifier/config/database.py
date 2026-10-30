import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from tweet_sentiment_classifier.config.env_config import get_settings

settings = get_settings()
DATABASE_URL = settings.DATABASE_URL

# SQLAlchemy 2.0 defaults postgresql:// to the 'psycopg' driver (psycopg 3).
# If psycopg is not installed but psycopg2 is available, transparently adapt to postgresql+psycopg2://
if DATABASE_URL.startswith("postgresql://"):
    try:
        import psycopg  # noqa: F401
    except ImportError:
        try:
            import psycopg2  # noqa: F401
            DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)
        except ImportError:
            pass

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
try:
    engine = create_engine(DATABASE_URL, connect_args=connect_args)
except ModuleNotFoundError as e:  # the driver for the configured database is not installed
    scheme = DATABASE_URL.split("://")[0]  # never print the rest: it holds the password
    raise RuntimeError(
        f"DATABASE_URL uses '{scheme}' but its Python driver '{e.name}' is not installed. For local use point it at "
        "SQLite: set DATABASE_URL=sqlite:///./monitoring.db (PowerShell: $env:DATABASE_URL = 'sqlite:///./monitoring.db', "
        "or Remove-Item Env:DATABASE_URL to fall back to the value in .env). For Postgres install requirements/serve.txt. "
        "A DATABASE_URL set in the terminal overrides the one in .env."
    ) from None
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
