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
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
