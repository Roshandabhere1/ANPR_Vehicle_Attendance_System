"""Database connection manager.

This module prefers a `DATABASE_URL` environment variable. If not set, it will
try to build a MySQL URL from `MYSQL_*` variables using `config.get_database_url()`.
MySQL is required; SQLite fallback is intentionally disabled.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

try:
    from config import get_database_url
except Exception:
    get_database_url = None

# Determine the DATABASE_URL (priority: env DATABASE_URL -> config builder)
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL and get_database_url is not None:
    DATABASE_URL = get_database_url()
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not configured. Set DATABASE_URL or MYSQL_* environment variables."
    )

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Dependency to get DB session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
