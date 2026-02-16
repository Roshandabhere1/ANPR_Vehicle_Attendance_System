"""Helper to create database tables from SQLAlchemy models.

Run: `python3 -m database.init_db` to create tables using current DATABASE_URL.
"""
from .database import engine, Base
from . import models  # noqa: F401 ensure models are imported so metadata is populated


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init_db()
    print("Database tables created (if not present).")
