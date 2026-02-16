"""Project configuration helpers (loads .env and builds MySQL DB URL).

Usage:
  - Set a full `DATABASE_URL` in the environment, or
  - Set `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_DB`.

This module exposes `get_database_url()` which returns a SQLAlchemy-compatible
MySQL URL. SQLite fallback is intentionally disabled.
"""
from __future__ import annotations
import os
from urllib.parse import quote_plus
from pathlib import Path


def _env_file_candidates() -> list[Path]:
    return [Path(".env"), Path(".env.local"), Path(".env.app")]


def _first_existing_env_file() -> Path | None:
    for path in _env_file_candidates():
        if path.exists() and path.is_file():
            return path
    return None

try:
    from dotenv import load_dotenv
    env_file = _first_existing_env_file()
    if env_file is not None:
        load_dotenv(dotenv_path=env_file)
    else:
        load_dotenv()
except Exception:
    # Fallback when python-dotenv is unavailable.
    env_file = _first_existing_env_file()
    if env_file is not None:
        with env_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def get_database_url() -> str:
    """Return DATABASE_URL from env or build a MySQL URL from MYSQL_* vars.

    Priority:
    1. `DATABASE_URL` env var (complete SQLAlchemy URL)
    2. Build from `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_DB`
    3. Raise error if not configured
    """
    url = os.getenv("DATABASE_URL")
    if url:
        return url

    user = os.getenv("MYSQL_USER")
    password = os.getenv("MYSQL_PASSWORD")
    host = os.getenv("MYSQL_HOST", "localhost")
    port = os.getenv("MYSQL_PORT", "3306")
    db = os.getenv("MYSQL_DB")

    if user and password and db:
        # Use PyMySQL dialect
        password_escaped = quote_plus(password)
        return f"mysql+pymysql://{user}:{password_escaped}@{host}:{port}/{db}"

    raise RuntimeError(
        "MySQL configuration missing. Set DATABASE_URL or MYSQL_USER, "
        "MYSQL_PASSWORD, MYSQL_HOST, MYSQL_PORT, MYSQL_DB in environment/.env.local."
    )


__all__ = ["get_database_url"]
