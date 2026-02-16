"""Simple DB health-check script.

Prints JSON-like status: connection, tables, counts.

Run with project venv:
  .venv/bin/python3 scripts/db_health.py
"""
import json
from sqlalchemy import create_engine, inspect, text
from config import get_database_url


def main():
    url = get_database_url()
    engine = create_engine(url)
    out = {"database_url": url, "connected": False, "tables": [], "counts": {}}
    try:
        with engine.connect() as conn:
            out["connected"] = True
            try:
                conn.execute(text("SELECT 1"))
            except Exception:
                out["connected"] = False
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            out["tables"] = tables
            for t in tables:
                try:
                    r = conn.execute(text(f"SELECT COUNT(*) as c FROM {t}"))
                    out["counts"][t] = int(r.scalar() or 0)
                except Exception:
                    out["counts"][t] = None
    except Exception as e:
        out["error"] = str(e)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
