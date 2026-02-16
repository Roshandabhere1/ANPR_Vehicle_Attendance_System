import hmac
import os
import secrets
from pathlib import Path
from typing import Generator

from fastapi import Cookie, Depends, Header, HTTPException, Response, status
from sqlalchemy.orm import Session

from adapters.detectors.yolo_tesseract_detector import YoloTesseractPlateDetector
from adapters.repositories.sqlalchemy_adapter import SQLAlchemyVehicleRepository
from core.ports.plate_detector import PlateDetectorPort
from database.database import SessionLocal


def _resolve_env_file_for_write() -> Path:
    candidates = [Path(".env"), Path(".env.local"), Path(".env.app")]
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    # If .env is taken by virtualenv directory, use .env.local.
    return Path(".env.local")


def _persist_env_value(key: str, value: str) -> None:
    env_path = _resolve_env_file_for_write()
    line = f"{key}={value}\n"
    if not env_path.exists():
        env_path.write_text(line, encoding="utf-8")
        return

    content = env_path.read_text(encoding="utf-8")
    if f"{key}=" in content:
        return
    with env_path.open("a", encoding="utf-8") as fh:
        fh.write("\n" + line)


def _required_api_key() -> str:
    key = os.getenv("ANPR_API_KEY", "").strip()
    if key:
        return key

    generated_key = secrets.token_urlsafe(32)
    os.environ["ANPR_API_KEY"] = generated_key
    _persist_env_value("ANPR_API_KEY", generated_key)
    return generated_key


def require_api_key(
    x_api_key: str | None = Header(default=None),
    anpr_api_key: str | None = Cookie(default=None),
) -> None:
    expected = _required_api_key()
    provided = x_api_key or anpr_api_key or ""
    if not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


def issue_api_cookie(response: Response) -> None:
    key = _required_api_key()
    secure_cookie = os.getenv("COOKIE_SECURE", "false").lower() in {"1", "true", "yes"}
    response.set_cookie(
        key="anpr_api_key",
        value=key,
        httponly=True,
        secure=secure_cookie,
        samesite="lax",
        max_age=60 * 60 * 24,
    )


def get_db_session() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_vehicle_repo(db: Session = Depends(get_db_session)) -> SQLAlchemyVehicleRepository:
    return SQLAlchemyVehicleRepository(db)


def get_plate_detector() -> PlateDetectorPort:
    return YoloTesseractPlateDetector()
