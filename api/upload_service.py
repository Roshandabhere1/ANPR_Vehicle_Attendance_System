import asyncio
import os
import re
import time
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile, status


UPLOAD_DIR = Path("data/uploads")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))
UPLOAD_TTL_SECONDS = int(os.getenv("UPLOAD_TTL_SECONDS", str(24 * 60 * 60)))
UPLOAD_CLEANUP_INTERVAL_SECONDS = int(os.getenv("UPLOAD_CLEANUP_INTERVAL_SECONDS", str(60 * 60)))
_cleanup_task: asyncio.Task | None = None


def _safe_vehicle_token(vehicle_number: str) -> str:
    cleaned = re.sub(r"[^A-Z0-9]", "", vehicle_number.upper())
    return cleaned[:20] if cleaned else "UNKNOWN"


def _safe_extension(filename: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .jpg, .jpeg and .png files are allowed.",
        )
    return suffix


async def save_upload_file(upload: UploadFile, prefix: str, vehicle_number: str = "") -> tuple[str, Path]:
    if not upload.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file selected.")

    ext = _safe_extension(upload.filename)
    content = await upload.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max allowed size is {MAX_UPLOAD_BYTES} bytes.",
        )

    vehicle_token = _safe_vehicle_token(vehicle_number) if vehicle_number else "NA"
    filename = f"{prefix}_{vehicle_token}_{uuid4().hex[:12]}{ext}"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    filepath = UPLOAD_DIR / filename
    filepath.write_bytes(content)
    await upload.close()
    return filename, filepath


def cleanup_expired_uploads(now_ts: float | None = None) -> int:
    """Delete uploaded files older than UPLOAD_TTL_SECONDS."""
    if not UPLOAD_DIR.exists():
        return 0

    cutoff = (now_ts or time.time()) - UPLOAD_TTL_SECONDS
    deleted = 0
    for path in UPLOAD_DIR.iterdir():
        if not path.is_file():
            continue
        try:
            if path.stat().st_mtime <= cutoff:
                path.unlink(missing_ok=True)
                deleted += 1
        except FileNotFoundError:
            continue
    return deleted


async def _cleanup_loop() -> None:
    while True:
        cleanup_expired_uploads()
        await asyncio.sleep(max(60, UPLOAD_CLEANUP_INTERVAL_SECONDS))


def start_upload_cleanup_scheduler() -> None:
    global _cleanup_task
    if _cleanup_task is None or _cleanup_task.done():
        _cleanup_task = asyncio.create_task(_cleanup_loop(), name="upload-cleanup-scheduler")


async def stop_upload_cleanup_scheduler() -> None:
    global _cleanup_task
    if _cleanup_task is None:
        return
    _cleanup_task.cancel()
    try:
        await _cleanup_task
    except asyncio.CancelledError:
        pass
    _cleanup_task = None
