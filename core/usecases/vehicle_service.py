"""Use-case layer for vehicle-related operations.

This module contains business logic that coordinates repositories (ports).
"""
from pathlib import Path
from difflib import SequenceMatcher

try:
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz
except Exception:  # pragma: no cover - fallback for environments without rapidfuzz
    rf_process = None
    rf_fuzz = None

from core.entities import VehicleRegistrationEntity, VehicleAttendanceEntity
from core.ports.plate_detector import PlateDetectorPort
from core.ports.repositories import VehicleRepositoryPort

MAX_VEHICLE_NUMBER_LENGTH = 10


def normalize_vehicle_number(vehicle: str) -> str:
    cleaned = "".join(ch for ch in str(vehicle).strip().upper() if ch.isalnum())
    return cleaned[:MAX_VEHICLE_NUMBER_LENGTH]


def match_registered_vehicle_number(
    detected_plate: str,
    db_plates: list[str],
    threshold: int = 80,
) -> tuple[bool, str | None, float]:
    detected_norm = normalize_vehicle_number(detected_plate)
    if not detected_norm or not db_plates:
        return False, None, 0.0

    normalized_db = [normalize_vehicle_number(p) for p in db_plates]

    if rf_process is not None and rf_fuzz is not None:
        result = rf_process.extractOne(detected_norm, normalized_db, scorer=rf_fuzz.ratio)
        if not result:
            return False, None, 0.0
        _, score, idx = result
        score = float(score)
    else:
        # Fallback when rapidfuzz is unavailable.
        scored = [
            (SequenceMatcher(None, detected_norm, plate).ratio() * 100.0, idx)
            for idx, plate in enumerate(normalized_db)
        ]
        score, idx = max(scored, key=lambda item: item[0], default=(0.0, -1))
        if idx < 0:
            return False, None, 0.0

    if score >= threshold:
        return True, normalize_vehicle_number(db_plates[idx]), score
    return False, None, score


def register_vehicle(repo: VehicleRepositoryPort, vehicle_number: str, owner_name: str, image_filename: str) -> VehicleRegistrationEntity:
    vn = normalize_vehicle_number(vehicle_number)
    if repo.exists(vn):
        raise ValueError("vehicle_exists")
    ent = VehicleRegistrationEntity(vehicle_number=vn, owner_name=owner_name, image_filename=image_filename)
    return repo.add_registration(ent)


def mark_attendance(repo: VehicleRepositoryPort, vehicle_number: str, image_filename: str):
    vn = normalize_vehicle_number(vehicle_number)
    if not repo.exists(vn):
        raise ValueError("vehicle_not_registered")
    owner = repo.get_owner(vn) or "Unknown"
    ent = VehicleAttendanceEntity(vehicle_number=vn, owner_name=owner, image_filename=image_filename)
    return repo.add_attendance(ent)


def mark_attendance_from_image(
    repo: VehicleRepositoryPort,
    detector: PlateDetectorPort,
    image_path: Path,
    image_filename: str,
) -> VehicleAttendanceEntity:
    detected_plate = detector.detect(image_path)
    if not detected_plate:
        raise ValueError("plate_not_detected")
    return mark_attendance(repo=repo, vehicle_number=detected_plate, image_filename=image_filename)


def list_registrations(repo: VehicleRepositoryPort, skip: int = 0, limit: int = 100):
    return repo.list_registrations(skip=skip, limit=limit)


def get_counts(repo: VehicleRepositoryPort):
    return repo.counts()
