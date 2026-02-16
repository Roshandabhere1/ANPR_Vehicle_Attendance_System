from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError

from adapters.repositories.sqlalchemy_adapter import SQLAlchemyVehicleRepository
from api.dependencies import get_plate_detector, get_vehicle_repo, require_api_key
from api.upload_service import save_upload_file
from api.viewmodels.vehicles import (
    AttendanceDeniedResponseVM,
    AttendanceSuccessResponseVM,
    RegisterVehicleResponseVM,
)
from core.ports.plate_detector import PlateDetectorPort
from core.usecases.vehicle_service import (
    mark_attendance,
    match_registered_vehicle_number,
    normalize_vehicle_number,
    register_vehicle,
)

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/register", response_model=RegisterVehicleResponseVM)
async def api_register(
    image: UploadFile = File(...),
    vehicle_number: str = Form(...),
    owner_name: str = Form(""),
    repo: SQLAlchemyVehicleRepository = Depends(get_vehicle_repo),
):
    vehicle_number = normalize_vehicle_number(vehicle_number)
    owner_name = owner_name.strip() or vehicle_number

    if not vehicle_number:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Vehicle number is required.")

    filename, _ = await save_upload_file(image, prefix="register", vehicle_number=vehicle_number)
    try:
        register_vehicle(
            repo=repo,
            vehicle_number=vehicle_number,
            owner_name=owner_name,
            image_filename=filename,
        )
    except ValueError as exc:
        if str(exc) == "vehicle_exists":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Vehicle {vehicle_number} already registered.",
            ) from exc
        raise
    except IntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Vehicle {vehicle_number} already registered.",
        ) from exc

    return RegisterVehicleResponseVM(
        success=True,
        message=f"Vehicle {vehicle_number} registered successfully!",
    )


@router.post("/mark-attendance", response_model=AttendanceSuccessResponseVM)
async def api_mark_attendance(
    image: UploadFile = File(...),
    repo: SQLAlchemyVehicleRepository = Depends(get_vehicle_repo),
    detector: PlateDetectorPort = Depends(get_plate_detector),
):
    filename, filepath = await save_upload_file(image, prefix="attendance")
    detected_raw = detector.detect(filepath)
    if not detected_raw:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not detect vehicle number from image. Please upload a clearer image.",
        )
    detected_plate = normalize_vehicle_number(detected_raw)

    registrations = repo.list_registrations(skip=0, limit=100000)
    registered_numbers = [row.vehicle_number for row in registrations if row.vehicle_number]
    found, corrected_plate, confidence = match_registered_vehicle_number(
        detected_plate,
        registered_numbers,
        threshold=80,
    )
    final_plate = corrected_plate if found and corrected_plate else detected_plate

    try:
        attendance = mark_attendance(
            repo=repo,
            vehicle_number=final_plate,
            image_filename=filename,
        )
    except ValueError as exc:
        if str(exc) == "vehicle_not_registered":
            denied_vm = AttendanceDeniedResponseVM(
                success=False,
                message=f"Vehicle {final_plate} not registered",
                predicted_number=detected_plate,
                detected_vehicle=detected_plate,
                corrected_vehicle=corrected_plate,
                match_confidence=round(confidence, 2),
            )
            return JSONResponse(status_code=status.HTTP_403_FORBIDDEN, content=denied_vm.model_dump())
        raise

    return AttendanceSuccessResponseVM(
        success=True,
        message=f"Attendance marked for {attendance.vehicle_number}",
        vehicle_number=attendance.vehicle_number,
        owner_name=attendance.owner_name,
        timestamp=datetime.now(),
        predicted_number=detected_plate,
        detected_vehicle=detected_plate,
        corrected_vehicle=final_plate,
        match_confidence=round(confidence, 2),
    )
