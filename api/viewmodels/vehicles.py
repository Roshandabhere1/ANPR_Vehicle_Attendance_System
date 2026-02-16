from datetime import datetime

from pydantic import BaseModel


class RegisterVehicleResponseVM(BaseModel):
    success: bool
    message: str


class AttendanceSuccessResponseVM(BaseModel):
    success: bool
    message: str
    vehicle_number: str
    owner_name: str
    timestamp: datetime
    predicted_number: str | None = None
    detected_vehicle: str | None = None
    corrected_vehicle: str | None = None
    match_confidence: float | None = None


class AttendanceDeniedResponseVM(BaseModel):
    success: bool
    message: str
    predicted_number: str | None = None
    detected_vehicle: str
    corrected_vehicle: str | None = None
    match_confidence: float | None = None
