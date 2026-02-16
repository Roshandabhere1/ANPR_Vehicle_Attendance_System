from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class VehicleRegistrationEntity:
    id: Optional[int] = None
    vehicle_number: str = ""
    owner_name: str = ""
    registration_time: Optional[datetime] = None
    registration_date: Optional[str] = None
    image_filename: Optional[str] = None
    created_at: Optional[datetime] = None


@dataclass
class VehicleAttendanceEntity:
    id: Optional[int] = None
    vehicle_number: str = ""
    owner_name: str = ""
    attendance_time: Optional[datetime] = None
    attendance_date: Optional[str] = None
    image_filename: Optional[str] = None
    confidence_score: Optional[str] = None
    created_at: Optional[datetime] = None
