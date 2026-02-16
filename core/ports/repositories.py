from abc import ABC, abstractmethod
from typing import List, Optional
from core.entities import VehicleRegistrationEntity, VehicleAttendanceEntity


class VehicleRepositoryPort(ABC):
    @abstractmethod
    def exists(self, vehicle_number: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_owner(self, vehicle_number: str) -> Optional[str]:
        raise NotImplementedError

    @abstractmethod
    def add_registration(self, entity: VehicleRegistrationEntity) -> VehicleRegistrationEntity:
        raise NotImplementedError

    @abstractmethod
    def add_attendance(self, entity: VehicleAttendanceEntity) -> VehicleAttendanceEntity:
        raise NotImplementedError

    @abstractmethod
    def list_registrations(self, skip: int = 0, limit: int = 100) -> List[VehicleRegistrationEntity]:
        raise NotImplementedError

    @abstractmethod
    def counts(self) -> dict:
        raise NotImplementedError
