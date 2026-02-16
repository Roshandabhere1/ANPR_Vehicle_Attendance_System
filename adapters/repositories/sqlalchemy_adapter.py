"""SQLAlchemy implementation of the VehicleRepositoryPort.

This adapter converts between SQLAlchemy ORM models and domain entities and
uses an injected SQLAlchemy session.
"""
from typing import List
from datetime import datetime

from core.ports.repositories import VehicleRepositoryPort
from core.entities import VehicleRegistrationEntity, VehicleAttendanceEntity

from database.models import VehicleRegistration as ORMRegistration, VehicleAttendance as ORMAttendance


class SQLAlchemyVehicleRepository(VehicleRepositoryPort):
    def __init__(self, db_session):
        # db_session is a SQLAlchemy Session (scoped per-request)
        self.db = db_session

    def exists(self, vehicle_number: str) -> bool:
        return self.db.query(ORMRegistration).filter(ORMRegistration.vehicle_number == vehicle_number).first() is not None

    def get_owner(self, vehicle_number: str):
        row = self.db.query(ORMRegistration).filter(ORMRegistration.vehicle_number == vehicle_number).first()
        return row.owner_name if row else None

    def add_registration(self, entity: VehicleRegistrationEntity) -> VehicleRegistrationEntity:
        orm = ORMRegistration(
            vehicle_number=entity.vehicle_number,
            owner_name=entity.owner_name,
            registration_date=entity.registration_date or datetime.now().strftime('%Y-%m-%d'),
            image_filename=entity.image_filename or None
        )
        self.db.add(orm)
        try:
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        self.db.refresh(orm)
        return VehicleRegistrationEntity(
            id=orm.id,
            vehicle_number=orm.vehicle_number,
            owner_name=orm.owner_name,
            registration_time=orm.registration_time,
            registration_date=orm.registration_date,
            image_filename=orm.image_filename,
            created_at=orm.created_at
        )

    def add_attendance(self, entity: VehicleAttendanceEntity) -> VehicleAttendanceEntity:
        orm = ORMAttendance(
            vehicle_number=entity.vehicle_number,
            owner_name=entity.owner_name,
            attendance_date=entity.attendance_date or datetime.now().strftime('%Y-%m-%d'),
            image_filename=entity.image_filename or None,
            confidence_score=entity.confidence_score or "0.85"
        )
        self.db.add(orm)
        try:
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        self.db.refresh(orm)
        return VehicleAttendanceEntity(
            id=orm.id,
            vehicle_number=orm.vehicle_number,
            owner_name=orm.owner_name,
            attendance_time=orm.attendance_time,
            attendance_date=orm.attendance_date,
            image_filename=orm.image_filename,
            confidence_score=orm.confidence_score,
            created_at=orm.created_at
        )

    def list_registrations(self, skip: int = 0, limit: int = 100) -> List[VehicleRegistrationEntity]:
        rows = self.db.query(ORMRegistration).offset(skip).limit(limit).all()
        return [VehicleRegistrationEntity(
            id=r.id,
            vehicle_number=r.vehicle_number,
            owner_name=r.owner_name,
            registration_time=r.registration_time,
            registration_date=r.registration_date,
            image_filename=r.image_filename,
            created_at=r.created_at
        ) for r in rows]

    def counts(self) -> dict:
        reg_count = self.db.query(ORMRegistration).count()
        att_count = self.db.query(ORMAttendance).count()
        return {"registrations": reg_count, "attendance": att_count}
