from pathlib import Path

from core.ports.plate_detector import PlateDetectorPort
from detect import detect_vehicle_number


class YoloTesseractPlateDetector(PlateDetectorPort):
    def detect(self, image_path: Path) -> str | None:
        return detect_vehicle_number(str(image_path))

