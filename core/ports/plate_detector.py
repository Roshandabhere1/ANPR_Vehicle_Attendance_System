from abc import ABC, abstractmethod
from pathlib import Path


class PlateDetectorPort(ABC):
    @abstractmethod
    def detect(self, image_path: Path) -> str | None:
        """Return detected vehicle plate text or None if not detected."""
        raise NotImplementedError

