from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class PersonBox:
    """Bir frame içinde tespit edilen kişinin piksel cinsinden sınırlayıcı
    kutusu (bounding box)."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    def clip(self, *, width: int, height: int) -> PersonBox:
        """Kutuyu frame sınırları içine kırpar (YOLO bazen frame dışına
        taşan koordinatlar döndürebilir)."""
        return PersonBox(
            x1=max(0, min(self.x1, width - 1)),
            y1=max(0, min(self.y1, height - 1)),
            x2=max(0, min(self.x2, width)),
            y2=max(0, min(self.y2, height)),
            confidence=self.confidence,
        )

    @property
    def is_empty(self) -> bool:
        return self.x2 <= self.x1 or self.y2 <= self.y1


class PersonDetectorLike(Protocol):
    """YoloDrowsinessDetector'ın enjekte edebileceği kişi tespitçilerinin
    kontratı. Gerçek uygulama (UltralyticsPersonDetector) YOLOv8n kullanır;
    testlerde ultralytics/torch kurulu olmadan sahte bir nesne enjekte edilebilir."""

    def __call__(self, frame: np.ndarray) -> PersonBox | None: ...
