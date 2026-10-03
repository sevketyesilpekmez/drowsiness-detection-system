from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


class VideoSourceError(RuntimeError):
    """Video kaynağı açılamadığında veya frame okunamadığında oluşur."""


class VideoSource:
    """
    Kamera indeksini ("0", "1") veya bir video dosyası yolunu
    ("sample_data/drowsy/test.mp4") aynı arayüzle açar.

    Desteklenen kaynak örnekleri:
        VIDEO_SOURCE=0
        VIDEO_SOURCE=1
        VIDEO_SOURCE=sample_data/drowsy/test.mp4

    Dosya kaynağı sonuna geldiğinde başa sarar (demo döngüsü için).
    Webcam koparsa otomatik yeniden bağlanmaz (bkz. README "bilinen
    sınırlamalar").
    """

    def __init__(
        self,
        source: str | int,
        *,
        requested_width: int | None = None,
        requested_height: int | None = None,
    ) -> None:
        """
        Args:
            source: Kamera indeksi veya video dosyası yolu.
            requested_width / requested_height: Yalnızca kamera kaynakları
                için (bkz. is_camera) — open() bunları cv2.VideoCapture'a
                CAP_PROP_FRAME_WIDTH/HEIGHT ile uygulamayı dener. Video
                dosyaları için yok sayılır (dosya zaten sabit bir
                çözünürlükte). `set()` platforma göre sessizce başarısız
                olabilir; sonucu zorlamıyoruz, gerçek uygulanan çözünürlük
                get_properties()'ten okunabilir.
        """
        self._source = self._normalize_source(source)
        self._capture: cv2.VideoCapture | None = None
        self._requested_width = requested_width
        self._requested_height = requested_height

    @staticmethod
    def _normalize_source(source: str | int) -> str | int:
        """"0"/"1" gibi sayısal stringleri kamera index'ine çevirir, diğer
        stringleri dosya yolu olarak bırakır."""
        if isinstance(source, int):
            return source

        stripped_source = source.strip()

        if stripped_source.isdigit():
            return int(stripped_source)

        return stripped_source

    @property
    def source(self) -> str | int:
        return self._source

    @property
    def is_camera(self) -> bool:
        return isinstance(self._source, int)

    @property
    def is_opened(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def open(self) -> None:
        """Video kaynağını açar. Zaten açıksa no-op (art arda çağrılırsa
        önceki handle sızmaz)."""
        if self.is_opened:
            return

        if not self.is_camera:
            video_path = Path(self._source)
            if not video_path.exists():
                raise VideoSourceError(f"Video dosyası bulunamadı: {video_path}")

        capture = cv2.VideoCapture(self._source)

        if not capture.isOpened():
            capture.release()
            raise VideoSourceError(f"Video kaynağı açılamadı: {self._source}")

        if self.is_camera:
            if self._requested_width is not None:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._requested_width)
            if self._requested_height is not None:
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._requested_height)

        self._capture = capture

    def read(self) -> np.ndarray:
        """Kaynaktan tek bir frame okur. Video dosyası sona geldiyse başa
        sarar; webcam için frame okunamazsa hata yükseltir."""
        if not self.is_opened:
            raise VideoSourceError("Video kaynağı açık değil. Önce open() çağrılmalı.")

        assert self._capture is not None
        success, frame = self._capture.read()

        if success and frame is not None:
            return frame

        if not self.is_camera:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = self._capture.read()

            if success and frame is not None:
                return frame

        raise VideoSourceError(f"Video kaynağından frame okunamadı: {self._source}")

    def get_properties(self) -> dict[str, Any]:
        """Video kaynağının temel özelliklerini döndürür (debug/log amaçlı)."""
        if not self.is_opened:
            return {
                "source": self._source,
                "is_camera": self.is_camera,
                "is_opened": False,
                "width": 0,
                "height": 0,
                "fps": 0.0,
                "frame_count": 0,
            }

        assert self._capture is not None
        return {
            "source": self._source,
            "is_camera": self.is_camera,
            "is_opened": True,
            "width": int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(self._capture.get(cv2.CAP_PROP_FPS)),
            "frame_count": int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        }

    def release(self) -> None:
        """Video kaynağını güvenli şekilde kapatır. Zaten kapalıysa no-op."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.release()
