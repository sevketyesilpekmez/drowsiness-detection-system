"""
Video dosyasından kare okuma — değerlendirme hattının tek OpenCV bağımlı
parçası.

`evaluation/__init__.py` bu modülü import ETMEZ, dolayısıyla
`from evaluation.metrics import ...` hâlâ saf kalır ve cv2 kurulu
olmayan bir CI ortamında çalışır. cv2 yalnızca gerçekten video okunacağı
zaman yüklenir.

`app.services.video_source.VideoSource` bilerek kullanılmıyor: o sınıf
canlı yayın için tasarlandı ve dosya sonuna gelince BAŞA SARIYOR
(demo döngüsü için). Bir değerlendirme koşusunda bu sonsuz döngü
demektir.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


def probe_video(path: str | Path) -> VideoInfo:
    """Videoyu açmadan önce temel özelliklerini okur."""
    import cv2

    video_path = Path(path)
    if not video_path.exists():
        raise ValueError(f"Video bulunamadı: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Video açılamadı: {video_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()

    if fps <= 0:
        raise ValueError(
            f"{video_path}: kare hızı okunamadı (fps={fps}). Etiket dosyasında "
            "'fps' alanını açıkça belirtin — tüm zaman hesabı buna dayanıyor."
        )

    return VideoInfo(path=video_path, fps=fps, frame_count=frame_count, width=width, height=height)


@contextmanager
def read_frames(path: str | Path) -> Iterator[Iterator]:
    """
    Videonun TÜM karelerini sırayla veren bir generator döndürür.

    Kare atlama burada YAPILMAZ — doğru zaman damgası için gerçek kare
    indeksleri gerekir, seyreltmeyi `replay()` yapar (bkz. runner.py).

    Context manager: generator tüketilmeden bırakılsa bile VideoCapture
    serbest bırakılır.
    """
    import cv2

    video_path = Path(path)
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Video açılamadı: {video_path}")

    def generate():
        while True:
            success, frame = capture.read()
            if not success or frame is None:
                return
            yield frame

    try:
        yield generate()
    finally:
        capture.release()
