from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Callable

import cv2
import numpy as np

from app.core.shared_state import SharedState
from app.core.states import DetectionState
from app.services.video_source import VideoSource, VideoSourceError

logger = logging.getLogger(__name__)

DetectFn = Callable[[np.ndarray], dict[str, Any]]

# Tek bir karenin anlık 1/elapsed'i çok gürültülü olduğu için son bu kadar
# karenin süresi biriktirilip ortalaması alınıyor (bkz. _run).
FPS_AVERAGE_WINDOW = 30

# detect_fn'in dönmesi ZORUNLU alanlar. Biri eksikse sessizce geçmiyor,
# MODEL_ERROR üretiliyor (bkz. _run). "fps"/"timestamp" burada yok çünkü
# onları engine ve SharedState kendisi ekliyor.
REQUIRED_DETECT_FN_KEYS = frozenset(
    {
        "state",
        "is_drowsy",
        "eyes_closed",
        "confidence",
        "closed_duration",
    }
)


def dummy_detect_fn(frame: np.ndarray) -> dict[str, Any]:
    """Gerçek bir detect_fn bağlanmadığında kullanılan basit stub — her
    zaman "uyanık" döner. Yeni bir detect_fn yazarken şablon olarak
    kullanılabilir."""
    return {
        "state": DetectionState.AWAKE,
        "is_drowsy": False,
        "eyes_closed": False,
        "confidence": 0.0,
        "closed_duration": 0.0,
    }


class DetectionEngine:
    """
    Video kaynağını arka planda sürekli okur, detect_fn ile işler ve
    sonucu SharedState içinde saklar.

    Kameranın tek sahibi bu sınıf — Flask route'ları kameraya doğrudan
    dokunmaz, yalnızca SharedState'ten okur.

    YOLO/MediaPipe/DrowsinessService'in ne olduğunu bilmez, hepsi
    detect_fn parametresinin arkasında (bkz. dummy_detect_fn) — böylece
    algılama mantığı değişse bile bu dosyaya dokunulmuyor.

    Tek bir okumanın/tespitin patlaması thread'i öldürmez, hata
    shared_state'e yazılıp kısa bir bekleme sonrası döngü devam eder.
    Video kaynağı gerçekten koparsa otomatik yeniden bağlanma yok
    (bilinçli bir kapsam kararı, bkz. README).

    Hata ayrımı:
    - video_source.read() patlarsa -> CAMERA_ERROR
    - detect_fn() patlarsa/sözleşmeyi bozarsa -> MODEL_ERROR
    - JPEG encode patlarsa -> SYSTEM_ERROR
    """

    def __init__(
        self,
        *,
        video_source: VideoSource,
        shared_state: SharedState,
        detect_fn: DetectFn = dummy_detect_fn,
        jpeg_quality: int = 85,
        retry_delay: float = 1.0,
    ) -> None:
        """
        Args:
            video_source: Webcam veya video dosyasını yöneten VideoSource.
            shared_state: Son frame ve tespit sonucunun tutulduğu SharedState.
            detect_fn: frame alıp sonuç dict'i döndüren tespit fonksiyonu.
            jpeg_quality: Stream için JPEG kalite seviyesi (0-100).
            retry_delay: Bir hatadan sonra tekrar denemeden önce beklenecek
                saniye. stop() bu bekleme sırasında çağrılırsa hemen çıkar.
        """
        if not 0 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality 0 ile 100 arasında olmalıdır.")
        if retry_delay < 0:
            raise ValueError("retry_delay negatif olamaz.")

        self._video_source = video_source
        self._shared_state = shared_state
        self._detect_fn = detect_fn
        self._jpeg_quality = jpeg_quality
        self._retry_delay = retry_delay

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # Ortalama FPS için son FPS_AVERAGE_WINDOW karenin süresi.
        self._frame_durations: deque[float] = deque(maxlen=FPS_AVERAGE_WINDOW)

    @property
    def is_running(self) -> bool:
        """Thread'in gerçekten hayatta olup olmadığı."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """
        Video kaynağını açar ve background thread'i başlatır.

        Kamera açılamazsa exception fırlatmaz — shared_state'i
        CAMERA_ERROR'a çevirir, thread'i hiç başlatmadan döner. Böylece
        Flask uygulaması kamerasız bir makinede bile ayakta kalır.
        """
        if self.is_running:
            logger.warning("DetectionEngine zaten çalışıyor, start() yok sayıldı.")
            return

        try:
            self._video_source.open()
        except VideoSourceError as exc:
            logger.error("Video kaynağı açılamadı: %s", exc)
            self._shared_state.set_error(str(exc), state=DetectionState.CAMERA_ERROR)
            return

        # Aynı örnek stop() sonrası tekrar start() edilirse önceki
        # oturumdan kalan süreler yeni ortalamayı bozmasın diye temizlenir.
        self._frame_durations.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="detection-engine",
            daemon=True,
        )
        self._thread.start()
        logger.info("DetectionEngine başlatıldı (kaynak: %s).", self._video_source.source)

    def stop(self, timeout: float = 5.0) -> None:
        """
        Thread'i durdurur, video kaynağını ve detect_fn'in tuttuğu
        kaynakları (varsa) serbest bırakır. start() hiç çağrılmamış olsa
        bile güvenle çağrılabilir.

        Sıra: thread join -> video_source.release() -> detect_fn.close()
        (varsa) — kaynaklar hâlâ kullanımdayken serbest bırakılmasın diye.

        Args:
            timeout: Thread'in kapanması için en fazla beklenecek saniye.
        """
        self._stop_event.set()

        if (
            self._thread is not None
            and self._thread.is_alive()
            and threading.current_thread() is not self._thread
        ):
            self._thread.join(timeout=timeout)

        self._video_source.release()

        # detect_fn kaynak tutuyorsa (örn. MediaPipe FaceMesh) burada
        # kapatılır; kaynaksız detect_fn'lerde close() yok, no-op.
        close_fn = getattr(self._detect_fn, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                logger.exception("detect_fn.close() sırasında hata oluştu.")

        self._thread = None
        logger.info("DetectionEngine durduruldu.")

    def _run(self) -> None:
        last_tick = time.monotonic()

        while not self._stop_event.is_set():
            read_start = time.monotonic()
            try:
                frame = self._video_source.read()
            except VideoSourceError as exc:
                logger.warning("Kamera okuma hatası: %s", exc)
                self._shared_state.set_error(str(exc), state=DetectionState.CAMERA_ERROR)
                self._stop_event.wait(self._retry_delay)
                continue
            camera_read_ms = (time.monotonic() - read_start) * 1000

            try:
                result = dict(self._detect_fn(frame))
                missing_keys = REQUIRED_DETECT_FN_KEYS - result.keys()
                if missing_keys:
                    raise ValueError(
                        f"detect_fn sözleşmeyi bozdu, eksik alan(lar): {sorted(missing_keys)}"
                    )
            except Exception as exc:
                logger.exception("detect_fn içinde hata oluştu.")
                self._shared_state.set_error(str(exc), state=DetectionState.MODEL_ERROR)
                self._stop_event.wait(self._retry_delay)
                continue

            now = time.monotonic()
            elapsed = now - last_tick
            last_tick = now
            result["fps"] = self._record_frame_and_get_average_fps(elapsed)
            # Kamera okuma süresi backend'den bağımsız her zaman burada
            # ölçülür; yolo_pytorch ayrıca person_detect_ms/eye_pipeline_ms
            # de ekler (bkz. YoloDrowsinessDetector.__call__).
            result["camera_read_ms"] = round(camera_read_ms, 1)

            encode_start = time.monotonic()
            try:
                jpeg_frame = self._encode_frame(frame)
            except Exception as exc:
                logger.exception("Frame encode hatası.")
                self._shared_state.set_error(str(exc), state=DetectionState.SYSTEM_ERROR)
                self._stop_event.wait(self._retry_delay)
                continue
            result["jpeg_encode_ms"] = round((time.monotonic() - encode_start) * 1000, 1)

            self._shared_state.update(frame=jpeg_frame, result=result)

        logger.info("DetectionEngine döngüsü sona erdi.")

    def _record_frame_and_get_average_fps(self, elapsed: float) -> float:
        """
        Bir karenin süresini son FPS_AVERAGE_WINDOW kareye ekler, ortalama
        fps'i döndürür.

        Süreleri ortalayıp sonra tersini almak (1/ortalama_süre), her
        karenin anlık fps değerinin aritmetik ortalamasını almaktan farklı
        ve daha doğru: nadir ama çok yavaş bir kare, ikinci yöntemde
        ortalamayı gerçek verimden daha kötü gösterirdi (bkz.
        test_average_fps_is_not_skewed_by_one_slow_frame).
        """
        if elapsed > 0:
            self._frame_durations.append(elapsed)

        if not self._frame_durations:
            return 0.0

        average_elapsed = sum(self._frame_durations) / len(self._frame_durations)
        return round(1.0 / average_elapsed, 1) if average_elapsed > 0 else 0.0

    def _encode_frame(self, frame: np.ndarray) -> bytes:
        """OpenCV frame'ini JPEG bytes verisine dönüştürür (MJPEG stream
        endpoint'i numpy array değil, bytes gönderir)."""
        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
        )

        if not success:
            raise RuntimeError("Frame JPEG formatına dönüştürülemedi.")

        return encoded.tobytes()
