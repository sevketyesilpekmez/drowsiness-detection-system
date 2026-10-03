from __future__ import annotations

from typing import Any

import numpy as np

from app.services.alarm_service import AlarmService
from app.services.drowsiness_service import DrowsinessService
from app.services.landmark_service import LandmarkService


class DrowsinessDetector:
    """
    Landmark ölçümünü süre tabanlı uyku kararına (ve isteğe bağlı alarma)
    dönüştürür.

    __call__ uygulaması sayesinde DetectionEngine'e doğrudan detect_fn
    olarak verilebilir (bkz. detection_engine.py, DetectFn = Callable
    [[np.ndarray], dict[str, Any]]).

    Çıktısı DetectionEngine'in REQUIRED_DETECT_FN_KEYS sözleşmesini
    karşılar (state/is_drowsy/eyes_closed/confidence/closed_duration —
    DrowsinessService.update()'ten gelir) + fazladan ear/left_ear/right_ear
    alanları (kontrat bunu izin verir, extra alanlar reddedilmez).
    alarm_service verildiyse ayrıca alarm_triggered/alarm_active eklenir.
    """

    def __init__(
        self,
        *,
        landmark_service: LandmarkService,
        drowsiness_service: DrowsinessService,
        alarm_service: AlarmService | None = None,
        draw_landmarks: bool = True,
    ) -> None:
        self._landmark_service = landmark_service
        self._drowsiness_service = drowsiness_service
        self._alarm_service = alarm_service
        self._draw_landmarks = draw_landmarks

    def __call__(self, frame: np.ndarray) -> dict[str, Any]:
        landmark_result = self._landmark_service.analyze(frame)

        result = self._drowsiness_service.update(
            face_detected=landmark_result.face_detected,
            eyes_closed=landmark_result.eyes_closed,
            confidence=landmark_result.confidence,
        )

        result.update(
            {
                "ear": landmark_result.ear,
                "left_ear": landmark_result.left_ear,
                "right_ear": landmark_result.right_ear,
            }
        )

        if self._alarm_service is not None:
            # alarm_triggered: bu frame'de alarm yeni tetiklendi mi (tek
            # karelik edge, polling istemciler kaçırabilir).
            # alarm_active: alarm şu an aktif mi (level).
            # alarm_trigger_count: monoton sayaç, polling istemcilerin
            # tetiklenmeyi kaçırmadan takip etmesi için (bkz. AlarmService).
            alarm_triggered = self._alarm_service.update(is_drowsy=result["is_drowsy"])
            result["alarm_triggered"] = alarm_triggered
            result["alarm_active"] = self._alarm_service.is_active
            result["alarm_trigger_count"] = self._alarm_service.trigger_count
        else:
            result["alarm_triggered"] = False
            result["alarm_active"] = False
            result["alarm_trigger_count"] = 0

        if self._draw_landmarks:
            self._landmark_service.draw_eye_landmarks(frame, landmark_result)

        return result

    def reset(self) -> None:
        """
        Birikmiş KARAR durumunu temizler; MediaPipe kaynağına dokunmaz.

        `close()`'tan ayrı — bkz. YoloDrowsinessDetector.reset(). Offline
        değerlendirme aynı dedektörü birden çok klipte kullanıyor ve
        klipler arası sıfırlamayı `close()` ile yapmak FaceMesh'i
        kapatıp dedektörü kullanılamaz hale getirirdi.

        Çağrıldıktan sonra dedektör kullanılmaya devam edebilir.
        Idempotenttir.
        """
        self._drowsiness_service.reset()

        if self._alarm_service is not None:
            self._alarm_service.reset()

    def close(self) -> None:
        """Kaynakları serbest bırakır; bundan sonra dedektör KULLANILAMAZ.

        Idempotent: LandmarkService.close(), DrowsinessService.reset() ve
        AlarmService.reset() hepsi tekrar çağrılabilir olduğu için bu da
        güvenle tekrarlanabilir. Yalnızca karar durumunu sıfırlamak için
        `reset()` kullanın."""
        self.reset()
        self._landmark_service.close()
