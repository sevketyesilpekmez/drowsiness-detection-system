from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from app.core.states import DetectionState

Clock = Callable[[], float]


class DrowsinessService:
    """
    Yüz ve göz durumuna göre kullanıcının uykulu olup olmadığına karar verir.

    Görüntü işlemez, kamera/OpenCV/MediaPipe/YOLO hakkında bir şey bilmez —
    yalnızca dışarıdan verilen ölçüm sonuçlarını kullanır.

    Üç ayrı soruyu üç ayrı state'e çevirir; bu ayrım kasıtlı:

        face_detected=False              -> NO_FACE
            Kadrajda kimse yok.

        measurement_valid=False (uzun)   -> UNVERIFIABLE
            Kişi var ama göz durumu okunamıyor (yüz çevrilmiş, göz
            bölgesi bulunamıyor, crop sınıflandırılamıyor).

        ölçüm alındı                     -> AWAKE / DROWSY
            Göz durumu gerçekten ölçüldü, karar buna dayanıyor.

    "Ölçemiyorum" ile "uyanık" aynı şey değildir. Bu ayrım olmadan sistem,
    gözü hiç göremediği karelerde sessizce AWAKE üretir ve uyuklayan bir
    sürücüyü uyarmadan geçer.

    Süre hesabında varsayılan time.monotonic(); testlerde sahte bir clock
    enjekte edilebilir.

    Thread-safe değildir — SharedState'in aksine yalnızca DetectionEngine'in
    tek background thread'inden, her frame'de tek bir update() çağrısıyla
    sürülmesi varsayılır.

    Not: face_detected/eyes_closed strict Python bool bekler. NumPy
    karşılaştırmaları (`ear < EAR_THRESHOLD`) numpy.bool_ döndürür, Python
    bool'un alt sınıfı değildir — çağıran taraf `bool(...)`'a çevirmeli.
    """

    def __init__(
        self,
        *,
        drowsiness_seconds: float = 2.0,
        closed_grace_period: float = 0.0,
        measurement_grace_seconds: float = 0.35,
        clock: Clock = time.monotonic,
    ) -> None:
        """
        Args:
            drowsiness_seconds: Gözlerin kesintisiz kaç saniye kapalı
                kalmasıyla DROWSY state'ine geçileceği.

            measurement_grace_seconds: Kadrajda kişi varken göz durumunun
                ölçülemediği (`measurement_valid=False`) bir serinin kaç
                saniyeye kadar tolere edileceği. Bu süre içinde önceki
                karar korunur (aktif bir "kapalı göz" serisi varsa o da
                kesilmez); süre aşılınca state UNVERIFIABLE olur.

                Neden saniye, neden kare değil: eski davranış "N ardışık
                kare" sayıyordu, ama bir karenin süresi FPS'e bağlı —
                30 FPS'te 2 kare 67 ms, 5 FPS'te 400 ms, yani aynı ayar
                makineye göre 6 kat farklı bir zaman toleransı demekti.
                Saniye cinsinden tolerans CPU/MPS/CUDA ve farklı kamera
                çözünürlükleri arasında aynı davranışı verir.

            closed_grace_period: Gerçek bir webcam testinde bulunan bir
                sorunun çözümü — tek bir kare "gözler açık" derse (Haar'ın
                gözü kaçırması vb.) birikmiş süre anında sıfırlanıyordu,
                bu DROWSY eşiğine çok yakın anlarda tekrar tekrar gözlendi
                (bkz. README "Zaman toleransı"). > 0 olduğunda, "kapalı"
                serisi sürerken gelen kısa bir "açık" okuması bu süre
                boyunca (saniye) yok sayılır; tolerans aşılırsa normal
                sıfırlanır. Varsayılan 0.0: eski davranış korunur.

                Aynı tolerans `face_detected=False` için de geçerli, ama
                yalnızca aktif bir "kapalı göz" serisi sürerken — yolo_
                pytorch'ta YOLO'nun tek karede kişiyi kaçırması artık
                alarmı erken kesmiyor. Aktif seri yokken face_detected=
                False hâlâ anında NO_FACE'e düşer.

            clock: Süre ölçümü fonksiyonu — gerçek çalışmada
                time.monotonic, testlerde sahte clock.
        """
        if drowsiness_seconds <= 0:
            raise ValueError("drowsiness_seconds sıfırdan büyük olmalıdır.")

        if closed_grace_period < 0:
            raise ValueError("closed_grace_period negatif olamaz.")

        if measurement_grace_seconds < 0:
            raise ValueError("measurement_grace_seconds negatif olamaz.")

        if not callable(clock):
            raise TypeError("clock çağrılabilir bir fonksiyon olmalıdır.")

        self._drowsiness_seconds = float(drowsiness_seconds)
        self._closed_grace_period = float(closed_grace_period)
        self._measurement_grace_seconds = float(measurement_grace_seconds)
        self._clock = clock

        # Ölçümün alınamadığı (measurement_valid=False) kesintisiz serinin
        # ne zaman başladığı. Ölçüm geri geldiğinde None'a döner.
        self._measurement_missing_since: float | None = None
        self._eyes_closed_since: float | None = None
        # "Kapalı" serisi sürerken gelen "açık" okumasının ne zaman
        # başladığını tutar (closed_grace_period ölçümü için). Seri
        # gerçekten kapalıyken None'a döner (bkz. update()).
        self._open_glitch_since: float | None = None

    @property
    def drowsiness_seconds(self) -> float:
        """DROWSY kararı için gereken göz kapalı kalma süresini döndürür."""
        return self._drowsiness_seconds

    @property
    def eyes_closed_since(self) -> float | None:
        """Gözlerin kapanmaya başladığı monotonic zamanı döndürür.
        Gözler açıkken veya yüz bulunamadığında None döner."""
        return self._eyes_closed_since

    def reset(self) -> None:
        """Birikmiş göz kapalı kalma süresini temizler."""
        self._eyes_closed_since = None
        self._open_glitch_since = None
        self._measurement_missing_since = None

    def update(
        self,
        *,
        face_detected: bool,
        eyes_closed: bool,
        confidence: float,
        measurement_valid: bool = True,
    ) -> dict[str, Any]:
        """
        Yeni ölçüm sonucuna göre sistem durumunu günceller.

        Args:
            face_detected: Frame içinde kullanılabilir bir kişi/yüz bulunup
                bulunmadığı. False -> NO_FACE (kadrajda kimse yok).
            eyes_closed: Tespit edilen gözlerin kapalı olup olmadığı.
                `measurement_valid=False` iken bu değer YOK SAYILIR.
            confidence: Ölçümün güven değeri. 0.0 ile 1.0 arasında olmalıdır.
            measurement_valid: Bu karede göz durumunun güvenilir şekilde
                ölçülüp ölçülemediği. Varsayılan True — ölçüm geçerliliğini
                ayrı raporlamayan çağıranlar (mediapipe/EAR backend'i)
                için eski davranış aynen korunur.

                False olduğunda kişi kadrajda demektir ama göz durumu
                bilinmiyordur: `measurement_grace_seconds` boyunca önceki
                karar korunur, süre aşılırsa UNVERIFIABLE üretilir. Bu
                ayrım olmadan sistem, gözü hiç ölçemediği halde "uyanık"
                raporlayabilir — bir uyku tespit sisteminde yanlış taraftaki
                hata budur.

        Returns:
            DetectionEngine'in REQUIRED_DETECT_FN_KEYS sözleşmesini
            karşılayan düz sözlük (state, is_drowsy, eyes_closed,
            confidence, closed_duration) — bkz. detection_engine.py.
        """
        self._validate_inputs(
            face_detected=face_detected,
            eyes_closed=eyes_closed,
            confidence=confidence,
            measurement_valid=measurement_valid,
        )

        now = self._clock()

        if not face_detected:
            # closed_grace_period toleransı NO_FACE için de geçerli, ama
            # sadece aktif bir "kapalı göz" serisi sürerken (bkz. __init__).
            tolerated = self._tolerated_glitch_result(now, confidence=confidence)
            if tolerated is not None:
                return tolerated

            self.reset()
            return self._build_result(
                state=DetectionState.NO_FACE,
                is_drowsy=False,
                eyes_closed=False,
                confidence=confidence,
                closed_duration=0.0,
            )

        if not measurement_valid:
            # Kişi kadrajda ama göz durumu okunamıyor. eyes_closed burada
            # bilerek kullanılmıyor — ölçüm yoksa "açık" da "kapalı" da
            # bilinmiyor demektir.
            if self._measurement_missing_since is None:
                self._measurement_missing_since = now

            missing_duration = now - self._measurement_missing_since

            if missing_duration >= self._measurement_grace_seconds:
                # Tolerans aşıldı: "ölçemiyorum"u AWAKE diye raporlamak
                # yerine açıkça UNVERIFIABLE'a geç. Birikmiş kapalı süre
                # de burada düşer, çünkü artık doğrulanamıyor.
                self.reset()
                return self._build_result(
                    state=DetectionState.UNVERIFIABLE,
                    is_drowsy=False,
                    eyes_closed=False,
                    confidence=confidence,
                    closed_duration=0.0,
                )

            # Tolerans içinde: aktif bir "kapalı göz" serisi varsa
            # kesilmeden sürsün (kısa bir ölçüm boşluğu alarmı iptal
            # etmemeli), yoksa önceki karar korunur.
            if self._eyes_closed_since is not None:
                closed_duration = max(0.0, now - self._eyes_closed_since)
                is_drowsy = closed_duration >= self._drowsiness_seconds
                return self._build_result(
                    state=DetectionState.DROWSY if is_drowsy else DetectionState.AWAKE,
                    is_drowsy=is_drowsy,
                    eyes_closed=True,
                    confidence=confidence,
                    closed_duration=closed_duration,
                )

            return self._build_result(
                state=DetectionState.AWAKE,
                is_drowsy=False,
                eyes_closed=False,
                confidence=confidence,
                closed_duration=0.0,
            )

        # Geçerli bir ölçüm geldi: ölçümsüz seri sayacı sıfırlanır.
        self._measurement_missing_since = None

        if not eyes_closed:
            # closed_grace_period > 0 ve devam eden bir "kapalı" seri varsa
            # önce toleransı kontrol et (bkz. __init__ docstring'i).
            tolerated = self._tolerated_glitch_result(now, confidence=confidence)
            if tolerated is not None:
                return tolerated
            # Tolerans yok/aşıldı: gözler gerçekten açılmış, normal sıfırlama.

            self.reset()
            return self._build_result(
                state=DetectionState.AWAKE,
                is_drowsy=False,
                eyes_closed=False,
                confidence=confidence,
                closed_duration=0.0,
            )

        # eyes_closed=True: devam eden (varsa) bir glitch toleransı iptal
        # edilir, seri kesintisiz kapalı kabul edilir.
        self._open_glitch_since = None

        if self._eyes_closed_since is None:
            self._eyes_closed_since = now

        closed_duration = max(0.0, now - self._eyes_closed_since)
        is_drowsy = closed_duration >= self._drowsiness_seconds

        return self._build_result(
            state=DetectionState.DROWSY if is_drowsy else DetectionState.AWAKE,
            is_drowsy=is_drowsy,
            eyes_closed=True,
            confidence=confidence,
            closed_duration=closed_duration,
        )

    def _tolerated_glitch_result(self, now: float, *, confidence: float) -> dict[str, Any] | None:
        """
        Aktif bir "kapalı göz" serisi sürerken gelen kısa bir glitch'in
        (eyes_closed=False ya da face_detected=False) tolerans penceresi
        içinde olup olmadığını kontrol eder. İçindeyse seri kesilmemiş
        gibi bir sonuç döndürür; tolerans yoksa/aşıldıysa None döner —
        çağıran taraf normal sıfırlama akışına devam etmeli.
        """
        if self._eyes_closed_since is None or self._closed_grace_period <= 0:
            return None

        if self._open_glitch_since is None:
            self._open_glitch_since = now

        glitch_duration = now - self._open_glitch_since
        if glitch_duration >= self._closed_grace_period:
            return None

        closed_duration = max(0.0, now - self._eyes_closed_since)
        is_drowsy = closed_duration >= self._drowsiness_seconds
        return self._build_result(
            state=DetectionState.DROWSY if is_drowsy else DetectionState.AWAKE,
            is_drowsy=is_drowsy,
            eyes_closed=True,
            confidence=confidence,
            closed_duration=closed_duration,
        )

    @staticmethod
    def _validate_inputs(
        *,
        face_detected: bool,
        eyes_closed: bool,
        confidence: float,
        measurement_valid: bool = True,
    ) -> None:
        """update() metoduna verilen değerlerin kontrata uygunluğunu doğrular."""
        if not isinstance(face_detected, bool):
            raise TypeError("face_detected bool olmalıdır.")

        if not isinstance(eyes_closed, bool):
            raise TypeError("eyes_closed bool olmalıdır.")

        if not isinstance(measurement_valid, bool):
            raise TypeError("measurement_valid bool olmalıdır.")

        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise TypeError("confidence sayısal bir değer olmalıdır.")

        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("confidence 0.0 ile 1.0 arasında olmalıdır.")

    @staticmethod
    def _build_result(
        *,
        state: str,
        is_drowsy: bool,
        eyes_closed: bool,
        confidence: float,
        closed_duration: float,
    ) -> dict[str, Any]:
        """DetectionEngine ile uyumlu sonuç sözlüğünü üretir."""
        return {
            "state": state,
            "is_drowsy": is_drowsy,
            "eyes_closed": eyes_closed,
            "confidence": round(float(confidence), 4),
            "closed_duration": round(float(closed_duration), 3),
        }
