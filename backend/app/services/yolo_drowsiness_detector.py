from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from app.services.alarm_service import AlarmService
from app.services.drowsiness_service import DrowsinessService
from app.services.eye_classifier_service import EyeClassifierService
from app.services.eye_region_locator import EyeBox, EyeLocatorLike
from app.services.eye_state_smoother import EyeStateSmoother
from app.services.person_detector import PersonBox, PersonDetectorLike


class YoloDrowsinessDetector:
    """
    OpenCV + YOLO (person) + göz bölgesi bulucu + PyTorch (göz
    sınıflandırma) hattını DrowsinessService/AlarmService ile birleştirip
    tek bir detect_fn'e dönüştürür.

    Göz bölgesi bulucu değiştirilebilir (bkz. `EyeLocatorLike`): başta
    yalnızca Haar cascade vardı, ama başını öne eğen kullanıcıda (uyuklama
    pozu) zorlandığı görüldü (bkz. README "Baş eğikliği"). Artık
    `MediaPipeEyeLocator` `FallbackEyeLocator` içinde birincil, Haar yedek
    — bu sınıf değişmedi, enjekte edilen `eye_locator` sadece `locate()`
    sözleşmesine uyuyor olmalı.

    DrowsinessDetector (MediaPipe/EAR tabanlı) ile aynı rolü oynar, aynı
    __call__(frame) -> dict imzasına sahip — DetectionEngine'e doğrudan
    detect_fn olarak verilebilir.

    Ne YOLO'yu ne PyTorch'u doğrudan bilir — hepsi enjekte edilen
    PersonDetectorLike/EyeClassifierModelLike sözleşmeleri arkasında,
    bu yüzden sahte nesnelerle test edilebilir.

    İki göz nasıl birleştirilir (ham karar, smoothing öncesi — bkz.
    _raw_decision): iki göz bulunduysa yalnızca ikisi de kapalıysa ham
    karar "kapalı". Tek göz bulunduysa, confidence `single_eye_confidence_
    threshold`'u (varsayılan 0.80) geçmiyorsa bu kare smoother'a geçersiz
    (missing) olarak bildirilir.

    Zamansal yumuşatma: gerçek testte (README "Zaman toleransı") Haar'ın
    ara sıra tek karede gözü kaçırması ham kararı dalgalandırıp DROWSY
    eşiğine yakın anlarda süreyi sıfırlıyordu. Ham karar artık önce
    `EyeStateSmoother`'dan geçip yumuşatılmış karara dönüşüyor. Sonuçta
    hem ham (`raw_eyes_closed`) hem yumuşatılmış (`smoothed_eyes_closed`)
    ayrı raporlanır; `eyes_closed` alanı yumuşatılmış kararı temsil eder.

    "Kişi var mı" ile "gözü ölçebildim mi" iki AYRI sinyal olarak
    DrowsinessService'e geçer. Bu mantık üç kez değişti:

      - En başta göz bulunamayan HER kare doğrudan NO_FACE'e düşüyordu.
        Bu, Haar'ın tek karede gözü kaçırmasında bile birikmiş süreyi
        sıfırlayıp EyeStateSmoother'ın toleransını by-pass ediyordu.
      - Sonra tam tersine çevrildi: yalnızca kişi kutusuna bakıldı. Bu da
        güvenlik açısından yanlış yönde bir hata üretti — sürücü yüzünü
        kameradan çevirdiğinde gövdesi görünmeye devam ettiği için
        face_detected=True kalıyor, göz ölçümü hiç alınamadığı halde
        sistem NO_FACE yerine AWAKE ("uyanık") raporluyordu.
      - Üçüncü hali göz KUTUSU bulunmasını yeterli sayıyordu
        (`eye_locator_miss_tolerance` ardışık kare). Buradaki incelik
        şuydu: kutu bulunup da içindeki crop'ların hepsi geçersiz
        çıkabilir (çok küçük kutu, sınıflandırma hatası, düşük güvenli
        tek göz). O durumda `classifications` boş kalıyor, ama kutu
        bulunduğu için kaçırma sayacı sıfırlanıyor ve sistem yine
        AWAKE üretiyordu — yani aynı hata bir katman derinde duruyordu.

    Şimdiki davranış: karar `measurement_valid` sinyaline dayanıyor ve bu
    sinyal göz kutusunun varlığına değil, GEÇERLİ BİR SINIFLANDIRMA
    alınıp alınmadığına bakıyor (`_raw_decision`'ın üçüncü dönüş değeri —
    smoother'a oy verilebilen kare, ölçülebilmiş kare demektir).
    Tolerans da artık kare değil saniye cinsinden ve
    DrowsinessService'in içinde (bkz. `measurement_grace_seconds`):
    kısa boşluklar seriyi bozmaz, kalıcı ölçümsüzlük UNVERIFIABLE olur.
    """

    def __init__(
        self,
        *,
        person_detector: PersonDetectorLike,
        eye_locator: EyeLocatorLike,
        eye_classifier: EyeClassifierService,
        drowsiness_service: DrowsinessService,
        eye_state_smoother: EyeStateSmoother | None = None,
        alarm_service: AlarmService | None = None,
        draw_boxes: bool = True,
        max_eyes_per_frame: int = 2,
        single_eye_confidence_threshold: float = 0.80,
    ) -> None:
        """
        Args:
            single_eye_confidence_threshold: Tek göz bulunduğunda, o gözün
                oyunun geçerli sayılması için gereken minimum güven. Bu
                eşiğin altındaki tek göz okuması "ölçüm alınamadı" sayılır
                (bkz. `_raw_decision`).

        Not: Ölçümsüz karelerin ne kadar tolere edileceği artık burada
        değil, `DrowsinessService(measurement_grace_seconds=...)` içinde
        ve saniye cinsinden — eski `eye_locator_miss_tolerance` (kare
        sayısı) FPS'e bağlı olduğu için makineden makineye farklı
        davranıyordu.
        """
        if max_eyes_per_frame <= 0:
            raise ValueError("max_eyes_per_frame sıfırdan büyük olmalıdır.")
        if not 0.0 <= single_eye_confidence_threshold <= 1.0:
            raise ValueError("single_eye_confidence_threshold 0.0 ile 1.0 arasında olmalıdır.")

        self._person_detector = person_detector
        self._eye_locator = eye_locator
        self._eye_classifier = eye_classifier
        self._drowsiness_service = drowsiness_service
        self._eye_state_smoother = eye_state_smoother if eye_state_smoother is not None else EyeStateSmoother()
        self._alarm_service = alarm_service
        self._draw_boxes = draw_boxes
        self._max_eyes_per_frame = max_eyes_per_frame
        self._single_eye_confidence_threshold = single_eye_confidence_threshold

    def __call__(self, frame: np.ndarray) -> dict[str, Any]:
        # Adım bazlı süre ölçümü (ms): FPS düşüklüğünün hangi adımdan
        # kaynaklandığını /api/detection/status üzerinden görebilmek için.
        person_detect_start = time.perf_counter()
        person_box = self._person_detector(frame)
        person_detect_ms = (time.perf_counter() - person_detect_start) * 1000

        eye_pipeline_start = time.perf_counter()
        eye_boxes: tuple[EyeBox, ...] = ()
        classifications = []

        if person_box is not None and not person_box.is_empty:
            eye_boxes = self._eye_locator.locate(frame, search_region=person_box)

            for eye_box in eye_boxes[: self._max_eyes_per_frame]:
                crop = self._crop(frame, eye_box)
                if crop is None:
                    continue

                try:
                    classifications.append(self._eye_classifier.classify(crop))
                except (ValueError, TypeError):
                    # Yanlış pozitif / çok küçük kutu beklenen bir durum —
                    # tek bir kötü tespit tüm frame'i bozmasın, o göz atlanır.
                    continue
        eye_pipeline_ms = (time.perf_counter() - eye_pipeline_start) * 1000

        person_found = person_box is not None and not person_box.is_empty
        eye_region_found = len(eye_boxes) > 0

        raw_eyes_closed, raw_confidence, valid_for_smoothing = self._raw_decision(classifications)

        smoothed_eyes_closed = self._eye_state_smoother.update(
            valid=valid_for_smoothing,
            eyes_closed=raw_eyes_closed,
            confidence=raw_confidence,
        )

        # İki ayrı sinyal (bkz. sınıf docstring'i):
        #   face_detected     -> kadrajda kişi var mı
        #   measurement_valid -> göz durumu GERÇEKTEN ölçülebildi mi
        # measurement_valid, göz kutusunun bulunmasına DEĞİL geçerli bir
        # sınıflandırma alınmasına bakar: kutu bulunup crop'ların hepsi
        # geçersiz çıkarsa (çok küçük kutu, sınıflandırma hatası, düşük
        # güvenli tek göz) ölçüm yok demektir.
        result = self._drowsiness_service.update(
            face_detected=person_found,
            eyes_closed=smoothed_eyes_closed,
            confidence=raw_confidence,
            measurement_valid=valid_for_smoothing,
        )

        result["eyes_examined"] = len(classifications)
        # Gözlemlenebilirlik: "kişi görülüyor ama yüz/göz okunamıyor"
        # durumunun /api/detection/status üzerinden ayırt edilebilmesi için.
        result["person_detected"] = person_found
        result["eye_region_found"] = eye_region_found
        result["face_detected"] = person_found
        result["measurement_valid"] = valid_for_smoothing
        result["raw_eyes_closed"] = raw_eyes_closed
        result["smoothed_eyes_closed"] = smoothed_eyes_closed
        result["raw_confidence"] = round(float(raw_confidence), 4)
        result["person_detect_ms"] = round(person_detect_ms, 1)
        result["eye_pipeline_ms"] = round(eye_pipeline_ms, 1)

        if self._alarm_service is not None:
            alarm_triggered = self._alarm_service.update(is_drowsy=result["is_drowsy"])
            result["alarm_triggered"] = alarm_triggered
            result["alarm_active"] = self._alarm_service.is_active
            result["alarm_trigger_count"] = self._alarm_service.trigger_count
        else:
            result["alarm_triggered"] = False
            result["alarm_active"] = False
            result["alarm_trigger_count"] = 0

        if self._draw_boxes:
            self._draw(frame, person_box=person_box, eye_boxes=eye_boxes)

        return result

    def reset(self) -> None:
        """
        Birikmiş KARAR durumunu temizler; kaynakları serbest BIRAKMAZ.

        `close()`'tan ayrı olmasının sebebi somut: offline değerlendirme
        aynı dedektörü birden çok klipte kullanır ve her klip arasında
        önceki klibin kapalı-göz serisinin sızmaması gerekir. Bunu
        `close()` ile yapmak MediaPipe FaceMesh'i kapatır ve dedektör
        ikinci klipte tamamen çalışmaz hale gelir (bkz.
        scripts/evaluate_system.py).

        Çağrıldıktan sonra dedektör kullanılmaya devam edebilir.
        Idempotenttir.
        """
        self._drowsiness_service.reset()
        self._eye_state_smoother.reset()

        if self._alarm_service is not None:
            self._alarm_service.reset()

    def close(self) -> None:
        """Kaynakları serbest bırakır; bundan sonra dedektör KULLANILAMAZ.

        person_detector/eye_classifier'ın (YOLO/torch) kapatılması bu
        sınıfın sorumluluğunda değil, GC ile temizlenirler.

        eye_locator `MediaPipeEyeLocator` (ya da onu saran
        `FallbackEyeLocator`) ise gerçek bir MediaPipe FaceMesh kaynağı
        tutar ve kapatılmalı — `close()` var mı diye kontrol edilip
        (Haar/test double'larında yoksa sessizce atlanır) çağrılıyor.

        Yalnızca karar durumunu sıfırlamak için `reset()` kullanın."""
        self.reset()

        close_eye_locator = getattr(self._eye_locator, "close", None)
        if callable(close_eye_locator):
            close_eye_locator()

    def _raw_decision(self, classifications: list) -> tuple[bool, float, bool]:
        """
        Sınıflandırma listesinden HAM (smoothing öncesi) kararı üretir.

        Returns:
            (raw_eyes_closed, raw_confidence, valid_for_smoothing) — bkz.
            sınıf docstring'indeki "İki göz nasıl birleştirilir" notu.
            valid_for_smoothing=False olduğunda raw_eyes_closed her zaman
            False'tur (anlamsız/güvenilmez bir kararın smoother'a "kapalı"
            olarak sızmasını önlemek için).
        """
        if len(classifications) >= 2:
            two = classifications[:2]
            raw_eyes_closed = all(c.is_closed for c in two)
            raw_confidence = sum(c.confidence for c in two) / len(two)
            return raw_eyes_closed, raw_confidence, True

        if len(classifications) == 1:
            single = classifications[0]
            if single.confidence >= self._single_eye_confidence_threshold:
                return single.is_closed, single.confidence, True
            # Düşük güvenli tek göz: bu kareyi smoother için "geçersiz"
            # (eksik) say — bkz. sınıf docstring'i.
            return False, single.confidence, False

        return False, 0.0, False

    @staticmethod
    def _crop(frame: np.ndarray, box: EyeBox) -> np.ndarray | None:
        crop = frame[box.y1 : box.y2, box.x1 : box.x2]
        return crop if crop.size > 0 else None

    @staticmethod
    def _draw(frame: np.ndarray, *, person_box: PersonBox | None, eye_boxes: tuple[EyeBox, ...]) -> None:
        if person_box is not None:
            cv2.rectangle(frame, (person_box.x1, person_box.y1), (person_box.x2, person_box.y2), (0, 200, 0), 2)

        for eye_box in eye_boxes:
            cv2.rectangle(frame, (eye_box.x1, eye_box.y1), (eye_box.x2, eye_box.y2), (0, 255, 255), 1)
