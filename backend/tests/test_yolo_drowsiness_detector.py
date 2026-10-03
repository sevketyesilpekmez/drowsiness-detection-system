from __future__ import annotations

import time

import numpy as np
import pytest

from app.core.detection_engine import REQUIRED_DETECT_FN_KEYS, DetectionEngine
from app.core.shared_state import SharedState
from app.core.states import DetectionState
from app.services.alarm_service import AlarmService
from app.services.drowsiness_service import DrowsinessService
from app.services.eye_classifier_service import CLOSED_EYES, OPEN_EYES, EyeClassifierService
from app.services.eye_region_locator import EyeRegionLocator
from app.services.eye_state_smoother import EyeStateSmoother
from app.services.person_detector import PersonBox
from app.services.yolo_drowsiness_detector import YoloDrowsinessDetector


class FakePersonDetector:
    def __init__(self, *, box=None):
        self.box = box
        self.call_count = 0

    def __call__(self, frame):
        self.call_count += 1
        return self.box


class FakeCascade:
    """test_eye_region_locator.py'deki ile aynı sahte cascade — burada
    EyeRegionLocator'ı gerçek Haar cascade'e ihtiyaç duymadan
    kullanabilmek için."""

    def __init__(self, *, detections=()):
        self.detections = detections

    def detectMultiScale(self, image, scaleFactor, minNeighbors, minSize):
        return self.detections


class FakeEyeModel:
    """EyeClassifierModelLike'a uyan, göz kutusuna göre farklı sonuç
    döndürebilen sahte model — test senaryosuna göre `responses` sırayla
    tüketilir, biterse son elemanı tekrar döndürür."""

    def __init__(self, *, responses):
        self._responses = list(responses)
        self._call_index = 0

    def __call__(self, eye_crop):
        index = min(self._call_index, len(self._responses) - 1)
        self._call_index += 1
        return self._responses[index]


def make_frame(width=200, height=150) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


PERSON_BOX = PersonBox(x1=20, y1=10, x2=180, y2=140, confidence=0.9)
TWO_EYE_DETECTIONS = [(30, 20, 20, 20), (100, 20, 20, 20)]  # (x, y, w, h) — search_region'a göre local


def make_pass_through_smoother() -> EyeStateSmoother:
    """Bu test dosyasındaki testlerin çoğu, YoloDrowsinessDetector'ın HAM
    karar mantığını (iki göz AND kuralı, tek göz eşiği, vb.) TEK bir
    çağrıyla doğruluyor — zamansal yumuşatmayı değil. Gerçek varsayılan
    EyeStateSmoother devrede olsaydı bu testlerin çoğu ilk çağrılarda
    hep 'henüz kararsız' (False) dönerdi.

    Bu yüzden burada smoother'ı etkisizleştiren bir "pass-through"
    konfigürasyon kullanılıyor: `minimum_valid_seconds=0` başlangıç
    korumasını kapatır, çok kısa bir pencere de her ölçümün tek başına
    kalmasını sağlar — yani ham karar aynı çağrıda yansır.

    Zamansal yumuşatmanın KENDİSİ ayrı "test_smoother_*" testlerinde ve
    tests/test_eye_state_smoother.py içinde gerçek ayarlarla test
    ediliyor."""
    return EyeStateSmoother(
        window_seconds=0.001,
        closed_ratio_threshold=0.5,
        missing_tolerance_seconds=0.0,
        minimum_valid_seconds=0.0,
    )


def make_detector(
    *,
    person_box=PERSON_BOX,
    eye_detections=TWO_EYE_DETECTIONS,
    eye_responses=((OPEN_EYES, 0.9), (OPEN_EYES, 0.9)),
    drowsiness_seconds=2.0,
    clock=None,
    alarm_service=None,
    draw_boxes=True,
    eye_state_smoother=None,
    single_eye_confidence_threshold=0.80,
):
    person_detector = FakePersonDetector(box=person_box)
    eye_locator = EyeRegionLocator(cascade=FakeCascade(detections=eye_detections))
    eye_classifier = EyeClassifierService(model=FakeEyeModel(responses=eye_responses))
    drowsiness_service = DrowsinessService(drowsiness_seconds=drowsiness_seconds, clock=clock or FakeClock())

    detector = YoloDrowsinessDetector(
        person_detector=person_detector,
        eye_locator=eye_locator,
        eye_classifier=eye_classifier,
        drowsiness_service=drowsiness_service,
        eye_state_smoother=eye_state_smoother if eye_state_smoother is not None else make_pass_through_smoother(),
        alarm_service=alarm_service,
        draw_boxes=draw_boxes,
        single_eye_confidence_threshold=single_eye_confidence_threshold,
    )
    return detector, person_detector


class FakeClock:
    def __init__(self, initial_time: float = 0.0) -> None:
        self._time = initial_time

    def __call__(self) -> float:
        return self._time

    def advance(self, seconds: float) -> None:
        self._time += seconds


def test_call_returns_awake_with_open_eyes():
    detector, _ = make_detector(eye_responses=((OPEN_EYES, 0.9), (OPEN_EYES, 0.85)))

    result = detector(make_frame())

    assert result["state"] == DetectionState.AWAKE
    assert result["eyes_closed"] is False
    assert result["eyes_examined"] == 2


def test_call_returns_no_face_when_no_person_detected():
    detector, person_detector = make_detector(person_box=None)

    result = detector(make_frame())

    assert result["state"] == DetectionState.NO_FACE
    assert result["eyes_examined"] == 0
    assert person_detector.call_count == 1


def test_face_detected_tolerates_short_eye_locator_miss():
    """Kişi kutusu bulunduğu halde göz bölgesinin bulunamadığı KISA
    kaçırmalar (Haar'ın tek karede gözü kaçırması) tolerans penceresi
    içinde NO_FACE üretmemeli — aksi halde EyeStateSmoother'ın kendi
    missing_frame_tolerance'ı by-pass edilir ve birikmiş kapalı-göz
    süresi tek bir kötü kare yüzünden sıfırlanır."""
    detector, _ = make_detector(eye_detections=())

    result = detector(make_frame())

    assert result["state"] != DetectionState.NO_FACE
    assert result["state"] == DetectionState.AWAKE
    assert result["face_detected"] is True
    assert result["person_detected"] is True
    assert result["eye_region_found"] is False
    assert result["eyes_examined"] == 0
    assert result["eyes_closed"] is False


def test_persistent_measurement_miss_becomes_unverifiable():
    """GÜVENLİK DAVRANIŞI: sürücü yüzünü kameradan çevirdiğinde gövdesi
    görünmeye devam eder, yani YOLO hâlâ bir kişi kutusu bulur. Göz
    ölçümü KALICI olarak alınamıyorsa sistem "uyanık" demeye devam
    etmemeli. Artık NO_FACE değil UNVERIFIABLE üretiliyor: kadrajda kimse
    olmaması ile kişinin ölçülememesi farklı iki durum."""
    clock = FakeClock()
    detector, _ = make_detector(eye_detections=(), clock=clock)

    # Tolerans penceresi (varsayılan 0.35 sn) içinde önceki karar korunur.
    assert detector(make_frame())["state"] == DetectionState.AWAKE
    clock.advance(0.2)
    assert detector(make_frame())["state"] == DetectionState.AWAKE

    clock.advance(0.2)
    result = detector(make_frame())

    assert result["state"] == DetectionState.UNVERIFIABLE
    assert result["person_detected"] is True
    assert result["eye_region_found"] is False
    assert result["measurement_valid"] is False


def test_eye_box_found_but_all_classifications_fail_is_not_awake():
    """REGRESYON: göz KUTUSUNUN bulunmuş olması, göz durumunun ölçüldüğü
    anlamına gelmez. Kutu bulunup içindeki crop'ların hepsi geçersiz
    çıkarsa (çok küçük kutu, sınıflandırıcı hatası) eski kod kaçırma
    sayacını sıfırlıyor ve sistem AWAKE üretiyordu — gözü hiç göremediği
    halde "uyanık" diyordu. Karar artık kutunun değil, geçerli bir
    SINIFLANDIRMANIN varlığına bağlı."""
    clock = FakeClock()
    detector, _ = make_detector(
        eye_detections=TWO_EYE_DETECTIONS,
        eye_responses=(("GECERSIZ_ETIKET", 0.9),),
        clock=clock,
    )

    first = detector(make_frame())
    assert first["eye_region_found"] is True
    assert first["eyes_examined"] == 0
    assert first["measurement_valid"] is False

    clock.advance(1.0)
    assert detector(make_frame())["state"] == DetectionState.UNVERIFIABLE


def test_persistent_low_confidence_single_eye_becomes_unverifiable():
    """Düşük güvenli tek göz okuması da "ölçüm alınamadı" sayılır —
    smoother'a oy veremeyen bir kare, uyku kararına da temel olamaz."""
    clock = FakeClock()
    detector, _ = make_detector(
        eye_detections=[(30, 20, 20, 20)],
        eye_responses=((CLOSED_EYES, 0.40),),
        single_eye_confidence_threshold=0.80,
        clock=clock,
    )

    assert detector(make_frame())["measurement_valid"] is False

    clock.advance(1.0)
    assert detector(make_frame())["state"] == DetectionState.UNVERIFIABLE


def test_measurement_timer_resets_after_successful_frame():
    """Ölçümsüz geçen süre sayacı kalıcı olmamalı: geçerli bir ölçüm
    geldiğinde sıfırlanır, yoksa uzun bir oturumda birikmiş tekil
    kaçırmalar bir noktadan sonra sistemi sürekli UNVERIFIABLE'a
    düşürürdü."""
    clock = FakeClock()
    detector, _ = make_detector(eye_detections=(), clock=clock)
    normal_locator = EyeRegionLocator(cascade=FakeCascade(detections=TWO_EYE_DETECTIONS))
    empty_locator = EyeRegionLocator(cascade=FakeCascade(detections=()))

    # Toleransın hemen altına kadar ölçümsüz git.
    detector(make_frame())
    clock.advance(0.3)
    assert detector(make_frame())["state"] == DetectionState.AWAKE

    # Geçerli bir ölçüm geldi -> sayaç sıfırlanmalı.
    detector._eye_locator = normal_locator
    clock.advance(0.05)
    recovered = detector(make_frame())
    assert recovered["state"] == DetectionState.AWAKE
    assert recovered["measurement_valid"] is True

    # Sayaç sıfırlandıysa, aynı süre tekrar baştan tolere edilir.
    detector._eye_locator = empty_locator
    clock.advance(0.3)
    assert detector(make_frame())["state"] == DetectionState.AWAKE


def test_measurement_miss_does_not_cancel_active_closed_streak():
    """Birikmiş bir "kapalı göz" serisi sürerken gelen kısa ölçüm
    boşluğu alarmı iptal etmemeli — aksi halde tam alarm eşiğine
    yaklaşırken tek bir kötü kare süreyi sıfırlar."""
    clock = FakeClock()
    detector, _ = make_detector(
        eye_responses=((CLOSED_EYES, 0.95), (CLOSED_EYES, 0.95)),
        drowsiness_seconds=1.0,
        clock=clock,
    )

    assert detector(make_frame())["eyes_closed"] is True

    # Ölçüm kesildi ama seri korunuyor, süre işlemeye devam ediyor.
    detector._eye_locator = EyeRegionLocator(cascade=FakeCascade(detections=()))
    clock.advance(0.3)
    held = detector(make_frame())
    assert held["eyes_closed"] is True
    assert held["closed_duration"] == pytest.approx(0.3)

    clock.advance(0.8)
    assert detector(make_frame())["state"] == DetectionState.UNVERIFIABLE


def test_call_treats_single_detected_closed_eye_as_closed():
    detector, _ = make_detector(eye_detections=[(30, 20, 20, 20)], eye_responses=((CLOSED_EYES, 0.9),))

    result = detector(make_frame())

    assert result["eyes_closed"] is True
    assert result["eyes_examined"] == 1


def test_call_requires_both_eyes_closed():
    """Tek göz açık görünüyorsa (diğeri kapalı olsa bile) yanlış pozitif
    DROWSY'yi önlemek için eyes_closed False olmalı — bkz.
    YoloDrowsinessDetector'ın sınıf docstring'indeki tasarım notu."""
    detector, _ = make_detector(eye_responses=((CLOSED_EYES, 0.9), (OPEN_EYES, 0.8)))

    result = detector(make_frame())

    assert result["eyes_closed"] is False


def test_call_result_satisfies_detect_fn_contract():
    detector, _ = make_detector()

    result = detector(make_frame())

    assert REQUIRED_DETECT_FN_KEYS <= result.keys()
    assert "eyes_examined" in result
    assert "alarm_triggered" in result


def test_call_transitions_to_drowsy_after_threshold():
    clock = FakeClock()
    detector, _ = make_detector(
        eye_responses=((CLOSED_EYES, 0.9), (CLOSED_EYES, 0.9)),
        drowsiness_seconds=2.0,
        clock=clock,
    )

    first = detector(make_frame())
    assert first["state"] == DetectionState.AWAKE
    assert first["closed_duration"] == 0.0

    clock.advance(2.5)
    second = detector(make_frame())

    assert second["state"] == DetectionState.DROWSY
    assert second["is_drowsy"] is True


def test_call_triggers_alarm_when_drowsy():
    clock = FakeClock()
    alarm_service = AlarmService(cooldown_seconds=5.0, clock=clock)
    detector, _ = make_detector(
        eye_responses=((CLOSED_EYES, 0.9), (CLOSED_EYES, 0.9)),
        drowsiness_seconds=1.0,
        clock=clock,
        alarm_service=alarm_service,
    )

    detector(make_frame())
    clock.advance(1.5)
    result = detector(make_frame())

    assert result["state"] == DetectionState.DROWSY
    assert result["alarm_triggered"] is True
    assert result["alarm_trigger_count"] == 1


def test_invalid_eye_crop_is_skipped_not_fatal():
    """EyeRegionLocator'ın çok küçük/geçersiz bir kutu döndürmesi (yanlış
    pozitif tespit) tüm frame'i MODEL_ERROR'a düşürmemeli — o göz atlanır,
    geri kalan gözlerle devam edilir."""
    # (0, 0, 2, 2): EyeClassifierService'in varsayılan min_crop_size=8
    # kontrolünü ihlal edecek kadar küçük bir "göz" tespiti.
    detector, _ = make_detector(
        eye_detections=[(0, 0, 2, 2), (100, 20, 20, 20)],
        eye_responses=((OPEN_EYES, 0.9),),
    )

    result = detector(make_frame())

    assert result["eyes_examined"] == 1
    assert result["state"] in (DetectionState.AWAKE, DetectionState.DROWSY, DetectionState.NO_FACE)


def test_draw_boxes_false_does_not_mutate_frame():
    detector, _ = make_detector(draw_boxes=False)
    frame = make_frame()
    original = frame.copy()

    detector(frame)

    assert np.array_equal(frame, original)


def test_draw_boxes_true_mutates_frame_when_person_detected():
    detector, _ = make_detector(draw_boxes=True)
    frame = make_frame()
    original = frame.copy()

    detector(frame)

    assert not np.array_equal(frame, original)


def test_close_calls_eye_locator_close_if_present():
    """GERÇEK KAYNAK SIZINTISI RİSKİNİN REGRESYON TESTİ: eye_locator
    (örn. MediaPipeEyeLocator) gerçek bir kaynak tutuyorsa (bkz.
    mediapipe_eye_locator.py), YoloDrowsinessDetector.close() bunu
    KAPATMALI. EyeRegionLocator'ın (Haar, kaynaksız) close() metodu
    yoktur — bu yüzden ayrıca sahte bir eye_locator enjekte ediyoruz."""

    class FakeEyeLocatorWithClose:
        def __init__(self):
            self.closed = False

        def locate(self, frame, *, search_region=None):
            return ()

        def close(self):
            self.closed = True

    fake_eye_locator = FakeEyeLocatorWithClose()
    detector = YoloDrowsinessDetector(
        person_detector=FakePersonDetector(box=None),
        eye_locator=fake_eye_locator,
        eye_classifier=EyeClassifierService(model=FakeEyeModel(responses=((OPEN_EYES, 0.9),))),
        drowsiness_service=DrowsinessService(drowsiness_seconds=2.0, clock=FakeClock()),
        eye_state_smoother=make_pass_through_smoother(),
    )

    detector.close()

    assert fake_eye_locator.closed is True


def test_close_does_not_crash_when_eye_locator_has_no_close():
    """EyeRegionLocator (Haar) close() metoduna sahip DEĞİL — close()'un
    bunu getattr ile güvenle atlaması, patlamaması gerekiyor (varsayılan
    make_detector() zaten gerçek EyeRegionLocator kullanıyor)."""
    detector, _ = make_detector()

    detector.close()  # patlamamalı


def test_close_resets_drowsiness_and_alarm_service():
    clock = FakeClock()
    alarm_service = AlarmService(cooldown_seconds=1.0, clock=clock)
    detector, _ = make_detector(
        eye_responses=((CLOSED_EYES, 0.9), (CLOSED_EYES, 0.9)),
        drowsiness_seconds=1.0,
        clock=clock,
        alarm_service=alarm_service,
    )

    detector(make_frame())
    clock.advance(1.5)
    detector(make_frame())
    assert alarm_service.is_active is True

    detector.close()

    assert alarm_service.is_active is False
    assert alarm_service.trigger_count == 0


def test_result_includes_raw_and_smoothed_fields():
    """Geriye dönük uyumluluk: eyes_closed YUMUŞATILMIŞ kararı temsil
    eder; raw_eyes_closed/smoothed_eyes_closed/raw_confidence ayrı ayrı
    da raporlanır (React tarafında debug ve smoke test'te hangi katmanın
    hata yaptığını ayırt edebilmek için)."""
    detector, _ = make_detector(eye_responses=((CLOSED_EYES, 0.9), (CLOSED_EYES, 0.9)))

    result = detector(make_frame())

    assert result["raw_eyes_closed"] is True
    assert result["smoothed_eyes_closed"] is True
    assert result["eyes_closed"] == result["smoothed_eyes_closed"]
    assert result["raw_confidence"] == pytest.approx(0.9)


def test_result_includes_per_step_timing_fields():
    """FPS düşüklüğü araştırmasının bir parçası: mentor'un istediği
    'yolo=.. eye=..' dökümünü ayrı bir profiling scripti yazmadan doğrudan
    result dict'inde (dolayısıyla /api/detection/status'ta) görebilmek
    için. Değerler gerçek zamana bağlı olduğundan tam sayı doğrulanamaz —
    sadece varlığı ve makul (negatif olmayan) olduğu kontrol edilir."""
    detector, _ = make_detector(eye_responses=((CLOSED_EYES, 0.9), (CLOSED_EYES, 0.9)))

    result = detector(make_frame())

    assert "person_detect_ms" in result
    assert "eye_pipeline_ms" in result
    assert result["person_detect_ms"] >= 0.0
    assert result["eye_pipeline_ms"] >= 0.0


def test_single_eye_below_confidence_threshold_treated_as_missing():
    """Tek göz bulunduysa ve confidence single_eye_confidence_threshold'un
    (varsayılan 0.80) altındaysa, bu kare smoother için GEÇERSİZ sayılır
    — ham karara hiç katılmaz (eyes_closed=True DEĞİL, False ve
    'geçersiz')."""
    # Gerçek (pass-through OLMAYAN) bir smoother ile: geçersiz kareler
    # missing_tolerance_seconds=0 aşılınca reset+False döner; burada asıl
    # test ettiğimiz şey raw_eyes_closed'ın False kalması ve smoothed'ın
    # da (tek düşük-confidence kare yüzünden) True OLMAMASI.
    detector, _ = make_detector(
        eye_detections=[(30, 20, 20, 20)],
        eye_responses=((CLOSED_EYES, 0.5),),  # 0.5 < 0.80 eşiği
        single_eye_confidence_threshold=0.80,
    )

    result = detector(make_frame())

    assert result["raw_eyes_closed"] is False
    assert result["smoothed_eyes_closed"] is False
    assert result["eyes_examined"] == 1  # ham tespit sayısı hâlâ doğru raporlanır


def test_single_eye_above_confidence_threshold_is_used():
    detector, _ = make_detector(
        eye_detections=[(30, 20, 20, 20)],
        eye_responses=((CLOSED_EYES, 0.85),),  # 0.85 >= 0.80 eşiği
        single_eye_confidence_threshold=0.80,
    )

    result = detector(make_frame())

    assert result["raw_eyes_closed"] is True
    assert result["smoothed_eyes_closed"] is True


def test_close_resets_eye_state_smoother():
    smoother = EyeStateSmoother(window_seconds=0.9, minimum_valid_seconds=0.0, clock=FakeClock())
    detector, _ = make_detector(eye_responses=((CLOSED_EYES, 0.9), (CLOSED_EYES, 0.9)), eye_state_smoother=smoother)

    detector(make_frame())
    assert smoother.valid_vote_count > 0

    detector.close()

    assert smoother.valid_vote_count == 0


def test_custom_eye_state_smoother_can_be_injected():
    """Constructor'a hiç smoother verilmezse varsayılan (gerçek) bir
    EyeStateSmoother otomatik oluşturulmalı."""
    detector = YoloDrowsinessDetector(
        person_detector=FakePersonDetector(box=None),
        eye_locator=EyeRegionLocator(cascade=FakeCascade()),
        eye_classifier=EyeClassifierService(model=FakeEyeModel(responses=((OPEN_EYES, 0.9),))),
        drowsiness_service=DrowsinessService(),
    )

    result = detector(make_frame())  # patlamamalı, varsayılan smoother sorunsuz çalışmalı

    assert "smoothed_eyes_closed" in result


def test_smoother_tolerates_single_missing_frame_end_to_end():
    """Kullanıcının GERÇEK webcam logundaki bug'ın YoloDrowsinessDetector
    seviyesinde uçtan uca replikasyonu: gözler art arda kapalı
    sınıflandırılırken TEK bir karede Haar hiç göz bulamıyor (eye_detections=()),
    sonra tekrar iki göz kapalı geliyor. Gerçek (pass-through OLMAYAN)
    varsayılan smoother + varsayılan closed_grace_period=0 olan bir
    DrowsinessService ile bile, DROWSY'ye ulaşmalı — çünkü artık hem
    face_detected hem smoothed_eyes_closed tek bir eksik kareden
    etkilenmiyor."""
    clock = FakeClock()
    smoother = EyeStateSmoother(
        window_seconds=0.9,
        closed_ratio_threshold=0.6,
        missing_tolerance_seconds=0.25,
        minimum_valid_seconds=0.35,
        clock=clock,
    )
    drowsiness_service = DrowsinessService(drowsiness_seconds=2.0, clock=clock)

    person_detector = FakePersonDetector(box=PERSON_BOX)
    # İlk 3 kare: 2 göz kapalı (cascade normal çalışıyor).
    eye_locator_normal = EyeRegionLocator(cascade=FakeCascade(detections=TWO_EYE_DETECTIONS))
    eye_classifier = EyeClassifierService(model=FakeEyeModel(responses=((CLOSED_EYES, 0.95), (CLOSED_EYES, 0.95))))

    detector = YoloDrowsinessDetector(
        person_detector=person_detector,
        eye_locator=eye_locator_normal,
        eye_classifier=eye_classifier,
        drowsiness_service=drowsiness_service,
        eye_state_smoother=smoother,
    )

    for _ in range(3):
        result = detector(make_frame())
        clock.advance(0.3)

    assert result["state"] == DetectionState.AWAKE
    assert result["smoothed_eyes_closed"] is True

    # TEK karelik glitch: Haar bu karede hiç göz bulamıyor (eye_locator'ı
    # geçici olarak boş-tespit döndürene değiştiriyoruz).
    detector._eye_locator = EyeRegionLocator(cascade=FakeCascade(detections=()))
    glitch_result = detector(make_frame())
    clock.advance(0.3)

    # Eski davranışta: face_detected=False -> NO_FACE -> koşulsuz reset.
    # Yeni davranışta: face_detected hâlâ True (YOLO kişiyi görüyor),
    # smoother toleransı devrede -> önceki 'kapalı' kararı korunmalı.
    assert glitch_result["state"] != DetectionState.NO_FACE
    assert glitch_result["smoothed_eyes_closed"] is True

    # Cascade tekrar normal çalışmaya başlıyor, seri devam ediyor. Yeterince
    # (gerçek eşiği rahatça aşacak kadar) kare besleyip DROWSY'ye ulaşana
    # kadar devam ediyoruz — tam iterasyon sayısını kırılgan şekilde
    # hesaplamak yerine bir üst sınırla döngü kuruyoruz.
    detector._eye_locator = eye_locator_normal
    result = None
    for _ in range(20):
        result = detector(make_frame())
        clock.advance(0.3)
        if result["state"] == DetectionState.DROWSY:
            break

    assert result["state"] == DetectionState.DROWSY
    assert result["is_drowsy"] is True


def test_rejects_invalid_max_eyes_per_frame():
    with pytest.raises(ValueError):
        YoloDrowsinessDetector(
            person_detector=FakePersonDetector(box=None),
            eye_locator=EyeRegionLocator(cascade=FakeCascade()),
            eye_classifier=EyeClassifierService(model=FakeEyeModel(responses=((OPEN_EYES, 0.9),))),
            drowsiness_service=DrowsinessService(),
            max_eyes_per_frame=0,
        )


def test_yolo_drowsiness_detector_integrates_with_detection_engine():
    """DrowsinessDetector'ın kendi entegrasyon testiyle aynı desen: gerçek
    bir DetectionEngine'e detect_fn olarak veriyoruz, MODEL_ERROR'a
    düşmeden çalışıp SharedState'e gerçek bir sonuç yazdığını
    doğruluyoruz."""

    class FakeVideoSource:
        def __init__(self):
            self.source = "fake"

        def open(self):
            pass

        def read(self):
            return make_frame()

        def release(self):
            pass

    detector, _ = make_detector(eye_responses=((OPEN_EYES, 0.9), (OPEN_EYES, 0.9)))

    shared_state = SharedState()
    engine = DetectionEngine(
        video_source=FakeVideoSource(),
        shared_state=shared_state,
        detect_fn=detector,
        retry_delay=0.02,
    )

    engine.start()
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if shared_state.get_result()["state"] == DetectionState.AWAKE:
                break
            time.sleep(0.01)
        else:
            pytest.fail("Engine 2 saniye içinde AWAKE durumuna ulaşamadı")

        result = shared_state.get_result()
        assert result["state"] == DetectionState.AWAKE
        assert result["error"] is None
    finally:
        engine.stop()


# --- reset() ile close() ayrımı ---------------------------------------------


class FakeEyeLocatorWithClose:
    def __init__(self):
        self.closed = False

    def locate(self, frame, *, search_region=None):
        return ()

    def close(self):
        self.closed = True


def make_detector_with_closable_locator():
    locator = FakeEyeLocatorWithClose()
    detector = YoloDrowsinessDetector(
        person_detector=FakePersonDetector(box=PERSON_BOX),
        eye_locator=locator,
        eye_classifier=EyeClassifierService(model=FakeEyeModel(responses=((OPEN_EYES, 0.9),))),
        drowsiness_service=DrowsinessService(drowsiness_seconds=2.0, clock=FakeClock()),
        eye_state_smoother=make_pass_through_smoother(),
        alarm_service=AlarmService(cooldown_seconds=1.0, clock=FakeClock()),
    )
    return detector, locator


def test_reset_does_not_close_the_eye_locator():
    """REGRESYON: offline değerlendirme klipler arasında dedektörü
    sıfırlarken close() çağırıyordu; bu MediaPipe FaceMesh'i kapattığı
    için ikinci klipte HER KARE hata veriyordu. reset() kaynağa
    dokunmamalı."""
    detector, locator = make_detector_with_closable_locator()

    detector.reset()

    assert locator.closed is False


def test_detector_is_still_usable_after_reset():
    detector, _ = make_detector_with_closable_locator()

    detector(make_frame())
    detector.reset()
    result = detector(make_frame())

    assert result["state"] in {DetectionState.AWAKE, DetectionState.UNVERIFIABLE}


def test_reset_clears_accumulated_state():
    """Klipler arası sızıntı olmamalı: önceki klibin kapalı-göz serisi
    sonrakinin süresine eklenmemeli."""
    clock = FakeClock()
    detector, _ = make_detector(
        eye_responses=((CLOSED_EYES, 0.95), (CLOSED_EYES, 0.95)),
        drowsiness_seconds=2.0,
        clock=clock,
    )

    detector(make_frame())
    clock.advance(1.5)
    assert detector(make_frame())["closed_duration"] == pytest.approx(1.5)

    detector.reset()

    clock.advance(0.1)
    assert detector(make_frame())["closed_duration"] == pytest.approx(0.0)


def test_reset_is_idempotent():
    detector, locator = make_detector_with_closable_locator()

    detector.reset()
    detector.reset()

    assert locator.closed is False


def test_close_still_releases_the_eye_locator():
    detector, locator = make_detector_with_closable_locator()

    detector.close()

    assert locator.closed is True
