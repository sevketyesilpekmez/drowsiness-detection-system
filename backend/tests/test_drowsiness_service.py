from __future__ import annotations

import pytest

from app.core.detection_engine import REQUIRED_DETECT_FN_KEYS
from app.core.states import DetectionState
from app.services.drowsiness_service import DrowsinessService


class FakeClock:
    def __init__(self, initial_time: float = 0.0) -> None:
        self._time = initial_time

    def __call__(self) -> float:
        return self._time

    def advance(self, seconds: float) -> None:
        self._time += seconds


def test_rejects_invalid_drowsiness_seconds() -> None:
    with pytest.raises(ValueError):
        DrowsinessService(drowsiness_seconds=0)

    with pytest.raises(ValueError):
        DrowsinessService(drowsiness_seconds=-1)


def test_returns_no_face_and_resets_timer() -> None:
    clock = FakeClock()
    service = DrowsinessService(drowsiness_seconds=2.0, clock=clock)

    service.update(face_detected=True, eyes_closed=True, confidence=0.9)
    clock.advance(1.0)

    result = service.update(face_detected=False, eyes_closed=False, confidence=0.0)

    assert result["state"] == DetectionState.NO_FACE
    assert result["is_drowsy"] is False
    assert result["eyes_closed"] is False
    assert result["closed_duration"] == 0.0
    assert service.eyes_closed_since is None


def test_open_eyes_return_awake_and_reset_timer() -> None:
    clock = FakeClock()
    service = DrowsinessService(drowsiness_seconds=2.0, clock=clock)

    service.update(face_detected=True, eyes_closed=True, confidence=0.8)
    clock.advance(1.5)

    result = service.update(face_detected=True, eyes_closed=False, confidence=0.95)

    assert result["state"] == DetectionState.AWAKE
    assert result["is_drowsy"] is False
    assert result["eyes_closed"] is False
    assert result["closed_duration"] == 0.0
    assert service.eyes_closed_since is None


def test_short_eye_closure_remains_awake() -> None:
    clock = FakeClock()
    service = DrowsinessService(drowsiness_seconds=2.0, clock=clock)

    first_result = service.update(face_detected=True, eyes_closed=True, confidence=0.9)

    assert first_result["state"] == DetectionState.AWAKE
    assert first_result["eyes_closed"] is True
    assert first_result["closed_duration"] == 0.0

    clock.advance(1.25)

    second_result = service.update(face_detected=True, eyes_closed=True, confidence=0.9)

    assert second_result["state"] == DetectionState.AWAKE
    assert second_result["is_drowsy"] is False
    assert second_result["closed_duration"] == 1.25


def test_long_eye_closure_returns_drowsy() -> None:
    clock = FakeClock()
    service = DrowsinessService(drowsiness_seconds=2.0, clock=clock)

    service.update(face_detected=True, eyes_closed=True, confidence=0.88)
    clock.advance(2.0)

    result = service.update(face_detected=True, eyes_closed=True, confidence=0.88)

    assert result["state"] == DetectionState.DROWSY
    assert result["is_drowsy"] is True
    assert result["eyes_closed"] is True
    assert result["closed_duration"] == 2.0


def test_threshold_is_inclusive() -> None:
    clock = FakeClock()
    service = DrowsinessService(drowsiness_seconds=1.5, clock=clock)

    service.update(face_detected=True, eyes_closed=True, confidence=1.0)
    clock.advance(1.5)

    result = service.update(face_detected=True, eyes_closed=True, confidence=1.0)

    assert result["state"] == DetectionState.DROWSY


def test_timer_restarts_after_eyes_reopen() -> None:
    clock = FakeClock()
    service = DrowsinessService(drowsiness_seconds=2.0, clock=clock)

    service.update(face_detected=True, eyes_closed=True, confidence=0.9)
    clock.advance(1.5)
    service.update(face_detected=True, eyes_closed=False, confidence=0.9)
    clock.advance(10.0)

    restarted = service.update(face_detected=True, eyes_closed=True, confidence=0.9)

    assert restarted["state"] == DetectionState.AWAKE
    assert restarted["closed_duration"] == 0.0


@pytest.mark.parametrize(
    ("confidence", "exception_type"),
    [
        (-0.1, ValueError),
        (1.1, ValueError),
        ("0.5", TypeError),
        (True, TypeError),
    ],
)
def test_rejects_invalid_confidence(confidence, exception_type) -> None:
    service = DrowsinessService()

    with pytest.raises(exception_type):
        service.update(face_detected=True, eyes_closed=False, confidence=confidence)


def test_rejects_non_boolean_detection_values() -> None:
    service = DrowsinessService()

    with pytest.raises(TypeError):
        service.update(face_detected=1, eyes_closed=False, confidence=0.5)

    with pytest.raises(TypeError):
        service.update(face_detected=True, eyes_closed=0, confidence=0.5)


# --- Ek testler: çok adımlı senaryolar ve DetectionEngine kontratıyla uyum ---


def test_closed_duration_accumulates_correctly_across_multiple_calls() -> None:
    """Gerçek kullanımda DetectionEngine bu servisi HER frame'de çağırır
    (saniyede onlarca kez), sadece iki uç nokta değil. Süre her ara adımda
    doğru birikmeli, sıfırlanmamalı."""
    clock = FakeClock()
    service = DrowsinessService(drowsiness_seconds=2.0, clock=clock)

    expected_durations = [0.0, 0.3, 0.6, 0.9, 1.2]

    for expected in expected_durations:
        result = service.update(face_detected=True, eyes_closed=True, confidence=0.9)
        assert result["closed_duration"] == pytest.approx(expected)
        assert result["state"] == DetectionState.AWAKE
        assert result["is_drowsy"] is False
        clock.advance(0.3)

    # Eşiği geçince DROWSY'ye dönmeli
    clock.advance(0.5)  # toplam 1.5 + 0.5 = 2.0
    final = service.update(face_detected=True, eyes_closed=True, confidence=0.9)
    assert final["state"] == DetectionState.DROWSY
    assert final["is_drowsy"] is True


def test_timer_restarts_after_no_face_then_face_regained() -> None:
    """NO_FACE -> yüz tekrar bulunup gözler kapalı: süre sıfırdan başlamalı,
    NO_FACE öncesi biriken süre 'miras kalmamalı'."""
    clock = FakeClock()
    service = DrowsinessService(drowsiness_seconds=2.0, clock=clock)

    service.update(face_detected=True, eyes_closed=True, confidence=0.9)
    clock.advance(1.8)  # eşiğe çok yakın ama henüz DROWSY değil

    no_face_result = service.update(face_detected=False, eyes_closed=False, confidence=0.0)
    assert no_face_result["state"] == DetectionState.NO_FACE

    clock.advance(5.0)  # yüz kayıpken zaman geçiyor, önemli olmamalı

    regained = service.update(face_detected=True, eyes_closed=True, confidence=0.9)

    assert regained["state"] == DetectionState.AWAKE
    assert regained["closed_duration"] == 0.0


def test_no_face_ignores_contradictory_eyes_closed_flag() -> None:
    """face_detected=False iken eyes_closed=True gibi çelişkili bir girdi
    verilse bile (gerçek LandmarkService bunu üretmemeli ama savunma amaçlı)
    NO_FACE her zaman öncelikli olmalı ve eyes_closed=False raporlanmalı."""
    service = DrowsinessService()

    result = service.update(face_detected=False, eyes_closed=True, confidence=0.5)

    assert result["state"] == DetectionState.NO_FACE
    assert result["eyes_closed"] is False


# --- closed_grace_period testleri (gerçek bir webcam testinde bulunan
# reset bug'ının düzeltmesi — bkz. DrowsinessService.__init__ docstring'i) ---


def test_rejects_negative_closed_grace_period() -> None:
    with pytest.raises(ValueError):
        DrowsinessService(closed_grace_period=-0.1)


def test_default_grace_period_preserves_immediate_reset_behavior() -> None:
    """closed_grace_period belirtilmezse (varsayılan 0.0) davranış AYNEN
    eskisi gibi kalmalı: tek bir 'açık' okuması anında sıfırlar. Bu test,
    yeni parametrenin var olan hiçbir çağrıyı (mediapipe backend'i dahil)
    bozmadığını kanıtlar."""
    clock = FakeClock()
    service = DrowsinessService(drowsiness_seconds=2.0, clock=clock)

    service.update(face_detected=True, eyes_closed=True, confidence=0.9)
    clock.advance(1.9)  # DROWSY eşiğine çok yakın

    glitch = service.update(face_detected=True, eyes_closed=False, confidence=0.9)
    assert glitch["closed_duration"] == 0.0
    assert glitch["eyes_closed"] is False

    clock.advance(0.01)
    resumed = service.update(face_detected=True, eyes_closed=True, confidence=0.9)
    assert resumed["closed_duration"] == 0.0  # sıfırdan başladı, 1.9s miras kalmadı


def test_grace_period_tolerates_single_frame_glitch_near_threshold() -> None:
    """Kullanıcının gerçek live_camera_smoke_test.py çıktısında görülen
    TAM senaryo: closed_duration 1.91s'e kadar tırmanıyor (DROWSY eşiği
    2.0s'e çok yakın), sonra TEK bir karede eyes_closed=False geliyor
    (Haar cascade'in geçici bir kaçırması), sonra hemen tekrar True
    oluyor. closed_grace_period=0.3 ile bu artık birikimi SIFIRLAMAMALI
    ve seri devam ederek DROWSY'ye ulaşmalı."""
    clock = FakeClock()
    service = DrowsinessService(drowsiness_seconds=2.0, closed_grace_period=0.3, clock=clock)

    # Gözler kapanmaya başlıyor, süre tırmanıyor (gerçek logdaki gibi ~0.21s adımlarla)
    for _ in range(9):
        result = service.update(face_detected=True, eyes_closed=True, confidence=1.0)
        clock.advance(0.21)

    assert result["closed_duration"] == pytest.approx(1.68, abs=0.05)
    assert result["state"] == DetectionState.AWAKE  # henüz eşiğe ulaşmadı

    # TEK karelik glitch: Haar cascade bu karede gözü bulamadı/yanlış sınıflandırdı
    glitch = service.update(face_detected=True, eyes_closed=False, confidence=0.5)

    # Eski davranışta burada closed_duration 0.0'a düşerdi (gerçek bug buydu).
    # Yeni davranışta, glitch grace_period (0.3s) içinde olduğu için seri
    # kesilmemiş SAYILMALI.
    assert glitch["eyes_closed"] is True
    assert glitch["closed_duration"] > 1.5  # biriken süre KORUNDU, sıfırlanmadı

    clock.advance(0.1)  # glitch süresi: 0.1s < 0.3s grace_period, hâlâ tolerans içinde

    resumed = service.update(face_detected=True, eyes_closed=True, confidence=1.0)
    assert resumed["closed_duration"] > 1.5

    # Kısa bir süre daha kapalı kalınca eşiğe ulaşıp DROWSY olmalı.
    clock.advance(0.3)
    final = service.update(face_detected=True, eyes_closed=True, confidence=1.0)
    assert final["state"] == DetectionState.DROWSY
    assert final["is_drowsy"] is True


def test_grace_period_exceeded_still_resets() -> None:
    """Glitch süresi grace_period'u AŞARSA (gözler gerçekten açıldıysa),
    yine normal şekilde sıfırlanmalı — tolerans sınırsız değil."""
    clock = FakeClock()
    service = DrowsinessService(drowsiness_seconds=2.0, closed_grace_period=0.3, clock=clock)

    service.update(face_detected=True, eyes_closed=True, confidence=0.9)
    clock.advance(1.0)
    service.update(face_detected=True, eyes_closed=False, confidence=0.9)

    clock.advance(0.5)  # 0.5s > 0.3s grace_period -> tolerans aşıldı
    still_open = service.update(face_detected=True, eyes_closed=False, confidence=0.9)

    assert still_open["closed_duration"] == 0.0
    assert still_open["eyes_closed"] is False

    clock.advance(0.01)
    resumed = service.update(face_detected=True, eyes_closed=True, confidence=0.9)
    assert resumed["closed_duration"] == 0.0  # gerçekten sıfırdan başladı


def test_grace_period_multiple_short_glitches_within_window() -> None:
    """Grace penceresi içinde birden fazla ardışık 'açık' karesi gelirse
    (tek bir uzun glitch yerine birkaç kısa), _open_glitch_since ilk
    glitch anından itibaren ölçülmeli — her karede yeniden başlamamalı.
    (Not: kasıtlı olarak tam sınırda değil, sınırın belirgin şekilde
    altında/üstünde iki değer kullanılıyor — kayan nokta yuvarlama
    hatalarıyla flaky bir teste dönüşmesin diye.)"""
    clock = FakeClock()
    service = DrowsinessService(drowsiness_seconds=5.0, closed_grace_period=0.3, clock=clock)

    service.update(face_detected=True, eyes_closed=True, confidence=0.9)
    clock.advance(1.0)

    # İki kısa glitch karesi, toplam 0.2s (< 0.3s grace_period) — tolere edilmeli.
    service.update(face_detected=True, eyes_closed=False, confidence=0.9)
    clock.advance(0.1)
    still_tolerated = service.update(face_detected=True, eyes_closed=False, confidence=0.9)
    assert still_tolerated["closed_duration"] > 0.0  # sıfırlanmadı
    assert still_tolerated["eyes_closed"] is True

    clock.advance(0.3)  # toplam glitch süresi artık 0.4s (> 0.3s) -> açıkça aşıldı

    result = service.update(face_detected=True, eyes_closed=False, confidence=0.9)
    assert result["closed_duration"] == 0.0
    assert result["eyes_closed"] is False


def test_brief_no_face_within_grace_period_is_tolerated() -> None:
    """GERÇEK KULLANIMDA bulunan ikinci bir bug'ın düzeltmesi: yolo_pytorch
    backend'inde face_detected SADECE o karede YOLO'nun kişiyi bulup
    bulamadığına bağlı. Alarm çalarken (gözler hâlâ kapalıyken) YOLO'nun
    TEK bir karede kişiyi kaçırması eskiden koşulsuz sıfırlıyordu — bu da
    kullanıcının şikayet ettiği gibi, gözler hiç açılmadan alarmın aniden
    kesilmesine yol açıyordu. Bu artık eyes_closed=False glitch'iyle AYNI
    tolerans penceresini (closed_grace_period) paylaşıyor: aktif bir kapalı
    seri sürerken kısa süreli face_detected=False biriken süreyi SİLMEMELİ.
    (Not: bu testin eski hali tam tersini iddia ediyordu — o zamanki
    varsayım gerçek kullanımda yanlış çıktı, bkz. DrowsinessService.__init__
    docstring'indeki "GÜNCELLEME" notu.)"""
    clock = FakeClock()
    service = DrowsinessService(drowsiness_seconds=2.0, closed_grace_period=0.3, clock=clock)

    service.update(face_detected=True, eyes_closed=True, confidence=0.9)
    clock.advance(1.5)

    glitch = service.update(face_detected=False, eyes_closed=False, confidence=0.0)
    assert glitch["state"] != DetectionState.NO_FACE
    assert glitch["eyes_closed"] is True
    assert glitch["closed_duration"] > 1.0  # biriken süre KORUNDU

    clock.advance(0.05)  # glitch süresi: 0.05s < 0.3s grace_period
    regained = service.update(face_detected=True, eyes_closed=True, confidence=0.9)
    assert regained["closed_duration"] > 1.0  # sıfırlanmadı, seri devam etti


def test_no_face_exceeding_grace_period_still_resets() -> None:
    """Tolerans sınırsız değil: face_detected=False glitch_period'u AŞARSA
    (kişi gerçekten kadraj dışına çıktıysa/kayboldu), yine normal şekilde
    NO_FACE'e düşmeli."""
    clock = FakeClock()
    service = DrowsinessService(drowsiness_seconds=2.0, closed_grace_period=0.3, clock=clock)

    service.update(face_detected=True, eyes_closed=True, confidence=0.9)
    clock.advance(1.5)
    service.update(face_detected=False, eyes_closed=False, confidence=0.0)

    clock.advance(0.5)  # 0.5s > 0.3s grace_period -> tolerans aşıldı
    still_missing = service.update(face_detected=False, eyes_closed=False, confidence=0.0)

    assert still_missing["state"] == DetectionState.NO_FACE
    assert still_missing["closed_duration"] == 0.0

    clock.advance(0.01)
    regained = service.update(face_detected=True, eyes_closed=True, confidence=0.9)
    assert regained["closed_duration"] == 0.0  # gerçekten sıfırdan başladı


def test_no_face_without_active_streak_still_resets_immediately() -> None:
    """Henüz aktif bir 'kapalı göz' serisi hiç başlamamışken (örn. kişi
    kameraya daha yeni giriyor ya da hiç girmedi) face_detected=False'un
    tolere edilecek bir şeyi yok — bu, "hiç kimse kamerada değil"
    durumunu maskelemez, koşulsuz ve anında NO_FACE'e düşmeli."""
    clock = FakeClock()
    service = DrowsinessService(drowsiness_seconds=2.0, closed_grace_period=0.3, clock=clock)

    result = service.update(face_detected=False, eyes_closed=False, confidence=0.0)

    assert result["state"] == DetectionState.NO_FACE
    assert result["closed_duration"] == 0.0


def test_no_face_grace_period_default_zero_preserves_immediate_reset() -> None:
    """closed_grace_period belirtilmezse (varsayılan 0.0), face_detected=False
    davranışı AYNEN eski/basit haliyle kalmalı: anında sıfırlar. Bu test,
    yeni tolerans mantığının mediapipe backend'i gibi grace_period=0.0
    kullanan hiçbir çağrıyı bozmadığını kanıtlar."""
    clock = FakeClock()
    service = DrowsinessService(drowsiness_seconds=2.0, clock=clock)

    service.update(face_detected=True, eyes_closed=True, confidence=0.9)
    clock.advance(1.5)

    no_face = service.update(face_detected=False, eyes_closed=False, confidence=0.0)

    assert no_face["state"] == DetectionState.NO_FACE
    assert no_face["closed_duration"] == 0.0


def test_result_satisfies_detect_fn_contract() -> None:
    """DrowsinessService.update()'in çıktısı, DetectionEngine'in
    REQUIRED_DETECT_FN_KEYS sözleşmesini karşılamalı — ileride LandmarkService
    ile birleştirilip gerçek bir detect_fn olarak enjekte edildiğinde
    DetectionEngine'in MODEL_ERROR'a düşmemesi için bu şart."""
    service = DrowsinessService()

    result = service.update(face_detected=True, eyes_closed=False, confidence=0.5)

    assert REQUIRED_DETECT_FN_KEYS <= result.keys()


# --- measurement_valid / UNVERIFIABLE ---------------------------------------


def test_measurement_valid_defaults_to_true_preserving_old_behaviour():
    """Geriye dönük uyumluluk: measurement_valid geçmeyen çağıranlar
    (mediapipe/EAR backend'i, mevcut testler) eskisi gibi çalışmalı."""
    service = DrowsinessService(drowsiness_seconds=2.0, clock=FakeClock())

    result = service.update(face_detected=True, eyes_closed=False, confidence=0.9)

    assert result["state"] == DetectionState.AWAKE


def test_unmeasurable_frame_holds_previous_decision_within_grace():
    clock = FakeClock()
    service = DrowsinessService(drowsiness_seconds=2.0, measurement_grace_seconds=0.5, clock=clock)

    service.update(face_detected=True, eyes_closed=False, confidence=0.9)

    clock.advance(0.2)
    result = service.update(
        face_detected=True, eyes_closed=False, confidence=0.0, measurement_valid=False
    )

    assert result["state"] == DetectionState.AWAKE


def test_unmeasurable_frame_becomes_unverifiable_after_grace():
    clock = FakeClock()
    service = DrowsinessService(drowsiness_seconds=2.0, measurement_grace_seconds=0.5, clock=clock)

    service.update(face_detected=True, eyes_closed=False, confidence=0.0, measurement_valid=False)
    clock.advance(0.6)
    result = service.update(
        face_detected=True, eyes_closed=False, confidence=0.0, measurement_valid=False
    )

    assert result["state"] == DetectionState.UNVERIFIABLE
    assert result["is_drowsy"] is False
    assert result["closed_duration"] == 0.0


def test_no_face_takes_precedence_over_unverifiable():
    """Kadrajda kimse yoksa sebep "ölçemiyorum" değil "kimse yok" —
    daha spesifik olan etiket kazanır."""
    clock = FakeClock()
    service = DrowsinessService(drowsiness_seconds=2.0, measurement_grace_seconds=0.0, clock=clock)

    result = service.update(
        face_detected=False, eyes_closed=False, confidence=0.0, measurement_valid=False
    )

    assert result["state"] == DetectionState.NO_FACE


def test_closed_streak_survives_short_measurement_gap():
    clock = FakeClock()
    service = DrowsinessService(drowsiness_seconds=1.0, measurement_grace_seconds=0.5, clock=clock)

    service.update(face_detected=True, eyes_closed=True, confidence=0.9)
    clock.advance(0.3)
    held = service.update(
        face_detected=True, eyes_closed=False, confidence=0.0, measurement_valid=False
    )

    assert held["eyes_closed"] is True
    assert held["closed_duration"] == pytest.approx(0.3)

    # Ölçüm geri geldiğinde seri kesintisiz sayılmaya devam eder.
    clock.advance(0.8)
    resumed = service.update(face_detected=True, eyes_closed=True, confidence=0.9)
    assert resumed["state"] == DetectionState.DROWSY
    assert resumed["closed_duration"] == pytest.approx(1.1)


def test_valid_measurement_resets_missing_timer():
    clock = FakeClock()
    service = DrowsinessService(drowsiness_seconds=2.0, measurement_grace_seconds=0.5, clock=clock)

    service.update(face_detected=True, eyes_closed=False, confidence=0.0, measurement_valid=False)
    clock.advance(0.4)
    service.update(face_detected=True, eyes_closed=False, confidence=0.9)

    clock.advance(0.4)
    result = service.update(
        face_detected=True, eyes_closed=False, confidence=0.0, measurement_valid=False
    )

    assert result["state"] == DetectionState.AWAKE


def test_rejects_negative_measurement_grace_seconds():
    with pytest.raises(ValueError):
        DrowsinessService(drowsiness_seconds=2.0, measurement_grace_seconds=-0.1)


def test_rejects_non_bool_measurement_valid():
    service = DrowsinessService(drowsiness_seconds=2.0)

    with pytest.raises(TypeError):
        service.update(
            face_detected=True, eyes_closed=False, confidence=0.9, measurement_valid="evet"
        )
