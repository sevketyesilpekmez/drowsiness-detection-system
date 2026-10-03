from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from app.core.detection_engine import REQUIRED_DETECT_FN_KEYS, DetectionEngine
from app.core.shared_state import SharedState
from app.core.states import DetectionState
from app.services.alarm_service import AlarmService
from app.services.drowsiness_detector import DrowsinessDetector
from app.services.drowsiness_service import DrowsinessService
from app.services.landmark_service import LEFT_EYE_INDICES, RIGHT_EYE_INDICES, LandmarkService


class FakeFaceMesh:
    """test_landmark_service.py'deki ile aynı sahte FaceMesh; burada
    DrowsinessDetector'ın gerçek LandmarkService + DrowsinessService
    birleşimini doğru sürdüğünü kanıtlamak için kullanılır."""

    def __init__(self, *, landmarks=None) -> None:
        self.closed = False
        self.output = SimpleNamespace(
            multi_face_landmarks=None if landmarks is None else [SimpleNamespace(landmark=landmarks)]
        )

    def process(self, image):
        return self.output

    def close(self) -> None:
        self.closed = True


class SwitchableFaceMesh:
    """Zamana yayılan bir senaryo (gözler kapanıyor, eşik aşılıyor, tekrar
    açılıyor) simüle etmek için hangi landmark setinin döneceği testin
    ortasında değiştirilebilir."""

    def __init__(self) -> None:
        self.closed = False
        self._landmarks = None

    def set_landmarks(self, landmarks) -> None:
        self._landmarks = landmarks

    def process(self, image):
        if self._landmarks is None:
            return SimpleNamespace(multi_face_landmarks=None)
        return SimpleNamespace(multi_face_landmarks=[SimpleNamespace(landmark=self._landmarks)])

    def close(self) -> None:
        self.closed = True


class FakeVideoSource:
    """DetectionEngine entegrasyon testi için: gerçek kamera/video yerine
    her seferinde aynı sabit frame'i döndürür."""

    def __init__(self) -> None:
        self.opened = False
        self.released = False
        self.source = "fake"

    def open(self) -> None:
        self.opened = True

    def read(self) -> np.ndarray:
        return make_frame()

    def release(self) -> None:
        self.released = True


def make_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def make_landmarks(*, eye_height: float):
    """test_landmark_service.py'deki ile aynı yardımcı: EAR = eye_height/100."""
    landmarks = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(478)]

    def assign_eye(indices):
        pixel_points = (
            (100.0, 100.0),
            (125.0, 100.0 - eye_height / 2),
            (175.0, 100.0 - eye_height / 2),
            (200.0, 100.0),
            (175.0, 100.0 + eye_height / 2),
            (125.0, 100.0 + eye_height / 2),
        )
        for index, (x, y) in zip(indices, pixel_points, strict=True):
            landmarks[index] = SimpleNamespace(x=x / 640.0, y=y / 480.0, z=0.0)

    assign_eye(LEFT_EYE_INDICES)
    assign_eye(RIGHT_EYE_INDICES)
    return landmarks


class FakeClock:
    def __init__(self, initial_time: float = 0.0) -> None:
        self._time = initial_time

    def __call__(self) -> float:
        return self._time

    def advance(self, seconds: float) -> None:
        self._time += seconds


def make_detector(*, face_mesh, drowsiness_seconds=2.0, clock=None):
    landmark_service = LandmarkService(face_mesh=face_mesh)
    drowsiness_service = DrowsinessService(
        drowsiness_seconds=drowsiness_seconds,
        clock=clock or FakeClock(),
    )
    return DrowsinessDetector(landmark_service=landmark_service, drowsiness_service=drowsiness_service)


def test_call_returns_awake_for_open_eyes():
    face_mesh = FakeFaceMesh(landmarks=make_landmarks(eye_height=30.0))
    detector = make_detector(face_mesh=face_mesh)

    result = detector(make_frame())

    assert result["state"] == DetectionState.AWAKE
    assert result["is_drowsy"] is False
    assert result["eyes_closed"] is False
    assert result["ear"] is not None


def test_call_returns_no_face_when_no_landmarks():
    face_mesh = FakeFaceMesh(landmarks=None)
    detector = make_detector(face_mesh=face_mesh)

    result = detector(make_frame())

    assert result["state"] == DetectionState.NO_FACE
    assert result["ear"] is None


def test_call_result_satisfies_detect_fn_contract():
    face_mesh = FakeFaceMesh(landmarks=make_landmarks(eye_height=30.0))
    detector = make_detector(face_mesh=face_mesh)

    result = detector(make_frame())

    assert REQUIRED_DETECT_FN_KEYS <= result.keys()
    assert result.keys() >= REQUIRED_DETECT_FN_KEYS | {"ear", "left_ear", "right_ear"}


def test_call_transitions_to_drowsy_after_threshold():
    """Uçtan uca senaryo: gözler kapanır, eşik dolmadan AWAKE, eşik dolunca
    DROWSY. LandmarkService ve DrowsinessService'in kendi testleri ayrı ayrı
    geçse bile birleşimlerinin doğru olduğunun garantisi değildir — bu
    yüzden ayrı bir test."""
    face_mesh = SwitchableFaceMesh()
    clock = FakeClock()
    detector = make_detector(face_mesh=face_mesh, drowsiness_seconds=2.0, clock=clock)

    face_mesh.set_landmarks(make_landmarks(eye_height=10.0))  # kapalı göz (EAR=0.1)

    first = detector(make_frame())
    assert first["state"] == DetectionState.AWAKE
    assert first["eyes_closed"] is True
    assert first["closed_duration"] == 0.0

    clock.advance(1.0)
    second = detector(make_frame())
    assert second["state"] == DetectionState.AWAKE
    assert second["closed_duration"] == pytest.approx(1.0)

    clock.advance(1.5)  # toplam 2.5s, eşik 2.0s
    third = detector(make_frame())
    assert third["state"] == DetectionState.DROWSY
    assert third["is_drowsy"] is True


def test_call_resets_when_eyes_reopen():
    face_mesh = SwitchableFaceMesh()
    clock = FakeClock()
    detector = make_detector(face_mesh=face_mesh, drowsiness_seconds=2.0, clock=clock)

    face_mesh.set_landmarks(make_landmarks(eye_height=10.0))
    detector(make_frame())
    clock.advance(1.9)

    face_mesh.set_landmarks(make_landmarks(eye_height=30.0))  # gözler açıldı
    reopened = detector(make_frame())
    assert reopened["state"] == DetectionState.AWAKE
    assert reopened["closed_duration"] == 0.0

    face_mesh.set_landmarks(make_landmarks(eye_height=10.0))
    restarted = detector(make_frame())
    assert restarted["closed_duration"] == 0.0  # yeniden başladı, 1.9s miras kalmadı


def test_call_without_alarm_service_still_reports_default_alarm_fields():
    """alarm_service verilmediğinde bile alarm_triggered/alarm_active
    alanları sözlükte bulunmalı (False olarak) — frontend'in bu alanları
    her zaman güvenle okuyabilmesi için, alarm servisinin varlığına göre
    şartlı bir sözleşme olmamalı."""
    face_mesh = FakeFaceMesh(landmarks=make_landmarks(eye_height=30.0))
    detector = make_detector(face_mesh=face_mesh)

    result = detector(make_frame())

    assert result["alarm_triggered"] is False
    assert result["alarm_active"] is False


def test_call_triggers_alarm_when_drowsy_threshold_reached():
    face_mesh = SwitchableFaceMesh()
    clock = FakeClock()
    landmark_service = LandmarkService(face_mesh=face_mesh)
    drowsiness_service = DrowsinessService(drowsiness_seconds=2.0, clock=clock)

    threading_event = threading.Event()
    alarm_service = AlarmService(
        alarm_callback=threading_event.set,
        cooldown_seconds=5.0,
        clock=clock,
    )

    detector = DrowsinessDetector(
        landmark_service=landmark_service,
        drowsiness_service=drowsiness_service,
        alarm_service=alarm_service,
    )

    face_mesh.set_landmarks(make_landmarks(eye_height=10.0))  # kapalı göz

    not_yet = detector(make_frame())
    assert not_yet["alarm_triggered"] is False
    assert not_yet["alarm_active"] is False

    clock.advance(2.5)  # eşik (2.0s) aşıldı
    triggered = detector(make_frame())

    assert triggered["state"] == DetectionState.DROWSY
    assert triggered["alarm_triggered"] is True
    assert triggered["alarm_active"] is True
    assert triggered["alarm_trigger_count"] == 1
    assert threading_event.wait(timeout=1.0)

    # Aynı frame'de tekrar tetiklenmemeli (cooldown), sayaç sabit kalmalı.
    still_drowsy = detector(make_frame())
    assert still_drowsy["alarm_triggered"] is False
    assert still_drowsy["alarm_active"] is True
    assert still_drowsy["alarm_trigger_count"] == 1

    # Cooldown dolup ikinci kez tetiklenince sayaç 2'ye çıkmalı — bu, HTTP
    # polling'in tek karelik alarm_triggered=True'yu kaçırsa bile yeni
    # alarmı fark edebilmesinin (count'un artmış olmasıyla) kanıtı.
    clock.advance(5.0)
    second_trigger = detector(make_frame())
    assert second_trigger["alarm_triggered"] is True
    assert second_trigger["alarm_trigger_count"] == 2


def test_close_resets_alarm_service():
    face_mesh = FakeFaceMesh(landmarks=None)
    landmark_service = LandmarkService(face_mesh=face_mesh)
    alarm_service = AlarmService(alarm_callback=lambda: None, cooldown_seconds=1.0)
    detector = DrowsinessDetector(
        landmark_service=landmark_service,
        drowsiness_service=DrowsinessService(),
        alarm_service=alarm_service,
    )

    alarm_service.update(is_drowsy=True)
    assert alarm_service.is_active is True
    assert alarm_service.trigger_count == 1

    detector.close()

    assert alarm_service.is_active is False
    assert alarm_service.last_triggered_at is None
    assert alarm_service.trigger_count == 0


def test_close_is_idempotent_and_closes_owned_landmark_service():
    face_mesh = FakeFaceMesh(landmarks=None)
    landmark_service = LandmarkService(face_mesh=face_mesh)
    drowsiness_service = DrowsinessService()
    detector = DrowsinessDetector(landmark_service=landmark_service, drowsiness_service=drowsiness_service)

    detector.close()
    detector.close()  # ikinci çağrı patlamamalı

    assert landmark_service.is_closed is True


def test_draw_landmarks_false_does_not_mutate_frame():
    face_mesh = FakeFaceMesh(landmarks=make_landmarks(eye_height=30.0))
    landmark_service = LandmarkService(face_mesh=face_mesh)
    detector = DrowsinessDetector(
        landmark_service=landmark_service,
        drowsiness_service=DrowsinessService(),
        draw_landmarks=False,
    )

    frame = make_frame()
    original = frame.copy()
    detector(frame)

    assert np.array_equal(frame, original)


def test_draw_landmarks_true_mutates_frame_when_face_detected():
    face_mesh = FakeFaceMesh(landmarks=make_landmarks(eye_height=30.0))
    landmark_service = LandmarkService(face_mesh=face_mesh)
    detector = DrowsinessDetector(
        landmark_service=landmark_service,
        drowsiness_service=DrowsinessService(),
        draw_landmarks=True,
    )

    frame = make_frame()
    original = frame.copy()
    detector(frame)

    assert not np.array_equal(frame, original)


def test_drowsiness_detector_integrates_with_detection_engine():
    """En üst seviye entegrasyon: DrowsinessDetector'ı gerçek bir
    DetectionEngine'e detect_fn olarak veriyoruz (dummy_detect_fn'in Faz
    2'de nasıl değiştirileceğinin provası) — MODEL_ERROR'a düşmeden çalışıp
    SharedState'e gerçek AWAKE sonucu yazdığını doğruluyoruz. Ayrıca
    engine.stop()'un TEK BAŞINA (ayrı bir detector.close() çağrısına
    gerek kalmadan) LandmarkService'in MediaPipe kaynağını serbest
    bıraktığını da kanıtlıyoruz — bkz. DetectionEngine.stop()'taki
    detect_fn.close() zinciri ve test_detection_engine.py'deki
    test_stop_closes_detect_fn_if_closeable."""
    face_mesh = FakeFaceMesh(landmarks=make_landmarks(eye_height=30.0))
    landmark_service = LandmarkService(face_mesh=face_mesh)
    detector = DrowsinessDetector(landmark_service=landmark_service, drowsiness_service=DrowsinessService())

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
        assert "ear" in result
    finally:
        engine.stop()

    assert landmark_service.is_closed is True


def test_reset_does_not_close_the_landmark_service():
    """REGRESYON: reset() ile close() ayrı olmalı. close() MediaPipe
    FaceMesh'i kapatıyor; klipler arası sıfırlama için kullanılırsa
    dedektör bir daha çalışmaz (bkz. scripts/evaluate_system.py)."""
    face_mesh = FakeFaceMesh(landmarks=None)
    landmark_service = LandmarkService(face_mesh=face_mesh)
    detector = DrowsinessDetector(
        landmark_service=landmark_service,
        drowsiness_service=DrowsinessService(),
    )

    detector.reset()

    result = detector(make_frame())
    assert result["state"] == DetectionState.NO_FACE  # hâlâ çalışıyor


def test_close_still_closes_the_landmark_service():
    face_mesh = FakeFaceMesh(landmarks=None)
    landmark_service = LandmarkService(face_mesh=face_mesh)
    detector = DrowsinessDetector(
        landmark_service=landmark_service,
        drowsiness_service=DrowsinessService(),
    )

    detector.close()

    with pytest.raises(RuntimeError):
        detector(make_frame())
