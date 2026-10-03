from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.services.landmark_service import (
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    LandmarkService,
    Point,
)


class FakeFaceMesh:
    def __init__(self, *, landmarks=None) -> None:
        self.closed = False
        self.received_image = None

        if landmarks is None:
            self.output = SimpleNamespace(multi_face_landmarks=None)
        else:
            self.output = SimpleNamespace(
                multi_face_landmarks=[SimpleNamespace(landmark=landmarks)]
            )

    def process(self, image):
        self.received_image = image
        return self.output

    def close(self) -> None:
        self.closed = True


def make_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def make_landmarks(*, eye_height: float) -> list[SimpleNamespace]:
    """Piksel koordinatları yerine normalize koordinatlar üretir. Her iki
    göz için yatay mesafe 100 piksel, dikey mesafeler eye_height olacak
    şekilde hazırlanır (bkz. EAR = 2*eye_height / (2*100) = eye_height/100)."""
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


def test_calculate_ear() -> None:
    points = (
        Point(0, 0),
        Point(2, -2),
        Point(8, -2),
        Point(10, 0),
        Point(8, 2),
        Point(2, 2),
    )

    ear = LandmarkService.calculate_ear(points)

    assert ear == pytest.approx(0.4)


def test_calculate_ear_requires_six_points() -> None:
    with pytest.raises(ValueError):
        LandmarkService.calculate_ear([Point(0, 0)] * 5)


def test_calculate_ear_rejects_zero_horizontal_distance() -> None:
    points = (
        Point(0, 0),
        Point(1, -1),
        Point(1, -1),
        Point(0, 0),
        Point(1, 1),
        Point(1, 1),
    )

    with pytest.raises(ValueError):
        LandmarkService.calculate_ear(points)


def test_returns_no_face_when_landmarks_are_missing() -> None:
    service = LandmarkService(face_mesh=FakeFaceMesh())

    result = service.analyze(make_frame())

    assert result.face_detected is False
    assert result.eyes_closed is False
    assert result.confidence == 0.0
    assert result.ear is None


def test_open_eye_returns_eyes_open() -> None:
    landmarks = make_landmarks(eye_height=30.0)

    service = LandmarkService(ear_threshold=0.22, face_mesh=FakeFaceMesh(landmarks=landmarks))

    result = service.analyze(make_frame())

    assert result.face_detected is True
    assert result.eyes_closed is False
    assert result.ear == pytest.approx(0.3)
    assert result.left_ear == pytest.approx(0.3)
    assert result.right_ear == pytest.approx(0.3)


def test_closed_eye_returns_eyes_closed() -> None:
    landmarks = make_landmarks(eye_height=10.0)

    service = LandmarkService(ear_threshold=0.22, face_mesh=FakeFaceMesh(landmarks=landmarks))

    result = service.analyze(make_frame())

    assert result.face_detected is True
    assert result.eyes_closed is True
    assert result.ear == pytest.approx(0.1)


def test_threshold_is_strictly_less_than() -> None:
    landmarks = make_landmarks(eye_height=22.0)

    service = LandmarkService(ear_threshold=0.22, face_mesh=FakeFaceMesh(landmarks=landmarks))

    result = service.analyze(make_frame())

    assert result.ear == pytest.approx(0.22)
    assert result.eyes_closed is False


def test_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError):
        LandmarkService(ear_threshold=0.0, face_mesh=FakeFaceMesh())

    with pytest.raises(ValueError):
        LandmarkService(ear_threshold=1.0, face_mesh=FakeFaceMesh())


def test_rejects_invalid_frame() -> None:
    service = LandmarkService(face_mesh=FakeFaceMesh())

    with pytest.raises(TypeError):
        service.analyze("not-an-array")

    with pytest.raises(ValueError):
        service.analyze(np.zeros((100, 100), dtype=np.uint8))


def test_close_is_idempotent() -> None:
    fake_mesh = FakeFaceMesh()
    service = LandmarkService(face_mesh=fake_mesh)

    service.close()
    service.close()

    assert service.is_closed is True


def test_analyze_after_close_fails() -> None:
    service = LandmarkService(face_mesh=FakeFaceMesh())

    service.close()

    with pytest.raises(RuntimeError):
        service.analyze(make_frame())


def test_close_does_not_close_injected_face_mesh() -> None:
    """Ownership sözleşmesi: face_mesh dışarıdan enjekte edildiyse (test
    senaryosu gibi), LandmarkService onun sahibi değildir ve close()
    çağrıldığında injected nesneyi KAPATMAMALIDIR — çağıran taraf paylaşılan
    bir FaceMesh'i kendisi yönetiyor olabilir."""
    fake_mesh = FakeFaceMesh()
    service = LandmarkService(face_mesh=fake_mesh)

    service.close()

    assert service.is_closed is True
    assert fake_mesh.closed is False
