from __future__ import annotations

import numpy as np
import pytest

from app.services.eye_classifier_service import (
    CLOSED_EYES,
    OPEN_EYES,
    EyeClassifierService,
)


class FakeModel:
    """EyeClassifierModelLike'a uyan sahte model — gerçek TorchEyeClassifier
    yerine kullanılır, bu yüzden testler torch kurulu olmadan çalışır."""

    def __init__(self, *, label=OPEN_EYES, confidence=0.9):
        self.label = label
        self.confidence = confidence
        self.received_crop = None

    def __call__(self, eye_crop):
        self.received_crop = eye_crop
        return self.label, self.confidence


def make_crop(size=24) -> np.ndarray:
    return np.zeros((size, size, 3), dtype=np.uint8)


def test_classify_returns_open_eyes():
    model = FakeModel(label=OPEN_EYES, confidence=0.87)
    service = EyeClassifierService(model=model)

    result = service.classify(make_crop())

    assert result.label == OPEN_EYES
    assert result.is_closed is False
    assert result.confidence == pytest.approx(0.87)


def test_classify_returns_closed_eyes():
    model = FakeModel(label=CLOSED_EYES, confidence=0.95)
    service = EyeClassifierService(model=model)

    result = service.classify(make_crop())

    assert result.label == CLOSED_EYES
    assert result.is_closed is True


def test_classify_passes_crop_to_model_unchanged():
    model = FakeModel()
    service = EyeClassifierService(model=model)
    crop = make_crop()

    service.classify(crop)

    assert model.received_crop is crop


def test_classify_rejects_invalid_label_from_model():
    model = FakeModel(label="SQUINTING", confidence=0.5)
    service = EyeClassifierService(model=model)

    with pytest.raises(ValueError):
        service.classify(make_crop())


def test_classify_rejects_out_of_range_confidence():
    model = FakeModel(label=OPEN_EYES, confidence=1.5)
    service = EyeClassifierService(model=model)

    with pytest.raises(ValueError):
        service.classify(make_crop())


def test_classify_rejects_non_numeric_confidence():
    model = FakeModel(label=OPEN_EYES, confidence="high")
    service = EyeClassifierService(model=model)

    with pytest.raises(TypeError):
        service.classify(make_crop())


def test_classify_rejects_non_ndarray_crop():
    service = EyeClassifierService(model=FakeModel())

    with pytest.raises(TypeError):
        service.classify("not-an-array")


def test_classify_rejects_too_small_crop():
    service = EyeClassifierService(model=FakeModel(), min_crop_size=16)

    with pytest.raises(ValueError):
        service.classify(make_crop(size=8))


def test_classify_accepts_grayscale_crop():
    model = FakeModel()
    service = EyeClassifierService(model=model)
    grayscale_crop = np.zeros((24, 24), dtype=np.uint8)

    result = service.classify(grayscale_crop)

    assert result.label == OPEN_EYES


def test_rejects_none_model():
    with pytest.raises(TypeError):
        EyeClassifierService(model=None)


def test_rejects_non_callable_model():
    with pytest.raises(TypeError):
        EyeClassifierService(model="not-callable")


def test_rejects_invalid_min_crop_size():
    with pytest.raises(ValueError):
        EyeClassifierService(model=FakeModel(), min_crop_size=0)
