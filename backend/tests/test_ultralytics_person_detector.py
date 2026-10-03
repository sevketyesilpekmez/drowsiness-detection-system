"""
NOT (dürüstlük): Bu testler gerçek ultralytics/torch KURULU OLMADAN
çalışır — UltralyticsPersonDetector'ın `model=` enjeksiyon parametresi
sayesinde, ultralytics'in kendi ham `YOLO(...)` nesnesinin döndürdüğü
`Results` API'sini (`.boxes.xyxy`, `.boxes.conf`) taklit eden sahte bir
nesne veriyoruz. Böylece sınıfın KENDİ mantığını (confidence eşiği,
en yüksek güvenli kutuyu seçme, clip, ve YENİ EKLENEN imgsz'in gerçekten
model çağrısına geçirilip geçirilmediği) gerçek bir GPU/ağırlık dosyası
olmadan doğrulayabiliyoruz. Modelin kendisinin (YOLOv8n ağırlıklarının)
doğruluğu test EDİLMİYOR — bu zaten mümkün değil (bkz. dosyanın başındaki
"ÖNEMLİ DÜRÜSTLÜK NOTU").
"""

import numpy as np
import pytest

from app.services.ultralytics_person_detector import UltralyticsPersonDetector


class _FakeBoxes:
    def __init__(self, xyxy, conf):
        self._xyxy = xyxy
        self._conf = conf

    def __len__(self):
        return len(self._conf)

    @property
    def conf(self):
        return _FakeTensor(self._conf)

    @property
    def xyxy(self):
        return _FakeXyxy(self._xyxy)


class _FakeTensor:
    """`.tolist()`'i olan minik bir sahte tensor."""

    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)


class _FakeXyxy:
    """`boxes.xyxy[i].tolist()` erişimini taklit eder."""

    def __init__(self, rows):
        self._rows = rows

    def __getitem__(self, index):
        return _FakeTensor(self._rows[index])


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeYoloModel:
    """Ultralytics'in `YOLO(weights)(frame, classes=..., device=...,
    imgsz=..., verbose=...)` çağrılabilir API'sini taklit eder. Kendisine
    geçirilen tüm kwarg'ları `self.calls`'a kaydeder ki testler imgsz'in
    gerçekten iletildiğini doğrulayabilsin."""

    def __init__(self, results=None):
        self._results = results if results is not None else []
        self.calls = []

    def __call__(self, frame, **kwargs):
        self.calls.append(kwargs)
        return self._results


def _frame(width=100, height=100):
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_imgsz_defaults_to_640_and_is_passed_to_model():
    """GERÇEK BUG'IN REGRESYON TESTİ: imgsz eklenmeden önce hiç
    geçirilmiyordu (ultralytics'in kendi varsayılanına — 640 — sessizce
    güveniliyordu). Artık açıkça geçiriliyor, varsayılan yine 640 olduğu
    için eski davranış korunuyor."""
    model = _FakeYoloModel(results=[])
    detector = UltralyticsPersonDetector(model=model)

    detector(_frame())

    assert model.calls[0]["imgsz"] == 640


def test_custom_imgsz_is_passed_to_model():
    model = _FakeYoloModel(results=[])
    detector = UltralyticsPersonDetector(model=model, imgsz=416)

    detector(_frame())

    assert model.calls[0]["imgsz"] == 416


def test_imgsz_zero_or_negative_raises():
    with pytest.raises(ValueError):
        UltralyticsPersonDetector(model=_FakeYoloModel(), imgsz=0)

    with pytest.raises(ValueError):
        UltralyticsPersonDetector(model=_FakeYoloModel(), imgsz=-320)


def test_device_is_passed_to_model():
    model = _FakeYoloModel(results=[])
    detector = UltralyticsPersonDetector(model=model, device="mps")

    detector(_frame())

    assert model.calls[0]["device"] == "mps"


def test_no_results_returns_none():
    detector = UltralyticsPersonDetector(model=_FakeYoloModel(results=[]))
    assert detector(_frame()) is None


def test_empty_boxes_returns_none():
    result = _FakeResult(boxes=None)
    detector = UltralyticsPersonDetector(model=_FakeYoloModel(results=[result]))
    assert detector(_frame()) is None


def test_below_confidence_threshold_returns_none():
    boxes = _FakeBoxes(xyxy=[[10, 10, 50, 50]], conf=[0.3])
    result = _FakeResult(boxes=boxes)
    detector = UltralyticsPersonDetector(
        model=_FakeYoloModel(results=[result]), confidence_threshold=0.5
    )
    assert detector(_frame()) is None


def test_picks_highest_confidence_box_and_clips_to_frame():
    boxes = _FakeBoxes(
        xyxy=[[10, 10, 50, 50], [5, 5, 200, 200]],
        conf=[0.6, 0.9],
    )
    result = _FakeResult(boxes=boxes)
    detector = UltralyticsPersonDetector(model=_FakeYoloModel(results=[result]))

    person_box = detector(_frame(width=100, height=100))

    assert person_box is not None
    assert person_box.confidence == pytest.approx(0.9)
    # (5, 5, 200, 200) 100x100'lük kareye clip edilmeli.
    assert person_box.x1 == 5
    assert person_box.y1 == 5
    assert person_box.x2 == 100
    assert person_box.y2 == 100


def test_invalid_confidence_threshold_raises():
    with pytest.raises(ValueError):
        UltralyticsPersonDetector(model=_FakeYoloModel(), confidence_threshold=1.5)
