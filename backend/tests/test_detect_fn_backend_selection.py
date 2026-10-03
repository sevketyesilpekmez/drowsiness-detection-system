"""
DETECTION_BACKEND / ALLOW_DETECTION_FALLBACK seçim mantığının testleri.

ÖNEMLİ: Bu testler _build_detect_fn'i, conftest.py'nin zorladığı
ENABLE_DETECTION_ENGINE=False'u BYPASS ederek, sahte bir Flask app'in
config'i üzerinden doğrudan çağırır (gerçek create_app() değil). Bu
bilinçli bir tasarım: gerçek kamera/model yükleme testler sırasında asla
tetiklenmemeli (bkz. conftest.py'deki temel invariant), ama backend SEÇME
mantığının (fallback var/yok) kendisi test edilebilir ve edilmeli.

Bu sandbox'ta torch/ultralytics GERÇEKTEN kurulu değil (bkz. README 7.
bölüm) — yani _try_build_yolo_pytorch_detect_fn'in başarısız olması bu
testlerde MOCK değil, GERÇEK bir başarısızlıktır. Bu yüzden aşağıdaki
testler "yolo_pytorch kurulamadığında ne olur" senaryosunu sahte değil
gerçek bir hatayla doğruluyor. Yalnızca "yolo_pytorch GERÇEKTEN
kurulduğunda ne olur" senaryosu (başarı yolu) bu ortamda doğrulanamıyor.
"""

from __future__ import annotations

import time

import pytest
from flask import Flask

from app import _build_detect_fn, _StrictBackendFailure, _try_build_yolo_pytorch_detect_fn
from app.config import Config
from app.core.detection_engine import DetectionEngine
from app.core.shared_state import SharedState
from app.core.states import DetectionState
from app.services.drowsiness_detector import DrowsinessDetector


def make_app(**overrides) -> Flask:
    """Config'ten türeyen ama testin ihtiyacına göre override edilebilen
    sahte bir Flask app. create_app() KULLANILMIYOR ki gerçek route/engine
    kurulumuna hiç girilmesin — yalnızca _build_detect_fn'in ihtiyaç
    duyduğu config anahtarları test ediliyor."""
    app = Flask(__name__)
    app.config.from_object(Config)
    app.config["ENABLE_DETECTION_ENGINE"] = True
    app.config.update(overrides)
    return app


def test_yolo_pytorch_unavailable_in_this_sandbox():
    """Bu testin diğer tüm testlerin varsayımını doğrular: torch/ultralytics
    gerçekten kurulu değil, bu yüzden aşağıdaki fallback testleri sahte
    değil gerçek bir başarısızlık üzerinden çalışıyor."""
    app = make_app(DETECTION_BACKEND="yolo_pytorch")

    detector, error = _try_build_yolo_pytorch_detect_fn(app)

    assert detector is None
    assert error is not None


def test_fallback_enabled_falls_back_to_mediapipe_when_yolo_unavailable():
    app = make_app(DETECTION_BACKEND="yolo_pytorch", ALLOW_DETECTION_FALLBACK=True)

    detector = _build_detect_fn(app)

    assert isinstance(detector, DrowsinessDetector)
    detector.close()


def test_fallback_disabled_produces_strict_failure_not_mediapipe():
    app = make_app(DETECTION_BACKEND="yolo_pytorch", ALLOW_DETECTION_FALLBACK=False)

    detector = _build_detect_fn(app)

    assert isinstance(detector, _StrictBackendFailure)
    assert not isinstance(detector, DrowsinessDetector)


def test_strict_failure_raises_with_informative_message_mentioning_flag():
    app = make_app(DETECTION_BACKEND="yolo_pytorch", ALLOW_DETECTION_FALLBACK=False)
    detector = _build_detect_fn(app)

    with pytest.raises(RuntimeError, match="ALLOW_DETECTION_FALLBACK=false"):
        detector(frame=None)


def test_strict_failure_close_is_noop():
    detector = _StrictBackendFailure("test mesajı")

    detector.close()  # exception atmamalı


def test_build_detect_fn_logs_selected_backend(caplog):
    """docker compose loglarında ilk görünmesi gereken satır — mentor'un
    doğrulama listesindeki "Detection backend: yolo_pytorch" satırı."""
    app = make_app(DETECTION_BACKEND="yolo_pytorch", ALLOW_DETECTION_FALLBACK=True)

    with caplog.at_level("INFO"):
        _build_detect_fn(app)

    assert "Detection backend: yolo_pytorch" in caplog.text


def test_mediapipe_backend_selected_directly_ignores_fallback_flag():
    """DETECTION_BACKEND=mediapipe iken ALLOW_DETECTION_FALLBACK'in hiç
    önemi yok — yolo hiç denenmiyor, doğrudan mediapipe kuruluyor."""
    app = make_app(DETECTION_BACKEND="mediapipe", ALLOW_DETECTION_FALLBACK=False)

    detector = _build_detect_fn(app)

    assert isinstance(detector, DrowsinessDetector)
    detector.close()


def test_strict_failure_detect_fn_produces_persistent_model_error_via_engine():
    """_StrictBackendFailure'ın DetectionEngine ile gerçek entegrasyonu:
    her karede aynı RuntimeError fırlatılır, engine bunu MODEL_ERROR'a
    çevirir ve state kalıcı olarak MODEL_ERROR'da kalır (mediapipe'a asla
    düşülmez) — mentor'un istediği tam olarak bu davranış."""

    class FakeVideoSource:
        def __init__(self):
            self.source = "fake"

        def open(self):
            pass

        def read(self):
            import numpy as np

            return np.zeros((10, 10, 3), dtype="uint8")

        def release(self):
            pass

    detector = _StrictBackendFailure(
        "YOLO/PyTorch backend'i başlatılamadı (ALLOW_DETECTION_FALLBACK=false): test"
    )
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
            if shared_state.get_result()["state"] == DetectionState.MODEL_ERROR:
                break
            time.sleep(0.01)
        else:
            pytest.fail("Engine 2 saniye içinde MODEL_ERROR durumuna ulaşamadı")

        result = shared_state.get_result()
        assert result["state"] == DetectionState.MODEL_ERROR
        assert "ALLOW_DETECTION_FALLBACK=false" in result["error"]

        # Bir tur daha bekle: state hâlâ MODEL_ERROR olmalı, mediapipe'a
        # "kendiliğinden" geçilmemeli — hata kalıcı.
        time.sleep(0.1)
        assert shared_state.get_result()["state"] == DetectionState.MODEL_ERROR
    finally:
        engine.stop()
