from app import create_app
from app.core.detection_engine import DetectionEngine
from app.core.shared_state import SharedState


def test_create_app_wires_shared_state_and_detection_engine():
    app = create_app()

    assert isinstance(app.extensions.get("shared_state"), SharedState)
    assert isinstance(app.extensions.get("detection_engine"), DetectionEngine)


def test_detection_engine_does_not_auto_start_in_test_environment():
    """En kritik regresyon testi: testler ENABLE_DETECTION_ENGINE'i hiç set
    etmez, dolayısıyla create_app() gerçek kamerayı ASLA açmamalı. Bu test
    kırılırsa, tüm test suite'i sandbox/CI'da kamera erişimine bağımlı hale
    gelir (yavaş/flaky testler, hatta geliştiricinin laptop'unda `pytest`
    çalıştırıldığında webcam'in gereksiz yere açılması gibi yan etkiler)."""
    app = create_app()
    engine = app.extensions["detection_engine"]

    assert app.config["ENABLE_DETECTION_ENGINE"] is False
    assert engine.is_running is False


def test_shared_state_is_isolated_per_app_instance():
    """SharedState global bir singleton DEĞİL — her create_app() çağrısı
    kendi izole örneğini almalı, yoksa testler birbirini kirletir."""
    app_a = create_app()
    app_b = create_app()

    assert app_a.extensions["shared_state"] is not app_b.extensions["shared_state"]
