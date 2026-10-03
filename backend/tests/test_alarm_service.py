from __future__ import annotations

import threading

import pytest

from app.services.alarm_service import AlarmService


class FakeClock:
    def __init__(self, initial_time: float = 0.0) -> None:
        self._time = initial_time

    def __call__(self) -> float:
        return self._time

    def advance(self, seconds: float) -> None:
        self._time += seconds


def test_rejects_invalid_arguments() -> None:
    with pytest.raises(TypeError):
        AlarmService(alarm_callback="not-callable")

    with pytest.raises(ValueError):
        AlarmService(alarm_callback=lambda: None, cooldown_seconds=-1)


def test_alarm_callback_is_optional() -> None:
    """Tarayıcı tabanlı alarm mimarisinde backend ses ÇALMAZ — AlarmService
    callback'siz kurulabilmeli, tetiklendiğinde sadece is_active/
    trigger_count günceller, hiçbir thread spawn etmez ya da hata vermez."""
    service = AlarmService(cooldown_seconds=1.0)

    triggered = service.update(is_drowsy=True)

    assert triggered is True
    assert service.is_active is True
    assert service.trigger_count == 1


def test_rejects_zero_cooldown() -> None:
    """cooldown_seconds=0 kabul edilseydi, is_drowsy=True sürdüğü sürece
    her update() çağrısında (her frame'de) yeni bir thread spawn edilirdi
    — sentetik video kaynaklarında fps 200+ olabildiğini zaten gördük
    (test_drowsiness_detector.py), bu yüzden 0 kesinlikle reddedilmeli."""
    with pytest.raises(ValueError):
        AlarmService(alarm_callback=lambda: None, cooldown_seconds=0)


def test_does_not_trigger_when_not_drowsy() -> None:
    calls: list[str] = []

    service = AlarmService(alarm_callback=lambda: calls.append("alarm"))

    triggered = service.update(is_drowsy=False)

    assert triggered is False
    assert service.is_active is False
    assert calls == []


def test_triggers_alarm_when_drowsy() -> None:
    called = threading.Event()

    service = AlarmService(alarm_callback=called.set)

    triggered = service.update(is_drowsy=True)

    assert triggered is True
    assert called.wait(timeout=1.0)
    assert service.is_active is True


def test_cooldown_prevents_repeated_trigger() -> None:
    clock = FakeClock()
    calls: list[str] = []

    service = AlarmService(
        alarm_callback=lambda: calls.append("alarm"),
        cooldown_seconds=5.0,
        clock=clock,
    )

    assert service.update(is_drowsy=True) is True
    assert service.update(is_drowsy=True) is False

    clock.advance(4.9)

    assert service.update(is_drowsy=True) is False

    clock.advance(0.1)

    assert service.update(is_drowsy=True) is True


def test_awake_clears_active_but_not_cooldown() -> None:
    clock = FakeClock()

    service = AlarmService(
        alarm_callback=lambda: None,
        cooldown_seconds=5.0,
        clock=clock,
    )

    service.update(is_drowsy=True)

    assert service.is_active is True

    service.update(is_drowsy=False)

    assert service.is_active is False

    clock.advance(2.0)

    assert service.update(is_drowsy=True) is False


def test_reset_clears_cooldown() -> None:
    clock = FakeClock()

    service = AlarmService(
        alarm_callback=lambda: None,
        cooldown_seconds=10.0,
        clock=clock,
    )

    service.update(is_drowsy=True)
    service.reset()

    assert service.last_triggered_at is None
    assert service.is_active is False
    assert service.update(is_drowsy=True) is True


def test_trigger_count_increments_only_on_actual_trigger() -> None:
    """trigger_count, polling istemcilerinin tek karelik alarm_triggered
    sinyalini kaçırmadan yeni alarmı tespit etmesi için var (bkz.
    AlarmService docstring'i). Bu yüzden yalnızca gerçekten tetiklenen
    çağrılarda artmalı — is_drowsy=False'ta veya cooldown içindeki
    tekrarlarda ARTMAMALI."""
    clock = FakeClock()
    service = AlarmService(cooldown_seconds=5.0, clock=clock)

    assert service.trigger_count == 0

    service.update(is_drowsy=True)
    assert service.trigger_count == 1

    service.update(is_drowsy=True)  # cooldown içinde, artmamalı
    assert service.trigger_count == 1

    service.update(is_drowsy=False)  # uyanık, artmamalı
    assert service.trigger_count == 1

    clock.advance(5.0)
    service.update(is_drowsy=True)  # cooldown doldu, yeni tetiklenme
    assert service.trigger_count == 2


def test_reset_clears_trigger_count() -> None:
    service = AlarmService(cooldown_seconds=1.0)

    service.update(is_drowsy=True)
    assert service.trigger_count == 1

    service.reset()

    assert service.trigger_count == 0


def test_rejects_non_bool_is_drowsy() -> None:
    service = AlarmService(alarm_callback=lambda: None)

    with pytest.raises(TypeError):
        service.update(is_drowsy=1)


def test_callback_exception_does_not_crash_and_is_logged(caplog) -> None:
    """alarm_callback kullanıcı tarafından verilir (örn. ses çalma) ve
    içinde hata olabilir. _run_callback_safely bu hatayı yutup loglamalı,
    background thread'i (dolayısıyla DetectionEngine'i) çökertmemeli."""
    failed = threading.Event()

    def _raise() -> None:
        failed.set()
        raise RuntimeError("boom")

    service = AlarmService(alarm_callback=_raise)

    with caplog.at_level("ERROR"):
        service.update(is_drowsy=True)
        assert failed.wait(timeout=1.0)

    # Callback'in thread'i loglamayı bitirmesi için kısa bir bekleme.
    import time as _time

    _time.sleep(0.05)

    assert "Alarm callback çalıştırılırken hata oluştu" in caplog.text
