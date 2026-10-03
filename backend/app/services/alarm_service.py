from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

Clock = Callable[[], float]
AlarmCallback = Callable[[], None]


class AlarmService:
    """
    Drowsy durumunda alarm tetikleme ve tekrar çalma aralığını yönetir.

    Uyku tespiti yapmaz, ses çalmaz — yalnızca is_drowsy değerine göre
    cooldown'lu bir tetikleme kararı üretir.

    Alarm sesi backend'de çalınmıyor: `alarm_callback` opsiyonel, web
    dashboard senaryosunda (React tarayıcıda ses çalar) hiç verilmez.
    Callback yalnızca backend'in kendi bir şey yapması gereken senaryolar
    için (loglama, harici röle) — verilirse ayrı bir daemon thread'de
    çalışır ki yavaş bir işlem detection döngüsünü bloklamasın.

    Thread-safety: update() tek background thread'den çağrılır, ama
    is_active/last_triggered_at/trigger_count Flask request handler'larından
    da okunabileceği için lock ile korunuyor.

    cooldown_seconds neden > 0 olmalı: 0 olsaydı is_drowsy=True sürdükçe
    her frame'de trigger_count artar, gereksiz thread'ler spawn edilirdi.

    trigger_count neden var: `alarm_triggered` yalnızca tetiklendiği tek
    update() çağrısında True — düşük frekanslı bir HTTP polling istemcisi
    bu tek kareyi kaçırabilir. trigger_count monoton arttığı için istemci
    "son gördüğümden büyük mü" diye karşılaştırıp hiçbir tetiklenmeyi
    kaçırmaz.
    """

    def __init__(
        self,
        *,
        alarm_callback: AlarmCallback | None = None,
        cooldown_seconds: float = 5.0,
        clock: Clock = time.monotonic,
    ) -> None:
        if alarm_callback is not None and not callable(alarm_callback):
            raise TypeError("alarm_callback None veya çağrılabilir bir fonksiyon olmalıdır.")

        if isinstance(cooldown_seconds, bool) or not isinstance(cooldown_seconds, (int, float)):
            raise TypeError("cooldown_seconds sayısal bir değer olmalıdır.")

        if cooldown_seconds <= 0:
            raise ValueError("cooldown_seconds sıfırdan büyük olmalıdır.")

        if not callable(clock):
            raise TypeError("clock çağrılabilir bir fonksiyon olmalıdır.")

        self._alarm_callback = alarm_callback
        self._cooldown_seconds = float(cooldown_seconds)
        self._clock = clock

        self._last_triggered_at: float | None = None
        self._active = False
        self._trigger_count = 0
        self._lock = threading.Lock()

    @property
    def cooldown_seconds(self) -> float:
        return self._cooldown_seconds

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._active

    @property
    def last_triggered_at(self) -> float | None:
        with self._lock:
            return self._last_triggered_at

    @property
    def trigger_count(self) -> int:
        """Uygulama başladığından (veya son reset()'ten) beri kaç kez
        tetiklendiğini gösteren monoton sayaç (bkz. sınıf docstring'i)."""
        with self._lock:
            return self._trigger_count

    def update(self, *, is_drowsy: bool) -> bool:
        """
        Güncel drowsy durumuna göre alarmı gerekirse tetikler.

        Returns:
            Alarm bu çağrıda tetiklendiyse True, aksi halde False.
        """
        if not isinstance(is_drowsy, bool):
            raise TypeError("is_drowsy bool olmalıdır.")

        if not is_drowsy:
            with self._lock:
                self._active = False

            return False

        now = self._clock()

        with self._lock:
            if not self._can_trigger(now):
                return False

            self._active = True
            self._last_triggered_at = now
            self._trigger_count += 1

        if self._alarm_callback is not None:
            thread = threading.Thread(
                target=self._run_callback_safely,
                name="alarm-callback",
                daemon=True,
            )
            thread.start()

        return True

    def reset(self) -> None:
        """Alarmın çalışma durumunu, cooldown geçmişini ve trigger_count'u
        temizler."""
        with self._lock:
            self._active = False
            self._last_triggered_at = None
            self._trigger_count = 0

    def _can_trigger(self, now: float) -> bool:
        if self._last_triggered_at is None:
            return True

        elapsed = now - self._last_triggered_at

        return elapsed >= self._cooldown_seconds

    def _run_callback_safely(self) -> None:
        try:
            self._alarm_callback()
        except Exception:
            logger.exception("Alarm callback çalıştırılırken hata oluştu.")
