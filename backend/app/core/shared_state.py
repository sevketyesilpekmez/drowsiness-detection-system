from __future__ import annotations

import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any

from app.core.states import DetectionState

# Son kaç alarmın (görüntü dahil) hafızada tutulacağı. Her görüntü
# ~50-150KB JPEG olduğu için 5 tanesi önemsiz bir bellek maliyeti.
ALARM_HISTORY_MAX_LEN = 5


class SharedState:
    """
    Kamera işleme thread'i ile Flask endpoint'leri arasında güvenli veri
    paylaşımı sağlar — yalnızca en son frame ve tespit sonucunu tutar,
    kamerayı açmaz, model çalıştırmaz.

    Not: `update()`'e verilen `result` dict'i her seferinde yeni ve düz
    (nested mutable alan içermeyen) olmalı — bu sayede iç kopyalama için
    ucuz bir shallow copy yeterli oluyor.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_frame: bytes | None = None
        self._frame_id: int = 0
        self._latest_result: dict[str, Any] = self._default_result(DetectionState.STARTING)
        self._error: str | None = None
        self._alarm_snapshot: dict[str, Any] | None = None
        self._alarm_history: deque[dict[str, Any]] = deque(maxlen=ALARM_HISTORY_MAX_LEN)
        self._next_alarm_id = 1

    @staticmethod
    def _default_result(state: str) -> dict[str, Any]:
        return {
            "state": state,
            "is_drowsy": False,
            "eyes_closed": False,
            "confidence": 0.0,
            "closed_duration": 0.0,
            "fps": 0.0,
            "timestamp": SharedState._current_timestamp(),
        }

    def update(
        self,
        *,
        frame: bytes | None,
        result: dict[str, Any],
        error: str | None = None,
    ) -> None:
        """Son frame ve tespit sonucunu atomik şekilde günceller.

        `alarm_triggered=True` gelen karede (alarm o an yeni tetiklendi)
        frame'i ayrıca "alarm anındaki görüntü" olarak saklıyoruz —
        alarmın hangi kareye bakılarak verildiği görülebilsin diye. Yeni
        bir alarm gelince bir öncekinin üzerine yazılır."""
        updated_result = dict(result)
        updated_result["timestamp"] = self._current_timestamp()

        with self._lock:
            self._latest_frame = frame
            self._frame_id += 1
            self._latest_result = updated_result
            self._error = error

            if frame is not None and result.get("alarm_triggered"):
                snapshot = {
                    "id": self._next_alarm_id,
                    "frame": frame,
                    "timestamp": updated_result["timestamp"],
                    "state": result.get("state"),
                    "confidence": result.get("confidence"),
                    "closed_duration": result.get("closed_duration"),
                }
                self._next_alarm_id += 1
                self._alarm_snapshot = snapshot
                # deque(maxlen=...) en eskisini kendiliğinden atar.
                self._alarm_history.append(snapshot)

    def update_result(
        self,
        *,
        result: dict[str, Any],
        error: str | None = None,
    ) -> None:
        """Frame'i değiştirmeden yalnızca tespit sonucunu günceller."""
        updated_result = dict(result)
        updated_result["timestamp"] = self._current_timestamp()

        with self._lock:
            self._latest_result = updated_result
            self._error = error

    def get_frame(self) -> bytes | None:
        """En son JPEG frame verisini döndürür."""
        with self._lock:
            return self._latest_frame

    def get_frame_with_id(self) -> tuple[int, bytes | None]:
        """Frame + frame_id'yi tek kilit altında, tutarlı bir çift olarak
        döndürür — stream endpoint'i "yeni frame geldi mi" sorusunu bu
        sayaçla cevaplar (bytes identity yerine, daha sağlam)."""
        with self._lock:
            return self._frame_id, self._latest_frame

    def get_result(self) -> dict[str, Any]:
        """En son tespit sonucunun bağımsız bir kopyasını döndürür.

        alarm_snapshot_available: frontend'in her poll'da ağır bir JPEG
        indirmeden "gösterecek bir alarm görüntüsü var mı" diye ucuza
        sorabilmesi için — asıl görüntü ayrı bir endpoint'ten çekilir.
        alarm_snapshot_* alanları görüntüyle birlikte bağlam (ne kadar
        süre kapalı kaldığı vb.) taşır; alarm hiç tetiklenmediyse None."""
        with self._lock:
            result = dict(self._latest_result)
            result["error"] = self._error

            if self._alarm_snapshot is not None:
                result["alarm_snapshot_available"] = True
                result["alarm_snapshot_id"] = self._alarm_snapshot.get("id")
                result["alarm_snapshot_closed_duration"] = self._alarm_snapshot.get("closed_duration")
                result["alarm_snapshot_confidence"] = self._alarm_snapshot.get("confidence")
                result["alarm_snapshot_state"] = self._alarm_snapshot.get("state")
                result["alarm_snapshot_timestamp"] = self._alarm_snapshot.get("timestamp")
            else:
                result["alarm_snapshot_available"] = False
                result["alarm_snapshot_id"] = None
                result["alarm_snapshot_closed_duration"] = None
                result["alarm_snapshot_confidence"] = None
                result["alarm_snapshot_state"] = None
                result["alarm_snapshot_timestamp"] = None

        return result

    def get_alarm_snapshot(self) -> dict[str, Any] | None:
        """En son tetiklenen alarmın anındaki frame'ini ve bağlamını
        döndürür. Hiç alarm tetiklenmediyse None."""
        with self._lock:
            if self._alarm_snapshot is None:
                return None
            return dict(self._alarm_snapshot)

    def get_alarm_history(self) -> list[dict[str, Any]]:
        """Son ALARM_HISTORY_MAX_LEN alarmın metadata'sını (frame hariç),
        en yeni önce döndürür. Görüntü ağır olduğu için buraya dahil
        edilmez, ayrı ayrı get_alarm_snapshot_by_id(id) ile çekilir."""
        with self._lock:
            return [
                {key: value for key, value in item.items() if key != "frame"}
                for item in reversed(self._alarm_history)
            ]

    def get_alarm_snapshot_by_id(self, alarm_id: int) -> dict[str, Any] | None:
        """Geçmişteki belirli bir alarmın frame'ini ve bağlamını döndürür.
        O id geçmişten düştüyse None."""
        with self._lock:
            for item in self._alarm_history:
                if item["id"] == alarm_id:
                    return dict(item)
            return None

    def get_snapshot(self) -> dict[str, Any]:
        """Frame + sonuç + hata bilgisini tek kilit altında, tutarlı bir
        üçlü olarak döndürür — get_frame()/get_result()'ı ayrı ayrı
        çağırmak aradaki anda update() çalışırsa tutarsızlığa yol açabilir."""
        with self._lock:
            return {
                "frame": self._latest_frame,
                "frame_id": self._frame_id,
                "result": dict(self._latest_result),
                "error": self._error,
            }

    def set_error(
        self,
        error: str,
        *,
        state: str = DetectionState.SYSTEM_ERROR,
    ) -> None:
        """Kamera veya model hatası oluştuğunda state'i günceller. Önceki
        sayısal alanları (confidence, fps, ...) korumaz — hata durumunda
        eski değerler yerine nötr varsayılanlara döner."""
        with self._lock:
            self._error = error
            self._latest_result = self._default_result(state)

    def clear_error(self) -> None:
        """Mevcut hata bilgisini temizler (state'i değiştirmez)."""
        with self._lock:
            self._error = None

    @staticmethod
    def _current_timestamp() -> str:
        """UTC zaman bilgisini ISO 8601 formatında üretir (görüntüleme
        amaçlı; süre hesapları time.monotonic() kullanır)."""
        return datetime.now(timezone.utc).isoformat()
