"""
Offline replay: kare akışını detect_fn'den geçirip zaman çizelgesi üretir.

`DetectionEngine`'in yaptığı işin deterministik ve thread'siz karşılığı.
Kasıtlı farklar:

  - Saat gerçek değil, kare indeksinden türetilir (bkz. clock.py).
  - Thread yok; her şey tek bir döngüde, sırayla.
  - JPEG encode edilmez, SharedState'e yazılmaz — ölçüm için gereksiz.

cv2 IMPORT ETMEZ. Kareleri nereden geldiğini bilmez, yalnızca üzerinde
dönebileceği bir iterable bekler. Bu sayede sahte karelerle (hatta
`None`'larla) test edilebilir; video okuma sorumluluğu çağırana aittir.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from evaluation.clock import FrameClock

# detect_fn patladığında kaydedilen durum — DetectionEngine'in aynı
# durumda ürettiği state ile birebir aynı olsun diye sabit olarak
# tutuluyor (app.core.states import edilmiyor: o modül app paketini,
# o da mediapipe/cv2'yi çeker).
MODEL_ERROR_STATE = "MODEL_ERROR"


@dataclass(frozen=True)
class TimelineEntry:
    """Tek bir işlenen karenin sonucu."""

    frame_index: int
    time_s: float
    state: str
    alarm_triggered: bool
    eyes_closed: bool
    closed_duration: float
    measurement_valid: bool | None
    error: str | None = None


@dataclass(frozen=True)
class ReplayResult:
    entries: tuple[TimelineEntry, ...]
    alarm_times_s: tuple[float, ...]
    states: tuple[str, ...]
    source_frame_count: int
    processed_frame_count: int
    duration_s: float
    frame_step: int
    detect_error_count: int


DetectFn = Callable[[Any], dict[str, Any]]


def replay(
    frames: Iterable[Any],
    detect_fn: DetectFn,
    *,
    source_fps: float,
    frame_step: int = 1,
    clock: FrameClock | None = None,
) -> ReplayResult:
    """
    Kareleri sırayla detect_fn'den geçirir ve zaman çizelgesi döndürür.

    Args:
        frames: Kaynak videonun TÜM kareleri, sırayla. Atlama burada
            yapılır (bkz. frame_step) — çağıranın önceden seyreltmesi
            gerekmez, çünkü doğru zaman damgası için gerçek kare
            indeksleri gerekir.
        detect_fn: `frame -> sonuç dict` fonksiyonu. Gerçek kullanımda
            `YoloDrowsinessDetector`, testlerde sahte bir fonksiyon.
        source_fps: Video dosyasının kare hızı.
        frame_step: Kaç karede bir işlem yapılacağı (bkz.
            `clock.frame_step_for_target_fps`). 1 = her kare.
        clock: detect_fn'in İÇİNDEKİ servislere enjekte edilmiş olan
            saat. Verilmesi ZORUNLU DEĞİL ama gerçek bir dedektörle
            koşarken şarttır: `DrowsinessService` ve `AlarmService`
            süreyi kendi saatlerinden okur, replay ayrı bir saat
            kullanırsa onlarınki hiç ilerlemez ve `closed_duration`
            sonsuza kadar 0 kalır — yani hiçbir zaman alarm üretilmez.
            Verilmezse yalnızca zaman damgaları için içeride bir tane
            oluşturulur (sahte detect_fn'lerle test ederken yeterli).

    detect_fn bir karede patlarsa döngü durmaz: o kare MODEL_ERROR olarak
    kaydedilir ve devam edilir — `DetectionEngine`'in davranışıyla aynı.
    Sessizce yutulmaz, `detect_error_count` ile raporlanır.
    """
    if source_fps <= 0:
        raise ValueError("source_fps sıfırdan büyük olmalıdır.")
    if frame_step < 1:
        raise ValueError("frame_step en az 1 olmalıdır.")

    if clock is None:
        clock = FrameClock(fps=source_fps)
    else:
        # Klipler arası taşımayı önle: saat başa sarılır ve bu klibin
        # gerçek kare hızına ayarlanır.
        clock.reset(fps=source_fps)
    entries: list[TimelineEntry] = []
    alarm_times: list[float] = []
    detect_errors = 0
    source_frame_count = 0

    for frame_index, frame in enumerate(frames):
        source_frame_count = frame_index + 1

        if frame_index % frame_step != 0:
            continue

        clock.seek_to_frame(frame_index)
        time_s = clock()

        try:
            result = dict(detect_fn(frame))
            error = None
        except Exception as exc:  # detect_fn sözleşmesi bozulabilir
            detect_errors += 1
            result = {}
            error = f"{type(exc).__name__}: {exc}"

        state = str(result.get("state", MODEL_ERROR_STATE)) if error is None else MODEL_ERROR_STATE
        alarm_triggered = bool(result.get("alarm_triggered", False))

        if alarm_triggered:
            alarm_times.append(time_s)

        entries.append(
            TimelineEntry(
                frame_index=frame_index,
                time_s=time_s,
                state=state,
                alarm_triggered=alarm_triggered,
                eyes_closed=bool(result.get("eyes_closed", False)),
                closed_duration=float(result.get("closed_duration", 0.0)),
                measurement_valid=_optional_bool(result.get("measurement_valid")),
                error=error,
            )
        )

    return ReplayResult(
        entries=tuple(entries),
        alarm_times_s=tuple(alarm_times),
        states=tuple(e.state for e in entries),
        source_frame_count=source_frame_count,
        processed_frame_count=len(entries),
        # Süre, İŞLENEN son kareye değil KAYNAK kare sayısına dayanır:
        # kare atlansa bile klip o kadar uzundur ve yanlış alarm oranının
        # paydası bundan hesaplanır.
        duration_s=source_frame_count / source_fps,
        frame_step=frame_step,
        detect_error_count=detect_errors,
    )


def timeline_to_rows(entries: Sequence[TimelineEntry]) -> list[dict[str, Any]]:
    """Zaman çizelgesini JSON/CSV'ye yazılabilir düz sözlüklere çevirir."""
    return [
        {
            "frame_index": e.frame_index,
            "time_s": round(e.time_s, 4),
            "state": e.state,
            "alarm_triggered": e.alarm_triggered,
            "eyes_closed": e.eyes_closed,
            "closed_duration": round(e.closed_duration, 4),
            "measurement_valid": e.measurement_valid,
            "error": e.error,
        }
        for e in entries
    ]


def _optional_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)
