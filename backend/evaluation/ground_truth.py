"""
Değerlendirme klipleri için etiket şeması: yükleme ve doğrulama.

Etiket dosyaları JSON'dur ve klip başına bir tanedir:

    {
      "clip": "glasses_01.mp4",
      "subject": "S03",
      "conditions": ["glasses", "daylight"],
      "fps": 30.0,
      "duration_s": 62.0,
      "events": [
        {"start_s": 10.2, "end_s": 14.0},
        {"start_s": 31.5, "end_s": 32.1, "expect_alarm": false, "label": "blink"}
      ]
    }

`expect_alarm` varsayılan true'dur. false verilen olaylar ÇELDİRİCİdir
(göz kırpma, kısa aşağı bakış): tespit oranına girmezler ama içlerine
düşen bir alarm yanlış alarm sayılır. Çeldirici etiketlemek zorunlu
değildir ama şiddetle önerilir — onlar olmadan yanlış alarm oranının
hangi davranıştan kaynaklandığı raporlanamaz.

Doğrulama bilerek katıdır: etiket hatası, ölçüm sonucunu sessizce
bozacağı için erken ve gürültülü şekilde yakalanmalıdır.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation.metrics import GroundTruthEvent


@dataclass(frozen=True)
class ClipLabels:
    """Tek bir değerlendirme klibinin etiketleri."""

    clip: str
    events: tuple[GroundTruthEvent, ...]
    fps: float | None = None
    duration_s: float | None = None
    subject: str | None = None
    conditions: tuple[str, ...] = field(default_factory=tuple)
    notes: str | None = None

    @property
    def expected_event_count(self) -> int:
        return sum(1 for e in self.events if e.expect_alarm)


def parse_clip_labels(data: dict[str, Any], *, source: str = "<bellek>") -> ClipLabels:
    """
    Sözlükten `ClipLabels` üretir ve doğrular.

    Raises:
        ValueError: Zorunlu alan eksikse, tip yanlışsa ya da alarm
            beklenen iki olay zaman olarak çakışıyorsa.
    """
    if not isinstance(data, dict):
        raise ValueError(f"{source}: etiket dosyasının kökü bir JSON nesnesi olmalı.")

    clip = data.get("clip")
    if not isinstance(clip, str) or not clip.strip():
        raise ValueError(f"{source}: 'clip' alanı zorunlu ve boş olmayan bir metin olmalı.")

    raw_events = data.get("events", [])
    if not isinstance(raw_events, list):
        raise ValueError(f"{source}: 'events' bir liste olmalı.")

    events = tuple(_parse_event(raw, index=i, source=source) for i, raw in enumerate(raw_events))
    _reject_overlapping_expected_events(events, source=source)

    fps = _optional_positive_number(data.get("fps"), name="fps", source=source)
    duration_s = _optional_positive_number(data.get("duration_s"), name="duration_s", source=source)

    if duration_s is not None:
        for event in events:
            if event.end_s > duration_s:
                raise ValueError(
                    f"{source}: olay klip sonunu aşıyor (end_s={event.end_s}, duration_s={duration_s}). "
                    "Etiketleme hatası ya da yanlış duration_s."
                )

    conditions = data.get("conditions", [])
    if not isinstance(conditions, list) or not all(isinstance(c, str) for c in conditions):
        raise ValueError(f"{source}: 'conditions' metin listesi olmalı.")

    subject = data.get("subject")
    if subject is not None and not isinstance(subject, str):
        raise ValueError(f"{source}: 'subject' metin olmalı.")

    notes = data.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise ValueError(f"{source}: 'notes' metin olmalı.")

    return ClipLabels(
        clip=clip,
        events=events,
        fps=fps,
        duration_s=duration_s,
        subject=subject,
        conditions=tuple(conditions),
        notes=notes,
    )


def load_clip_labels(path: str | Path) -> ClipLabels:
    """Bir JSON etiket dosyasını okur ve doğrular."""
    label_path = Path(path)

    try:
        data = json.loads(label_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Etiket dosyası bulunamadı: {label_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label_path}: geçersiz JSON — {exc}") from exc

    return parse_clip_labels(data, source=str(label_path))


def load_label_directory(directory: str | Path) -> tuple[ClipLabels, ...]:
    """
    Bir klasördeki tüm .json etiket dosyalarını (isim sırasıyla) yükler.

    Sıralama deterministiktir — rapor çıktısının çalıştırmadan
    çalıştırmaya aynı olması için.
    """
    label_dir = Path(directory)

    if not label_dir.is_dir():
        raise ValueError(f"Etiket klasörü bulunamadı: {label_dir}")

    paths = sorted(label_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"{label_dir}: hiç .json etiket dosyası yok.")

    return tuple(load_clip_labels(p) for p in paths)


def _parse_event(raw: Any, *, index: int, source: str) -> GroundTruthEvent:
    where = f"{source}: events[{index}]"

    if not isinstance(raw, dict):
        raise ValueError(f"{where} bir nesne olmalı.")

    for key in ("start_s", "end_s"):
        if key not in raw:
            raise ValueError(f"{where}: '{key}' zorunlu.")
        if isinstance(raw[key], bool) or not isinstance(raw[key], (int, float)):
            raise ValueError(f"{where}: '{key}' sayısal olmalı.")

    expect_alarm = raw.get("expect_alarm", True)
    if not isinstance(expect_alarm, bool):
        raise ValueError(f"{where}: 'expect_alarm' bool olmalı.")

    label = raw.get("label", "eyes_closed")
    if not isinstance(label, str):
        raise ValueError(f"{where}: 'label' metin olmalı.")

    try:
        return GroundTruthEvent(
            start_s=float(raw["start_s"]),
            end_s=float(raw["end_s"]),
            expect_alarm=expect_alarm,
            label=label,
        )
    except ValueError as exc:
        raise ValueError(f"{where}: {exc}") from exc


def _reject_overlapping_expected_events(
    events: tuple[GroundTruthEvent, ...], *, source: str
) -> None:
    """
    Alarm beklenen iki olay çakışamaz.

    Çakışırlarsa bir alarmın hangi olaya ait olduğu belirsizleşir ve
    tespit oranı etiketleme sırasına göre değişir — ölçümün
    tekrarlanabilirliğini bozan bir durum. Çeldiriciler (expect_alarm
    false) bu kısıtın dışında: bir göz kırpmasının uzun bir kapanmayla
    zaman olarak örtüşmesi anlamlı bir etiketleme olabilir.
    """
    expected = sorted(
        (e for e in events if e.expect_alarm), key=lambda e: e.start_s
    )

    for previous, current in zip(expected, expected[1:]):
        if current.start_s < previous.end_s:
            raise ValueError(
                f"{source}: alarm beklenen olaylar çakışıyor "
                f"([{previous.start_s}, {previous.end_s}] ve [{current.start_s}, {current.end_s}]). "
                "Hangi alarmın hangi olaya ait olduğu belirsizleşir."
            )


def _optional_positive_number(value: Any, *, name: str, source: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{source}: '{name}' sayısal olmalı.")
    if value <= 0:
        raise ValueError(f"{source}: '{name}' sıfırdan büyük olmalı.")
    return float(value)
