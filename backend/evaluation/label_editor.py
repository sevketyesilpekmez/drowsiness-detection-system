"""
Etiket düzenleme mantığı — GUI'siz, saf.

`scripts/label_clips.py` bunun üzerine ince bir OpenCV kabuğu geçirir.
Ayrımın sebebi: etiketleme hatası ölçüm sonucunu SESSİZCE bozar, yani
bu mantığın test edilebilir olması gerekiyor. GUI içine gömülseydi
yalnızca elle denenebilirdi.

Tasarım kuralı: hiçbir metot exception fırlatmaz, hepsi kullanıcıya
gösterilecek kısa bir mesaj döndürür. Etiketleme sırasında yanlış tuşa
basmak normaldir; program çökmemeli, ne olduğunu söylemeli.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evaluation.ground_truth import parse_clip_labels

DEFAULT_EVENT_LABEL = "eyes_closed"
DISTRACTOR_LABEL = "blink"


@dataclass
class EditableEvent:
    """Henüz JSON'a yazılmamış, üzerinde oynanabilen bir olay."""

    start_s: float
    end_s: float
    expect_alarm: bool = True
    label: str = DEFAULT_EVENT_LABEL

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "start_s": round(self.start_s, 3),
            "end_s": round(self.end_s, 3),
        }
        # expect_alarm=True varsayılan olduğu için yazılmıyor: etiket
        # dosyaları elle de okunuyor, gereksiz alan gürültü yapar.
        if not self.expect_alarm:
            data["expect_alarm"] = False
        if self.label != DEFAULT_EVENT_LABEL:
            data["label"] = self.label
        return data


@dataclass
class LabelEditor:
    """
    Bir klibin etiketlerini adım adım kurar.

    Kullanım akışı (bkz. scripts/label_clips.py):
        begin_event(t)  -> aralık başlatılır ("bekleyen" duruma geçer)
        end_event(t)    -> aralık kapanır ve olay listesine eklenir
        toggle_last_distractor() -> son olay çeldiriciye çevrilir
        undo()          -> son işlem geri alınır
        to_dict()       -> JSON'a yazılabilir sözlük
    """

    clip: str
    fps: float
    duration_s: float
    subject: str | None = None
    conditions: tuple[str, ...] = field(default_factory=tuple)
    notes: str | None = None

    events: list[EditableEvent] = field(default_factory=list)
    pending_start_s: float | None = None

    @property
    def has_pending(self) -> bool:
        return self.pending_start_s is not None

    @property
    def expected_event_count(self) -> int:
        return sum(1 for e in self.events if e.expect_alarm)

    @property
    def distractor_count(self) -> int:
        return sum(1 for e in self.events if not e.expect_alarm)

    def begin_event(self, time_s: float) -> str:
        """
        Aralık başlangıcını işaretler.

        Zaten bekleyen bir başlangıç varsa HATA VERMEZ, başlangıcı
        günceller: etiketleme sırasında "yanlış yerde başlattım, biraz
        geri sarıp tekrar basayım" en sık yapılan düzeltmedir.
        """
        if time_s < 0:
            return "Zaman negatif olamaz."

        if self.has_pending:
            previous = self.pending_start_s
            self.pending_start_s = time_s
            return f"Başlangıç güncellendi: {previous:.2f} -> {time_s:.2f} sn"

        self.pending_start_s = time_s
        return f"Başlangıç: {time_s:.2f} sn (bitiş için 'o')"

    def end_event(self, time_s: float) -> str:
        """Bekleyen aralığı kapatır ve olayı listeye ekler."""
        if not self.has_pending:
            return "Önce 'c' ile bir başlangıç işaretleyin."

        start_s = float(self.pending_start_s)

        if time_s <= start_s:
            return f"Bitiş ({time_s:.2f}) başlangıçtan ({start_s:.2f}) sonra olmalı."

        if time_s > self.duration_s:
            return f"Bitiş klip süresini ({self.duration_s:.2f} sn) aşıyor."

        conflict = self._find_overlap(start_s, time_s)
        if conflict is not None:
            return (
                f"Alarm beklenen bir olayla çakışıyor "
                f"([{conflict.start_s:.2f}, {conflict.end_s:.2f}]). "
                "Çakışan olaylar ölçümü belirsizleştirir."
            )

        event = EditableEvent(start_s=start_s, end_s=time_s)
        self.events.append(event)
        self.pending_start_s = None

        return f"Olay eklendi: {start_s:.2f} - {time_s:.2f} sn ({event.duration_s:.2f} sn)"

    def toggle_last_distractor(self) -> str:
        """
        Son eklenen olayı çeldirici <-> gerçek olay arasında değiştirir.

        Çeldirici (göz kırpma vb.) tespit oranına girmez ama içine düşen
        alarm yanlış alarm olarak sayılır.
        """
        if not self.events:
            return "Henüz olay yok."

        event = self.events[-1]

        if event.expect_alarm:
            # Çeldiriciye çevirirken çakışma kısıtı kalkıyor; ters yönde
            # (çeldiriciden gerçek olaya) çakışma oluşabilir, o yüzden
            # aşağıda tekrar kontrol ediliyor.
            event.expect_alarm = False
            event.label = DISTRACTOR_LABEL
            return f"Çeldirici yapıldı: {event.start_s:.2f} - {event.end_s:.2f} sn"

        conflict = self._find_overlap(event.start_s, event.end_s, ignore=event)
        if conflict is not None:
            return (
                f"Gerçek olaya çevrilemez: [{conflict.start_s:.2f}, {conflict.end_s:.2f}] "
                "ile çakışıyor."
            )

        event.expect_alarm = True
        event.label = DEFAULT_EVENT_LABEL
        return f"Gerçek olay yapıldı: {event.start_s:.2f} - {event.end_s:.2f} sn"

    def undo(self) -> str:
        """
        Son işlemi geri alır.

        Bekleyen bir başlangıç varsa ÖNCE o iptal edilir — kullanıcının
        en son yaptığı iş odur, tamamlanmış bir olayı silmek beklenmedik
        olurdu.
        """
        if self.has_pending:
            cancelled = float(self.pending_start_s)
            self.pending_start_s = None
            return f"Bekleyen başlangıç iptal edildi ({cancelled:.2f} sn)."

        if not self.events:
            return "Geri alınacak bir şey yok."

        event = self.events.pop()
        return f"Olay silindi: {event.start_s:.2f} - {event.end_s:.2f} sn"

    def clear(self) -> str:
        self.events.clear()
        self.pending_start_s = None
        return "Tüm etiketler temizlendi."

    def to_dict(self) -> dict[str, Any]:
        """JSON'a yazılabilir sözlük üretir (olaylar zaman sırasında)."""
        data: dict[str, Any] = {
            "clip": self.clip,
            "fps": round(self.fps, 4),
            "duration_s": round(self.duration_s, 3),
            "events": [e.to_dict() for e in sorted(self.events, key=lambda e: e.start_s)],
        }

        if self.subject:
            data["subject"] = self.subject
        if self.conditions:
            data["conditions"] = list(self.conditions)
        if self.notes:
            data["notes"] = self.notes

        # Anahtar sırası okunabilirlik için sabitleniyor.
        ordered_keys = ["clip", "subject", "conditions", "fps", "duration_s", "notes", "events"]
        return {key: data[key] for key in ordered_keys if key in data}

    def validate(self) -> str | None:
        """
        Üretilen sözlüğü GERÇEK yükleyiciden geçirir.

        Bu, aracın en önemli güvenliği: kaydedilen dosya, değerlendirme
        koşusunun kullanacağı `parse_clip_labels` ile birebir aynı
        doğrulamadan geçer. Böylece "etiketledim ama koşu patlıyor"
        durumu, kaydetme anında yakalanır.

        Returns:
            Hata varsa mesaj, yoksa None.
        """
        try:
            parse_clip_labels(self.to_dict(), source=self.clip)
        except ValueError as exc:
            return str(exc)
        return None

    def status_line(self) -> str:
        """GUI'nin üst satırda göstereceği özet."""
        pending = (
            f"kaydediliyor: {self.pending_start_s:.2f} sn"
            if self.has_pending
            else "hazır"
        )
        return (
            f"{self.clip} | {self.expected_event_count} olay, "
            f"{self.distractor_count} çeldirici | {pending}"
        )

    def warnings(self, *, drowsiness_seconds: float) -> list[str]:
        """
        Kaydetmeden önce gösterilecek uyarılar.

        Hata değil, dikkat çekmesi gereken durumlar — etiketleyici
        kişinin bunları görmesi, sonradan raporda anlamsız sayılar
        görmesinden iyidir.
        """
        messages = []

        short = [e for e in self.events if e.expect_alarm and e.duration_s < drowsiness_seconds]
        if short:
            messages.append(
                f"{len(short)} olay {drowsiness_seconds} sn eşiğinden kısa; "
                "alarm üretmesi beklenmez ve tespit oranına girmez."
            )

        if self.expected_event_count == 0:
            messages.append("Hiç gerçek olay yok — bu klip tespit oranına katkı vermeyecek.")

        if self.distractor_count == 0:
            messages.append(
                "Hiç çeldirici yok. Göz kırpma gibi çeldiriciler olmadan "
                "yanlış alarm oranı anlamlı ölçülemez."
            )

        return messages

    def _find_overlap(
        self, start_s: float, end_s: float, *, ignore: EditableEvent | None = None
    ) -> EditableEvent | None:
        """Verilen aralıkla çakışan, ALARM BEKLENEN bir olay varsa döner.

        Çeldiriciler kısıtın dışında: bir göz kırpmasının uzun bir
        kapanmayla örtüşmesi anlamlı bir etiketlemedir."""
        for event in self.events:
            if event is ignore or not event.expect_alarm:
                continue
            if start_s < event.end_s and end_s > event.start_s:
                return event
        return None
