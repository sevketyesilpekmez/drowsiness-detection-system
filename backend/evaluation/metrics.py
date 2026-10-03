"""
Uçtan uca değerlendirme metrikleri — saf fonksiyonlar.

Girdi: etiketli olaylar + sistemin ürettiği alarm zaman damgaları.
Çıktı: tespit oranı, alarm gecikmesi, yanlış alarm sıklığı, kör kare oranı.

Video okumaz, model çalıştırmaz, zaman ölçmez. Bu ayrım kasıtlı: karar
kurallarının doğruluğu, bir kamera ya da checkpoint olmadan sınanabilmeli.

Zaman birimi her yerde SANİYE ve klip başlangıcına göredir.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

# Alarmın olay bittikten sonra da kabul edileceği süre. Gözler açıldığı
# anda alarmın kesilmesi beklenir ama işleme gecikmesi (kare kuyruğu,
# smoother penceresi) küçük bir taşmayı normal kılar.
DEFAULT_MATCH_TOLERANCE_SECONDS = 0.5

# "Kör" sayılan durumlar: sistemin sürücü hakkında hiçbir ölçüm
# üretemediği kareler. Yüksek bir kör oran, %100 tespit oranını bile
# anlamsızlaştırır — bu yüzden ayrı raporlanır.
BLIND_STATES = frozenset({"NO_FACE", "UNVERIFIABLE", "CAMERA_ERROR", "MODEL_ERROR", "SYSTEM_ERROR"})


@dataclass(frozen=True)
class GroundTruthEvent:
    """
    Etiketlenmiş tek bir olay.

    Args:
        start_s: Olayın başladığı an (gözlerin kapandığı an).
        end_s: Olayın bittiği an (gözlerin açıldığı an).
        expect_alarm: Bu olayın alarm ÜRETMESİ bekleniyor mu. Göz kırpma,
            kısa aşağı bakış gibi çeldiriciler için False verilir —
            bunlar tespit oranına dahil edilmez, ama içlerine düşen bir
            alarm yine de yanlış alarm sayılır.
        label: Serbest metin (ör. "eyes_closed", "blink") — raporda
            hangi çeldiricinin yanlış alarm ürettiğini görebilmek için.
    """

    start_s: float
    end_s: float
    expect_alarm: bool = True
    label: str = "eyes_closed"

    def __post_init__(self) -> None:
        if self.end_s <= self.start_s:
            raise ValueError(f"Olay bitişi başlangıcından sonra olmalı: start={self.start_s} end={self.end_s}")
        if self.start_s < 0:
            raise ValueError("Olay başlangıcı negatif olamaz.")

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def is_eligible(self, drowsiness_seconds: float) -> bool:
        """
        Bu olayın alarm üretmesi FİZİKSEL olarak mümkün mü.

        Eşik 2.5 sn iken 1.5 sn süren bir göz kapanmasının alarm
        üretmemesi doğru davranıştır; bunu "kaçırma" saymak sistemi
        haksız yere cezalandırır ve tespit oranını anlamsızlaştırır.
        Bu tür olaylar tespit oranından çıkarılıp ayrıca raporlanır
        (bkz. `ineligible_event_count`) — çoğu zaman etiketleme hatasına
        ya da yanlış seçilmiş bir eşiğe işaret ederler.
        """
        return self.expect_alarm and self.duration_s >= drowsiness_seconds


@dataclass(frozen=True)
class AlarmMatch:
    """Bir alarmın hangi olaya (varsa) atandığı."""

    alarm_time_s: float
    event_index: int | None
    latency_s: float | None

    @property
    def is_false_alarm(self) -> bool:
        return self.event_index is None


@dataclass(frozen=True)
class MatchResult:
    matches: tuple[AlarmMatch, ...]
    hit_event_indices: frozenset[int]
    missed_event_indices: tuple[int, ...]
    ineligible_event_indices: tuple[int, ...]
    duplicate_alarm_count: int


def match_alarms(
    events: Sequence[GroundTruthEvent],
    alarm_times_s: Iterable[float],
    *,
    drowsiness_seconds: float,
    match_tolerance_s: float = DEFAULT_MATCH_TOLERANCE_SECONDS,
) -> MatchResult:
    """
    Alarmları etiketli olaylarla eşleştirir.

    Eşleşme penceresi, olayın kendisi DEĞİL:

        [start_s + drowsiness_seconds, end_s + match_tolerance_s]

    Alt sınırın eşik kadar kaydırılmasının sebebi: sistem, gözler
    kapandığı ANDA alarm vermemeli — `drowsiness_seconds` boyunca kapalı
    kalmasını beklemeli. Bu sınırdan ÖNCE gelen bir alarm erken isabet
    değil, YANLIŞ ALARM'dır; aksi halde "her karede alarm çal" diyen
    bozuk bir sistem %100 tespit oranı alırdı.

    Bir olaya birden fazla alarm düşerse ilki isabet sayılır, kalanlar
    yok sayılır (yanlış alarm olarak da sayılmaz) — alarm cooldown'ı
    aynı olay içinde tekrar tetikleyebilir, bu bir hata değildir.

    Args:
        events: Etiketli olaylar. Zaman sırasında olması gerekmez.
        alarm_times_s: Sistemin alarm ürettiği anlar.
        drowsiness_seconds: Sistemin çalıştığı eşik (config'deki
            DROWSINESS_SECONDS ile aynı olmalı).
        match_tolerance_s: Olay bittikten sonra alarmın hâlâ isabet
            sayılacağı süre.
    """
    if drowsiness_seconds <= 0:
        raise ValueError("drowsiness_seconds sıfırdan büyük olmalıdır.")
    if match_tolerance_s < 0:
        raise ValueError("match_tolerance_s negatif olamaz.")

    eligible_indices = [i for i, e in enumerate(events) if e.is_eligible(drowsiness_seconds)]
    ineligible_indices = tuple(
        i for i, e in enumerate(events) if e.expect_alarm and not e.is_eligible(drowsiness_seconds)
    )

    matches: list[AlarmMatch] = []
    hit_indices: set[int] = set()
    duplicate_count = 0

    # Alarmlar zaman sırasına göre işlenir: bir olaya düşen İLK alarm
    # isabet sayılacağı için sıra sonucu etkiler.
    for alarm_time in sorted(alarm_times_s):
        matched_index = _find_matching_event(
            events,
            eligible_indices,
            alarm_time,
            drowsiness_seconds=drowsiness_seconds,
            match_tolerance_s=match_tolerance_s,
        )

        if matched_index is None:
            matches.append(AlarmMatch(alarm_time_s=alarm_time, event_index=None, latency_s=None))
            continue

        if matched_index in hit_indices:
            # Aynı olayın ikinci alarmı: cooldown tekrarı, hata değil.
            duplicate_count += 1
            continue

        hit_indices.add(matched_index)
        expected_at = events[matched_index].start_s + drowsiness_seconds
        matches.append(
            AlarmMatch(
                alarm_time_s=alarm_time,
                event_index=matched_index,
                latency_s=alarm_time - expected_at,
            )
        )

    missed = tuple(i for i in eligible_indices if i not in hit_indices)

    return MatchResult(
        matches=tuple(matches),
        hit_event_indices=frozenset(hit_indices),
        missed_event_indices=missed,
        ineligible_event_indices=ineligible_indices,
        duplicate_alarm_count=duplicate_count,
    )


def _find_matching_event(
    events: Sequence[GroundTruthEvent],
    eligible_indices: Sequence[int],
    alarm_time: float,
    *,
    drowsiness_seconds: float,
    match_tolerance_s: float,
) -> int | None:
    """Alarmın düştüğü uygun olayı bulur; birden fazlaysa en erken
    başlayanı seçer (olaylar çakışmıyorsa zaten tek olur)."""
    candidates = []

    for index in eligible_indices:
        event = events[index]
        window_start = event.start_s + drowsiness_seconds
        window_end = event.end_s + match_tolerance_s

        if window_start <= alarm_time <= window_end:
            candidates.append(index)

    if not candidates:
        return None

    return min(candidates, key=lambda i: events[i].start_s)


def percentile(values: Sequence[float], fraction: float) -> float:
    """
    Doğrusal interpolasyonlu yüzdelik.

    numpy yerine elle yazıldı: bu modülün tek amacı saf kalmak, ve
    küçük örneklemlerde (10-20 klip) yüzdelik tanımının şeffaf olması
    raporlanabilirlik açısından önemli.
    """
    if not values:
        raise ValueError("Boş diziden yüzdelik hesaplanamaz.")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction 0.0 ile 1.0 arasında olmalıdır.")

    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])

    position = fraction * (len(ordered) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    weight = position - lower_index

    return float(ordered[lower_index] * (1 - weight) + ordered[upper_index] * weight)


def state_observability(states: Sequence[str]) -> dict[str, Any]:
    """
    Kare durumlarından gözlemlenebilirlik metrikleri üretir.

    `blind_frame_ratio` neden önemli: sistem karelerin yarısında sürücüyü
    hiç göremediyse, kalan yarıda %100 tespit yapması güven verici
    değildir. Tespit oranı tek başına bu körlüğü gizler.
    """
    total = len(states)
    if total == 0:
        return {"frame_count": 0, "blind_frame_ratio": 0.0, "state_counts": {}}

    counts: dict[str, int] = {}
    for state in states:
        counts[state] = counts.get(state, 0) + 1

    blind = sum(count for state, count in counts.items() if state in BLIND_STATES)

    return {
        "frame_count": total,
        "blind_frame_ratio": blind / total,
        "state_counts": dict(sorted(counts.items())),
    }


def clip_metrics(
    *,
    events: Sequence[GroundTruthEvent],
    alarm_times_s: Sequence[float],
    clip_duration_s: float,
    drowsiness_seconds: float,
    states: Sequence[str] = (),
    match_tolerance_s: float = DEFAULT_MATCH_TOLERANCE_SECONDS,
) -> dict[str, Any]:
    """
    Tek bir klip için uçtan uca metrikleri hesaplar.

    `false_alarms_per_minute`'ın paydası klibin TAMAMI değil, alarmın
    meşru olmadığı süredir (toplam süre eksi alarm beklenen pencereler).
    Aksi halde uzun bir uyku olayı içeren klip, sırf o süre boyunca
    alarm meşru olduğu için yapay olarak düşük bir yanlış alarm oranı
    gösterirdi.
    """
    if clip_duration_s <= 0:
        raise ValueError("clip_duration_s sıfırdan büyük olmalıdır.")

    result = match_alarms(
        events,
        alarm_times_s,
        drowsiness_seconds=drowsiness_seconds,
        match_tolerance_s=match_tolerance_s,
    )

    eligible_count = len(result.hit_event_indices) + len(result.missed_event_indices)
    hit_count = len(result.hit_event_indices)
    false_alarms = [m for m in result.matches if m.is_false_alarm]
    latencies = [m.latency_s for m in result.matches if m.latency_s is not None]

    alarm_expected_seconds = sum(
        min(event.end_s, clip_duration_s) - min(event.start_s, clip_duration_s)
        for index, event in enumerate(events)
        if index in result.hit_event_indices or index in result.missed_event_indices
    )
    exposure_seconds = max(0.0, clip_duration_s - alarm_expected_seconds)

    return {
        "eligible_event_count": eligible_count,
        "hit_count": hit_count,
        "missed_count": len(result.missed_event_indices),
        "ineligible_event_count": len(result.ineligible_event_indices),
        "detection_rate": (hit_count / eligible_count) if eligible_count else None,
        "alarm_latency_mean_s": (sum(latencies) / len(latencies)) if latencies else None,
        "alarm_latency_median_s": percentile(latencies, 0.5) if latencies else None,
        "alarm_latency_p95_s": percentile(latencies, 0.95) if latencies else None,
        # Ham gecikmeler: aggregate_clip_metrics bunları havuzlar. Klip
        # ortalamalarının ortalamasını almak, farklı olay sayılı klipleri
        # eşit ağırlıklandırıp yanıltıcı bir p95 üretirdi.
        "alarm_latencies_s": tuple(latencies),
        "false_alarm_count": len(false_alarms),
        "false_alarm_exposure_s": exposure_seconds,
        "false_alarms_per_minute": (len(false_alarms) / exposure_seconds * 60.0) if exposure_seconds > 0 else None,
        "duplicate_alarm_count": result.duplicate_alarm_count,
        "clip_duration_s": clip_duration_s,
        **state_observability(states),
    }


def aggregate_clip_metrics(clip_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """
    Birden çok klibin sonucunu tek bir satıra indirir (koşul bazlı
    tablolar için).

    Oranlar klip ortalaması olarak DEĞİL, havuzlanmış sayımlardan
    hesaplanır: 1 olaylı bir klip ile 10 olaylı bir klibin tespit
    oranına eşit ağırlıkta katkı vermesi yanıltıcı olurdu.

    Gecikme yüzdelikleri de klip ortalamalarından değil, tüm
    kliplerdeki tekil gecikmelerin havuzundan hesaplanır.
    """
    if not clip_results:
        return {"clip_count": 0}

    def total(key: str) -> int:
        return sum(int(r.get(key) or 0) for r in clip_results)

    eligible = total("eligible_event_count")
    hits = total("hit_count")
    false_alarms = total("false_alarm_count")
    exposure = sum(float(r.get("false_alarm_exposure_s") or 0.0) for r in clip_results)
    frames = total("frame_count")
    blind_frames = sum(
        float(r.get("blind_frame_ratio") or 0.0) * int(r.get("frame_count") or 0) for r in clip_results
    )

    pooled_latencies: list[float] = []
    for r in clip_results:
        pooled_latencies.extend(r.get("alarm_latencies_s", ()))

    return {
        "clip_count": len(clip_results),
        "eligible_event_count": eligible,
        "hit_count": hits,
        "missed_count": total("missed_count"),
        "ineligible_event_count": total("ineligible_event_count"),
        "detection_rate": (hits / eligible) if eligible else None,
        "false_alarm_count": false_alarms,
        "false_alarms_per_minute": (false_alarms / exposure * 60.0) if exposure > 0 else None,
        "alarm_latency_mean_s": (sum(pooled_latencies) / len(pooled_latencies)) if pooled_latencies else None,
        # Medyan, tek bir çok geç alarmın ortalamayı sürüklediği küçük
        # örneklemlerde tipik davranışı ortalamadan daha iyi anlatır.
        "alarm_latency_median_s": percentile(pooled_latencies, 0.5) if pooled_latencies else None,
        "alarm_latency_p95_s": percentile(pooled_latencies, 0.95) if pooled_latencies else None,
        "alarm_latencies_s": tuple(pooled_latencies),
        "blind_frame_ratio": (blind_frames / frames) if frames else None,
        "frame_count": frames,
    }
