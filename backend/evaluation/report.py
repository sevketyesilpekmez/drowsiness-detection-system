"""
Değerlendirme sonuçlarını okunabilir Markdown tablolarına çevirir.

Saf: dosya yazmaz, yalnızca metin üretir (test edilebilir olsun diye).
Çıktı doğrudan README'ye yapıştırılabilecek biçimdedir.

Tek bir genel ortalama BİLEREK yeterli sayılmıyor: gözlüklü kullanımın
zayıf kalması, toplamda güçlü diğer koşulların arkasında gizlenebilir.
Bu yüzden hem koşul hem denek bazında kırılım üretiliyor.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from evaluation.metrics import aggregate_clip_metrics

# Koşul bazlı kırılım için: bir klip birden fazla koşul etiketi
# taşıyabilir (ör. ["glasses", "lowlight"]) ve her birinin satırına
# katkı verir. Toplam satırı bu yüzden koşul satırlarının toplamı
# olmak zorunda değil.
OVERALL_LABEL = "TOPLAM"

# Hiç koşul etiketi olmayan klipler için grup adı.
UNLABELLED_CONDITION = "etiketsiz"

# subject alanı boş bırakılmış klipler için grup adı.
UNKNOWN_SUBJECT = "bilinmiyor"


def _group(
    clip_rows: Sequence[Mapping[str, Any]],
    key_fn: Callable[[Mapping[str, Any]], list[str]],
) -> dict[str, dict[str, Any]]:
    """Klipleri verilen anahtar fonksiyonuna göre gruplar ve her grup
    için havuzlanmış metrikleri üretir."""
    buckets: dict[str, list[Mapping[str, Any]]] = {}

    for row in clip_rows:
        for key in key_fn(row):
            buckets.setdefault(key, []).append(row["metrics"])

    grouped = {key: aggregate_clip_metrics(metrics) for key, metrics in sorted(buckets.items())}
    grouped[OVERALL_LABEL] = aggregate_clip_metrics([row["metrics"] for row in clip_rows])

    return grouped


def group_by_condition(clip_rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Klipleri koşul etiketlerine göre gruplar.

    Bir klip birden fazla koşul taşıyabilir (ör. gözlüklü VE düşük ışık)
    ve her grubun satırına katkı verir; bu yüzden koşul satırlarının
    toplamı TOPLAM satırına eşit OLMAK ZORUNDA DEĞİLDİR. Raporu okuyan
    biri bunu bilmezse sayıların tutmadığını sanabilir.
    """
    return _group(clip_rows, lambda row: list(row.get("conditions") or []) or [UNLABELLED_CONDITION])


def group_by_subject(clip_rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Klipleri deneğe göre gruplar.

    Koşul kırılımının gizlediği bir şeyi açar: sistem belirli bir KİŞİDE
    kötü çalışıyor olabilir (göz şekli, ten tonu, gözlük çerçevesi).
    Koşul ortalamasında bu, o kişinin bulunduğu her koşula dağılarak
    seyrelir. Bir denekte belirgin şekilde düşük tespit oranı, koşul
    tablosundaki hiçbir satırda görünmeyebilir.

    Her klip tek bir deneğe ait olduğu için bu gruplar örtüşmez;
    koşulların aksine satırların toplamı TOPLAM'a eşittir.
    """
    return _group(clip_rows, lambda row: [row.get("subject") or UNKNOWN_SUBJECT])


def format_ratio(value: float | None, *, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def format_seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def format_percent(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def format_rate_with_counts(rate: float | None, hits: Any, total: Any) -> str:
    """
    Oranı ham sayılarla birlikte gösterir: "%80.0 (4/5)".

    Küçük örneklemde çıplak yüzde yanıltıcı: 4/5 ile 40/50 aynı %80'dir
    ama güven düzeyleri çok farklıdır. Koşul başına 4-5 olayla
    çalışıldığı için sayılar her zaman görünür olmalı.
    """
    if rate is None:
        return f"- (0/{int(total or 0)})"
    return f"{rate * 100:.1f}% ({int(hits or 0)}/{int(total or 0)})"


def _latency_cell(metrics: Mapping[str, Any]) -> str:
    return "{} / {} / {}".format(
        format_seconds(metrics.get("alarm_latency_mean_s")),
        format_seconds(metrics.get("alarm_latency_median_s")),
        format_seconds(metrics.get("alarm_latency_p95_s")),
    )


def clip_table(clip_rows: Sequence[Mapping[str, Any]]) -> str:
    """Klip bazlı ayrıntı tablosu."""
    lines = [
        "| Klip | Denek | Olay | İsabet | Gecikme ort/med/p95 (sn) | Yanlış alarm/dk | Kör kare |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in clip_rows:
        metrics = row["metrics"]
        lines.append(
            "| {clip} | {subject} | {events} | {hits} | {latency} | {fa} | {blind} |".format(
                clip=row["clip"],
                subject=row.get("subject") or UNKNOWN_SUBJECT,
                events=metrics["eligible_event_count"],
                hits=metrics["hit_count"],
                latency=_latency_cell(metrics),
                fa=format_ratio(metrics["false_alarms_per_minute"], digits=2),
                blind=format_percent(metrics["blind_frame_ratio"]),
            )
        )

    return "\n".join(lines)


def _breakdown_table(rows: Mapping[str, Mapping[str, Any]], *, first_column: str) -> str:
    lines = [
        f"| {first_column} | Klip | Tespit | Gecikme ort/med/p95 (sn) | Yanlış alarm/dk | Kör kare |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    # TOPLAM her zaman en sonda; diğerleri alfabetik (deterministik çıktı).
    ordered = sorted(k for k in rows if k != OVERALL_LABEL)
    if OVERALL_LABEL in rows:
        ordered.append(OVERALL_LABEL)

    for key in ordered:
        agg = rows[key]
        lines.append(
            "| {key} | {clips} | {rate} | {latency} | {fa} | {blind} |".format(
                key=key,
                clips=agg["clip_count"],
                rate=format_rate_with_counts(
                    agg["detection_rate"], agg["hit_count"], agg["eligible_event_count"]
                ),
                latency=_latency_cell(agg),
                fa=format_ratio(agg["false_alarms_per_minute"], digits=2),
                blind=format_percent(agg["blind_frame_ratio"]),
            )
        )

    return "\n".join(lines)


def condition_table(condition_rows: Mapping[str, Mapping[str, Any]]) -> str:
    """Koşul bazlı özet — raporun asıl gösterilecek kısmı."""
    return _breakdown_table(condition_rows, first_column="Koşul")


def subject_table(subject_rows: Mapping[str, Mapping[str, Any]]) -> str:
    """Denek bazlı özet — koşul ortalamasının gizlediği kişisel
    başarısızlıkları açar."""
    return _breakdown_table(subject_rows, first_column="Denek")


def build_markdown_report(
    *,
    clip_rows: Sequence[Mapping[str, Any]],
    condition_rows: Mapping[str, Mapping[str, Any]],
    run_info: Mapping[str, Any],
    subject_rows: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    """Tam raporu üretir: koşullar (parametreler), özetler, klip ayrıntısı."""
    settings = " · ".join(f"{key}={value}" for key, value in sorted(run_info.items()))

    ineligible = sum(int(r["metrics"].get("ineligible_event_count") or 0) for r in clip_rows)
    warnings = []
    if ineligible:
        warnings.append(
            f"> **{ineligible} olay eşikten kısa** olduğu için tespit oranına dahil edilmedi. "
            "Alarm üretmemeleri doğru davranış; bu sayı yüksekse etiketleme ya da "
            "`DROWSINESS_SECONDS` gözden geçirilmeli."
        )

    errors = sum(int(r.get("detect_error_count") or 0) for r in clip_rows)
    if errors:
        warnings.append(f"> **{errors} karede detect_fn hata verdi** (MODEL_ERROR olarak sayıldı).")

    total_events = sum(int(r["metrics"].get("eligible_event_count") or 0) for r in clip_rows)
    if 0 < total_events < 20:
        warnings.append(
            f"> **Yalnızca {total_events} olay var.** Bu örneklemde tespit oranındaki "
            "birkaç puanlık farklar gürültüdür; oranların yanındaki ham sayılara bakın."
        )

    sections = [
        "## Uçtan Uca Değerlendirme",
        "",
        f"`{settings}`",
        "",
        "### Koşul bazlı",
        "",
        condition_table(condition_rows),
        "",
        "> Bir klip birden fazla koşul taşıyabilir, bu yüzden koşul satırlarının "
        "toplamı TOPLAM satırına eşit olmak zorunda değildir.",
    ]

    if subject_rows:
        sections.extend(["", "### Denek bazlı", "", subject_table(subject_rows)])

    sections.extend(["", "### Klip bazlı", "", clip_table(clip_rows)])

    if warnings:
        sections.extend(["", *warnings])

    return "\n".join(sections) + "\n"
