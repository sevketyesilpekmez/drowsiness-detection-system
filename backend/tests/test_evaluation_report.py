from __future__ import annotations

import pytest

from evaluation.metrics import GroundTruthEvent, clip_metrics
from evaluation.report import (
    OVERALL_LABEL,
    UNKNOWN_SUBJECT,
    UNLABELLED_CONDITION,
    build_markdown_report,
    clip_table,
    condition_table,
    format_percent,
    format_rate_with_counts,
    format_ratio,
    format_seconds,
    group_by_condition,
    group_by_subject,
    subject_table,
)

THRESHOLD = 2.5


def make_row(
    clip, conditions, *, alarms=(12.6,), events=((10.0, 16.0),), states=("AWAKE",), subject=None
):
    metrics = clip_metrics(
        events=[GroundTruthEvent(start_s=s, end_s=e) for s, e in events],
        alarm_times_s=list(alarms),
        clip_duration_s=60.0,
        drowsiness_seconds=THRESHOLD,
        states=list(states),
    )
    return {
        "clip": clip,
        "conditions": list(conditions),
        "subject": subject,
        "metrics": metrics,
    }


# --- biçimlendirme ----------------------------------------------------------


def test_formatters_render_none_as_dash():
    """None ile 0.0 rapor tablosunda aynı şey değil: biri 'ölçüm yoktu',
    diğeri 'ölçüldü ve sıfır çıktı'."""
    assert format_ratio(None) == "-"
    assert format_seconds(None) == "-"
    assert format_percent(None) == "-"

    assert format_ratio(0.0) == "0.000"
    assert format_seconds(0.0) == "0.00"
    assert format_percent(0.0) == "0.0%"


def test_percent_formatting():
    assert format_percent(0.9522) == "95.2%"


# --- group_by_condition -----------------------------------------------------


def test_clip_contributes_to_every_condition_it_carries():
    rows = [make_row("a.mp4", ["glasses", "lowlight"])]

    grouped = group_by_condition(rows)

    assert grouped["glasses"]["clip_count"] == 1
    assert grouped["lowlight"]["clip_count"] == 1
    assert grouped[OVERALL_LABEL]["clip_count"] == 1


def test_condition_rows_need_not_sum_to_overall():
    """Çok koşullu klipler yüzünden koşul satırlarının toplamı TOPLAM'a
    eşit değil — rapor okuyucusunun bilmesi gereken bir özellik."""
    rows = [make_row("a.mp4", ["glasses", "lowlight"]), make_row("b.mp4", ["normal"])]

    grouped = group_by_condition(rows)

    per_condition_total = sum(
        grouped[c]["clip_count"] for c in grouped if c != OVERALL_LABEL
    )
    assert per_condition_total == 3
    assert grouped[OVERALL_LABEL]["clip_count"] == 2


def test_clip_without_conditions_is_bucketed_separately():
    grouped = group_by_condition([make_row("a.mp4", [])])

    assert UNLABELLED_CONDITION in grouped
    assert grouped[UNLABELLED_CONDITION]["clip_count"] == 1


def test_grouping_pools_events_across_clips():
    rows = [
        make_row("a.mp4", ["normal"], alarms=(12.6,), events=((10.0, 16.0),)),
        make_row("b.mp4", ["normal"], alarms=(), events=((10.0, 16.0),)),
    ]

    grouped = group_by_condition(rows)

    assert grouped["normal"]["eligible_event_count"] == 2
    assert grouped["normal"]["hit_count"] == 1
    assert grouped["normal"]["detection_rate"] == pytest.approx(0.5)


def test_group_by_condition_handles_empty_input():
    grouped = group_by_condition([])

    assert grouped[OVERALL_LABEL]["clip_count"] == 0


# --- tablolar ---------------------------------------------------------------


def test_condition_table_puts_overall_last_and_sorts_the_rest():
    """Deterministik çıktı: aynı veri her koşuda aynı tabloyu üretmeli,
    yoksa rapor diff'leri gürültüye boğulur."""
    rows = [make_row("a.mp4", ["zzz"]), make_row("b.mp4", ["aaa"])]

    table = condition_table(group_by_condition(rows))
    lines = [line for line in table.splitlines() if line.startswith("| ")][2:]

    assert lines[0].startswith("| aaa")
    assert lines[1].startswith("| zzz")
    assert lines[2].startswith(f"| {OVERALL_LABEL}")


def test_clip_table_lists_every_clip():
    rows = [make_row("a.mp4", ["normal"]), make_row("b.mp4", ["normal"])]

    table = clip_table(rows)

    assert "a.mp4" in table
    assert "b.mp4" in table


def test_report_warns_about_ineligible_events():
    """Eşikten kısa olaylar sessizce yutulmamalı — genelde etiketleme
    hatasına ya da yanlış eşiğe işaret ederler."""
    rows = [make_row("a.mp4", ["normal"], alarms=(), events=((10.0, 11.0),))]

    report = build_markdown_report(
        clip_rows=rows,
        condition_rows=group_by_condition(rows),
        run_info={"target_fps": 8},
    )

    assert "eşikten kısa" in report


def test_report_warns_about_detect_errors():
    rows = [make_row("a.mp4", ["normal"])]
    rows[0]["detect_error_count"] = 3

    report = build_markdown_report(
        clip_rows=rows, condition_rows=group_by_condition(rows), run_info={}
    )

    assert "detect_fn hata verdi" in report


def test_report_includes_run_settings_so_numbers_are_reproducible():
    rows = [make_row("a.mp4", ["normal"])]

    report = build_markdown_report(
        clip_rows=rows,
        condition_rows=group_by_condition(rows),
        run_info={"target_fps": 8, "drowsiness_seconds": 2.5},
    )

    assert "target_fps=8" in report
    assert "drowsiness_seconds=2.5" in report


def test_report_has_no_warnings_when_everything_is_clean():
    rows = [make_row("a.mp4", ["normal"])]

    report = build_markdown_report(
        clip_rows=rows, condition_rows=group_by_condition(rows), run_info={}
    )

    assert "eşikten kısa" not in report
    assert "detect_fn hata verdi" not in report


# --- kişi bazlı kırılım -----------------------------------------------------


def test_group_by_subject_splits_per_person():
    rows = [
        make_row("a.mp4", ["normal"], subject="S01"),
        make_row("b.mp4", ["glasses"], subject="S01"),
        make_row("c.mp4", ["normal"], subject="S02"),
    ]

    grouped = group_by_subject(rows)

    assert grouped["S01"]["clip_count"] == 2
    assert grouped["S02"]["clip_count"] == 1
    assert grouped[OVERALL_LABEL]["clip_count"] == 3


def test_subject_rows_sum_to_overall_unlike_conditions():
    """Her klip TEK bir deneğe ait, dolayısıyla koşulların aksine
    denek satırları örtüşmez ve toplamları TOPLAM'a eşittir."""
    rows = [
        make_row("a.mp4", ["glasses", "lowlight"], subject="S01"),
        make_row("b.mp4", ["normal"], subject="S02"),
    ]

    subjects = group_by_subject(rows)

    per_subject = sum(subjects[k]["clip_count"] for k in subjects if k != OVERALL_LABEL)
    assert per_subject == subjects[OVERALL_LABEL]["clip_count"] == 2


def test_missing_subject_is_bucketed_as_unknown():
    grouped = group_by_subject([make_row("a.mp4", ["normal"], subject=None)])

    assert grouped[UNKNOWN_SUBJECT]["clip_count"] == 1


def test_subject_breakdown_reveals_a_person_hidden_by_condition_average():
    """KIRILIMIN VARLIK SEBEBİ: S02 hiçbir olayı yakalamıyor ama iki
    koşula dağıldığı için koşul tablosunda ikisi de %50 görünüyor —
    hiçbir satır "bu kişide sistem çalışmıyor" demiyor."""
    rows = [
        make_row("a.mp4", ["normal"], subject="S01", alarms=(12.6,)),
        make_row("b.mp4", ["glasses"], subject="S01", alarms=(12.6,)),
        make_row("c.mp4", ["normal"], subject="S02", alarms=()),
        make_row("d.mp4", ["glasses"], subject="S02", alarms=()),
    ]

    conditions = group_by_condition(rows)
    subjects = group_by_subject(rows)

    assert conditions["normal"]["detection_rate"] == pytest.approx(0.5)
    assert conditions["glasses"]["detection_rate"] == pytest.approx(0.5)
    # Denek kırılımı sorunu açıkça gösteriyor.
    assert subjects["S01"]["detection_rate"] == pytest.approx(1.0)
    assert subjects["S02"]["detection_rate"] == pytest.approx(0.0)


def test_subject_table_renders():
    rows = [make_row("a.mp4", ["normal"], subject="S01")]

    table = subject_table(group_by_subject(rows))

    assert "| Denek |" in table
    assert "S01" in table


def test_report_includes_subject_section_when_provided():
    rows = [make_row("a.mp4", ["normal"], subject="S01")]

    report = build_markdown_report(
        clip_rows=rows,
        condition_rows=group_by_condition(rows),
        subject_rows=group_by_subject(rows),
        run_info={},
    )

    assert "Denek bazlı" in report


def test_report_omits_subject_section_when_absent():
    rows = [make_row("a.mp4", ["normal"])]

    report = build_markdown_report(
        clip_rows=rows, condition_rows=group_by_condition(rows), run_info={}
    )

    assert "Denek bazlı" not in report


# --- medyan ve ham sayılar --------------------------------------------------


def test_rate_is_shown_with_raw_counts():
    """Küçük örneklemde çıplak yüzde yanıltıcı: 4/5 ile 40/50 aynı %80
    ama güven düzeyleri çok farklı."""
    assert format_rate_with_counts(0.8, 4, 5) == "80.0% (4/5)"


def test_rate_with_no_events_shows_dash_and_zero():
    assert format_rate_with_counts(None, 0, 0) == "- (0/0)"


def test_tables_include_median_latency():
    rows = [make_row("a.mp4", ["normal"], subject="S01")]

    assert "ort/med/p95" in condition_table(group_by_condition(rows))
    assert "ort/med/p95" in clip_table(rows)


def test_aggregate_exposes_median_latency():
    from evaluation.metrics import aggregate_clip_metrics

    rows = [
        make_row("a.mp4", ["n"], alarms=(12.6,)),
        make_row("b.mp4", ["n"], alarms=(13.6,)),
        make_row("c.mp4", ["n"], alarms=(14.6,)),
    ]

    agg = aggregate_clip_metrics([r["metrics"] for r in rows])

    assert agg["alarm_latency_median_s"] == pytest.approx(1.1)


def test_report_warns_when_the_sample_is_small():
    rows = [make_row("a.mp4", ["normal"], subject="S01")]

    report = build_markdown_report(
        clip_rows=rows, condition_rows=group_by_condition(rows), run_info={}
    )

    assert "gürültüdür" in report


def test_report_explains_why_condition_rows_do_not_sum():
    rows = [make_row("a.mp4", ["glasses", "lowlight"])]

    report = build_markdown_report(
        clip_rows=rows, condition_rows=group_by_condition(rows), run_info={}
    )

    assert "eşit olmak zorunda değildir" in report
