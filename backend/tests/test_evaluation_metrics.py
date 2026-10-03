from __future__ import annotations

import pytest

from evaluation.metrics import (
    GroundTruthEvent,
    aggregate_clip_metrics,
    clip_metrics,
    match_alarms,
    percentile,
    state_observability,
)

THRESHOLD = 2.5


def event(start, end, *, expect_alarm=True, label="eyes_closed"):
    return GroundTruthEvent(start_s=start, end_s=end, expect_alarm=expect_alarm, label=label)


# --- GroundTruthEvent -------------------------------------------------------


def test_event_rejects_reversed_interval():
    with pytest.raises(ValueError):
        GroundTruthEvent(start_s=5.0, end_s=3.0)


def test_event_rejects_negative_start():
    with pytest.raises(ValueError):
        GroundTruthEvent(start_s=-1.0, end_s=3.0)


def test_event_shorter_than_threshold_is_not_eligible():
    """Eşik 2.5 sn iken 1.5 sn'lik bir kapanmanın alarm üretmemesi DOĞRU
    davranıştır — bunu kaçırma saymak sistemi haksız cezalandırır."""
    assert event(10.0, 11.5).is_eligible(THRESHOLD) is False
    assert event(10.0, 14.0).is_eligible(THRESHOLD) is True


def test_distractor_event_is_never_eligible():
    assert event(10.0, 20.0, expect_alarm=False).is_eligible(THRESHOLD) is False


# --- match_alarms: temel eşleştirme -----------------------------------------


def test_alarm_inside_window_is_a_hit_with_latency():
    events = [event(10.0, 15.0)]

    result = match_alarms(events, [12.9], drowsiness_seconds=THRESHOLD)

    assert result.hit_event_indices == {0}
    assert result.missed_event_indices == ()
    assert result.matches[0].latency_s == pytest.approx(0.4)  # 12.9 - (10.0 + 2.5)


def test_alarm_before_threshold_is_false_alarm_not_early_hit():
    """EN KRİTİK KURAL: sistem, gözler kapandığı anda alarm vermemeli.
    Eşikten önce gelen alarm erken isabet değil yanlış alarmdır — aksi
    halde 'her karede alarm çal' diyen bozuk bir sistem %100 tespit
    oranı alırdı."""
    events = [event(10.0, 15.0)]

    result = match_alarms(events, [11.0], drowsiness_seconds=THRESHOLD)

    assert result.hit_event_indices == frozenset()
    assert result.missed_event_indices == (0,)
    assert result.matches[0].is_false_alarm is True


def test_alarm_exactly_at_threshold_boundary_is_a_hit():
    events = [event(10.0, 15.0)]

    result = match_alarms(events, [12.5], drowsiness_seconds=THRESHOLD)

    assert result.hit_event_indices == {0}
    assert result.matches[0].latency_s == pytest.approx(0.0)


def test_alarm_within_tolerance_after_event_end_is_a_hit():
    events = [event(10.0, 15.0)]

    result = match_alarms(events, [15.3], drowsiness_seconds=THRESHOLD, match_tolerance_s=0.5)

    assert result.hit_event_indices == {0}


def test_alarm_beyond_tolerance_is_false_alarm():
    events = [event(10.0, 15.0)]

    result = match_alarms(events, [16.0], drowsiness_seconds=THRESHOLD, match_tolerance_s=0.5)

    assert result.hit_event_indices == frozenset()
    assert result.matches[0].is_false_alarm is True


def test_event_without_alarm_is_missed():
    events = [event(10.0, 15.0)]

    result = match_alarms(events, [], drowsiness_seconds=THRESHOLD)

    assert result.missed_event_indices == (0,)


def test_repeated_alarms_in_same_event_count_once_and_are_not_false_alarms():
    """Alarm cooldown'ı aynı olay içinde tekrar tetikleyebilir; bu bir
    hata değil, yanlış alarm olarak sayılmamalı."""
    events = [event(10.0, 20.0)]

    result = match_alarms(events, [12.6, 17.6], drowsiness_seconds=THRESHOLD)

    assert result.hit_event_indices == {0}
    assert result.duplicate_alarm_count == 1
    assert [m for m in result.matches if m.is_false_alarm] == []


def test_alarm_inside_distractor_window_is_false_alarm():
    """Göz kırpma penceresine düşen alarm da yanlış alarmdır —
    çeldiriciler tespit oranına girmez ama yanlış alarmı gizlemez."""
    events = [event(10.0, 10.4, expect_alarm=False, label="blink")]

    result = match_alarms(events, [10.2], drowsiness_seconds=THRESHOLD)

    assert result.matches[0].is_false_alarm is True
    assert result.hit_event_indices == frozenset()


def test_ineligible_event_is_reported_separately_not_as_miss():
    events = [event(10.0, 11.0)]  # 1.0 sn < 2.5 sn eşik

    result = match_alarms(events, [], drowsiness_seconds=THRESHOLD)

    assert result.missed_event_indices == ()
    assert result.ineligible_event_indices == (0,)


def test_alarms_are_matched_in_time_order_regardless_of_input_order():
    events = [event(10.0, 15.0), event(30.0, 35.0)]

    unordered = match_alarms(events, [33.0, 13.0], drowsiness_seconds=THRESHOLD)
    ordered = match_alarms(events, [13.0, 33.0], drowsiness_seconds=THRESHOLD)

    assert unordered.hit_event_indices == ordered.hit_event_indices == {0, 1}
    assert [m.alarm_time_s for m in unordered.matches] == [13.0, 33.0]


def test_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        match_alarms([], [], drowsiness_seconds=0)
    with pytest.raises(ValueError):
        match_alarms([], [], drowsiness_seconds=THRESHOLD, match_tolerance_s=-0.1)


# --- percentile -------------------------------------------------------------


def test_percentile_interpolates_linearly():
    assert percentile([0.0, 1.0], 0.5) == pytest.approx(0.5)
    assert percentile([0.0, 10.0], 0.9) == pytest.approx(9.0)


def test_percentile_single_value():
    assert percentile([4.2], 0.95) == pytest.approx(4.2)


def test_percentile_rejects_empty_and_out_of_range():
    with pytest.raises(ValueError):
        percentile([], 0.5)
    with pytest.raises(ValueError):
        percentile([1.0], 1.5)


# --- state_observability ----------------------------------------------------


def test_blind_frame_ratio_counts_unmeasurable_states():
    states = ["AWAKE", "AWAKE", "UNVERIFIABLE", "NO_FACE"]

    result = state_observability(states)

    assert result["frame_count"] == 4
    assert result["blind_frame_ratio"] == pytest.approx(0.5)
    assert result["state_counts"]["UNVERIFIABLE"] == 1


def test_state_observability_handles_empty_input():
    assert state_observability([])["blind_frame_ratio"] == 0.0


# --- clip_metrics -----------------------------------------------------------


def test_clip_metrics_happy_path():
    events = [event(10.0, 15.0), event(40.0, 46.0)]

    result = clip_metrics(
        events=events,
        alarm_times_s=[12.8, 43.0],
        clip_duration_s=60.0,
        drowsiness_seconds=THRESHOLD,
        states=["AWAKE"] * 90 + ["NO_FACE"] * 10,
    )

    assert result["detection_rate"] == pytest.approx(1.0)
    assert result["false_alarm_count"] == 0
    assert result["alarm_latency_mean_s"] == pytest.approx((0.3 + 0.5) / 2)
    assert result["blind_frame_ratio"] == pytest.approx(0.1)


def test_false_alarm_rate_excludes_time_where_alarming_is_legitimate():
    """Payda klibin tamamı olsaydı, uzun bir uyku olayı içeren klip
    yapay olarak düşük bir yanlış alarm oranı gösterirdi."""
    events = [event(0.0, 30.0)]  # klibin yarısı alarm beklenen pencere

    result = clip_metrics(
        events=events,
        alarm_times_s=[2.5, 55.0],
        clip_duration_s=60.0,
        drowsiness_seconds=THRESHOLD,
    )

    assert result["false_alarm_exposure_s"] == pytest.approx(30.0)
    assert result["false_alarm_count"] == 1
    assert result["false_alarms_per_minute"] == pytest.approx(2.0)


def test_clip_metrics_with_no_eligible_events_reports_none_detection_rate():
    """Sıfıra bölmek yerine None — 'olay yoktu' ile '%0 tespit' rapor
    tablosunda aynı şey değildir."""
    result = clip_metrics(
        events=[],
        alarm_times_s=[],
        clip_duration_s=60.0,
        drowsiness_seconds=THRESHOLD,
    )

    assert result["detection_rate"] is None
    assert result["alarm_latency_mean_s"] is None


def test_clip_metrics_rejects_non_positive_duration():
    with pytest.raises(ValueError):
        clip_metrics(events=[], alarm_times_s=[], clip_duration_s=0.0, drowsiness_seconds=THRESHOLD)


# --- aggregate_clip_metrics -------------------------------------------------


def test_aggregate_pools_counts_instead_of_averaging_rates():
    """1 olaylı klip ile 10 olaylı klip tespit oranına eşit ağırlıkta
    katkı vermemeli."""
    small = clip_metrics(
        events=[event(0.0, 5.0)],
        alarm_times_s=[2.6],
        clip_duration_s=10.0,
        drowsiness_seconds=THRESHOLD,
    )
    big = clip_metrics(
        events=[event(t, t + 5.0) for t in range(0, 100, 10)],
        alarm_times_s=[],
        clip_duration_s=110.0,
        drowsiness_seconds=THRESHOLD,
    )

    result = aggregate_clip_metrics([small, big])

    # Klip ortalaması olsaydı (1.0 + 0.0) / 2 = 0.5 çıkardı.
    assert result["eligible_event_count"] == 11
    assert result["hit_count"] == 1
    assert result["detection_rate"] == pytest.approx(1 / 11)


def test_aggregate_pools_latencies_for_percentiles():
    clips = [
        clip_metrics(
            events=[event(10.0, 20.0)],
            alarm_times_s=[12.5 + delay],
            clip_duration_s=30.0,
            drowsiness_seconds=THRESHOLD,
        )
        for delay in (0.1, 0.2, 0.3, 2.0)
    ]

    result = aggregate_clip_metrics(clips)

    assert result["alarm_latency_mean_s"] == pytest.approx((0.1 + 0.2 + 0.3 + 2.0) / 4)
    assert result["alarm_latency_p95_s"] > 1.0


def test_aggregate_handles_empty_input():
    assert aggregate_clip_metrics([]) == {"clip_count": 0}
