from __future__ import annotations

import json

import pytest

from evaluation.clock import FrameClock, frame_step_for_target_fps
from evaluation.ground_truth import load_clip_labels, load_label_directory, parse_clip_labels
from evaluation.runner import replay, timeline_to_rows


# --- FrameClock -------------------------------------------------------------


def test_clock_time_is_derived_from_frame_index():
    clock = FrameClock(fps=30.0)

    assert clock() == pytest.approx(0.0)
    clock.seek_to_frame(30)
    assert clock() == pytest.approx(1.0)
    clock.seek_to_frame(45)
    assert clock() == pytest.approx(1.5)


def test_clock_is_correct_when_frames_are_skipped():
    """Kare atlandığında da zaman gerçek video zamanı kalmalı: 30 FPS'te
    4 kare atlamak 133 ms geçtiği anlamına gelir."""
    clock = FrameClock(fps=30.0)

    clock.seek_to_frame(4)

    assert clock() == pytest.approx(4 / 30)


def test_clock_refuses_to_go_backwards():
    """DrowsinessService süreleri now-since ile hesaplıyor; saatin geri
    gitmesi negatif süreler ve sessizce yanlış kararlar üretirdi."""
    clock = FrameClock(fps=30.0)
    clock.seek_to_frame(10)

    with pytest.raises(ValueError):
        clock.seek_to_frame(9)


def test_clock_rejects_invalid_construction():
    with pytest.raises(ValueError):
        FrameClock(fps=0)
    with pytest.raises(ValueError):
        FrameClock(fps=30.0, start_time_s=-1.0)


def test_clock_is_callable_like_time_monotonic():
    """DrowsinessService/AlarmService `Callable[[], float]` bekliyor."""
    clock = FrameClock(fps=10.0)

    assert callable(clock)
    assert isinstance(clock(), float)


# --- frame_step_for_target_fps ----------------------------------------------


@pytest.mark.parametrize(
    ("source_fps", "target_fps", "expected_step"),
    [
        (30.0, 8.0, 4),
        (30.0, 15.0, 2),
        (30.0, 30.0, 1),
        (30.0, 60.0, 1),  # kaynaktan hızlı işlenemez
        (30.0, None, 1),
        (25.0, 5.0, 5),
    ],
)
def test_frame_step_matches_target_rate(source_fps, target_fps, expected_step):
    assert frame_step_for_target_fps(source_fps, target_fps) == expected_step


def test_frame_step_rejects_invalid_input():
    with pytest.raises(ValueError):
        frame_step_for_target_fps(0, 8.0)
    with pytest.raises(ValueError):
        frame_step_for_target_fps(30.0, 0)


# --- replay -----------------------------------------------------------------


def constant_detect_fn(result):
    return lambda frame: dict(result)


AWAKE_RESULT = {
    "state": "AWAKE",
    "is_drowsy": False,
    "eyes_closed": False,
    "confidence": 0.9,
    "closed_duration": 0.0,
    "alarm_triggered": False,
}


def test_replay_produces_one_entry_per_processed_frame():
    result = replay(range(10), constant_detect_fn(AWAKE_RESULT), source_fps=10.0)

    assert result.processed_frame_count == 10
    assert result.source_frame_count == 10
    assert result.duration_s == pytest.approx(1.0)


def test_replay_skips_frames_according_to_step():
    result = replay(range(12), constant_detect_fn(AWAKE_RESULT), source_fps=30.0, frame_step=4)

    assert result.processed_frame_count == 3
    assert [e.frame_index for e in result.entries] == [0, 4, 8]
    # Süre işlenen kareye değil KAYNAK kare sayısına dayanır.
    assert result.duration_s == pytest.approx(12 / 30)


def test_replay_timestamps_are_real_video_times_not_processing_order():
    result = replay(range(9), constant_detect_fn(AWAKE_RESULT), source_fps=30.0, frame_step=3)

    assert [round(e.time_s, 4) for e in result.entries] == [0.0, 0.1, 0.2]


def test_replay_collects_alarm_times():
    alarm_frames = {5, 6}

    def detect_fn(frame):
        return {**AWAKE_RESULT, "state": "DROWSY", "alarm_triggered": frame in alarm_frames}

    result = replay(range(10), detect_fn, source_fps=10.0)

    assert result.alarm_times_s == pytest.approx((0.5, 0.6))


def test_replay_is_deterministic():
    """Aynı girdi iki kez aynı sonucu vermeli — ölçüm aracının en temel
    gereksinimi. Gerçek saat kullanılsaydı bu test kararsız olurdu."""

    def detect_fn(frame):
        return {**AWAKE_RESULT, "alarm_triggered": frame == 7}

    first = replay(range(20), detect_fn, source_fps=25.0, frame_step=2)
    second = replay(range(20), detect_fn, source_fps=25.0, frame_step=2)

    assert first == second


def test_replay_records_detect_errors_as_model_error_and_continues():
    def flaky_detect_fn(frame):
        if frame == 3:
            raise RuntimeError("checkpoint bozuk")
        return dict(AWAKE_RESULT)

    result = replay(range(6), flaky_detect_fn, source_fps=10.0)

    assert result.detect_error_count == 1
    assert result.processed_frame_count == 6  # döngü durmadı
    assert result.states[3] == "MODEL_ERROR"
    assert "checkpoint bozuk" in result.entries[3].error


def test_replay_carries_measurement_valid_when_present():
    result = replay(
        range(2),
        constant_detect_fn({**AWAKE_RESULT, "state": "UNVERIFIABLE", "measurement_valid": False}),
        source_fps=10.0,
    )

    assert result.entries[0].measurement_valid is False
    assert result.states[0] == "UNVERIFIABLE"


def test_replay_tolerates_detect_fn_without_optional_fields():
    """mediapipe backend'i measurement_valid göndermiyor; replay bunu
    eksik alan olarak değil 'bilinmiyor' olarak taşımalı."""
    result = replay(range(1), constant_detect_fn({"state": "AWAKE"}), source_fps=10.0)

    assert result.entries[0].measurement_valid is None
    assert result.entries[0].closed_duration == 0.0


def test_replay_handles_empty_frame_source():
    result = replay([], constant_detect_fn(AWAKE_RESULT), source_fps=10.0)

    assert result.processed_frame_count == 0
    assert result.duration_s == 0.0
    assert result.alarm_times_s == ()


def test_replay_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        replay([], constant_detect_fn(AWAKE_RESULT), source_fps=0)
    with pytest.raises(ValueError):
        replay([], constant_detect_fn(AWAKE_RESULT), source_fps=10.0, frame_step=0)


def test_timeline_to_rows_is_json_serialisable():
    result = replay(range(3), constant_detect_fn(AWAKE_RESULT), source_fps=10.0)

    rows = timeline_to_rows(result.entries)

    assert json.loads(json.dumps(rows)) == rows


# --- ground truth -----------------------------------------------------------


MINIMAL_LABELS = {
    "clip": "test_01.mp4",
    "fps": 30.0,
    "duration_s": 60.0,
    "events": [{"start_s": 10.0, "end_s": 15.0}],
}


def test_parse_minimal_labels():
    labels = parse_clip_labels(MINIMAL_LABELS)

    assert labels.clip == "test_01.mp4"
    assert len(labels.events) == 1
    assert labels.events[0].expect_alarm is True
    assert labels.expected_event_count == 1


def test_parse_distractor_event():
    labels = parse_clip_labels(
        {**MINIMAL_LABELS, "events": [{"start_s": 1.0, "end_s": 1.3, "expect_alarm": False, "label": "blink"}]}
    )

    assert labels.events[0].expect_alarm is False
    assert labels.events[0].label == "blink"
    assert labels.expected_event_count == 0


def test_rejects_overlapping_expected_events():
    """Çakışan olaylar, bir alarmın hangi olaya ait olduğunu belirsiz
    kılar ve tespit oranını etiket sırasına bağımlı hale getirir."""
    with pytest.raises(ValueError, match="çakışıyor"):
        parse_clip_labels(
            {
                **MINIMAL_LABELS,
                "events": [{"start_s": 10.0, "end_s": 15.0}, {"start_s": 14.0, "end_s": 20.0}],
            }
        )


def test_allows_distractor_overlapping_an_expected_event():
    labels = parse_clip_labels(
        {
            **MINIMAL_LABELS,
            "events": [
                {"start_s": 10.0, "end_s": 15.0},
                {"start_s": 12.0, "end_s": 12.2, "expect_alarm": False, "label": "blink"},
            ],
        }
    )

    assert len(labels.events) == 2


def test_rejects_event_beyond_clip_duration():
    with pytest.raises(ValueError, match="klip sonunu aşıyor"):
        parse_clip_labels({**MINIMAL_LABELS, "events": [{"start_s": 50.0, "end_s": 70.0}]})


def test_rejects_missing_clip_name():
    with pytest.raises(ValueError, match="clip"):
        parse_clip_labels({"events": []})


def test_rejects_reversed_event_with_helpful_location():
    with pytest.raises(ValueError, match=r"events\[0\]"):
        parse_clip_labels({**MINIMAL_LABELS, "events": [{"start_s": 15.0, "end_s": 10.0}]})


def test_rejects_non_numeric_event_bounds():
    with pytest.raises(ValueError, match="sayısal"):
        parse_clip_labels({**MINIMAL_LABELS, "events": [{"start_s": "10", "end_s": 15.0}]})


def test_load_clip_labels_from_file(tmp_path):
    path = tmp_path / "test_01.json"
    path.write_text(json.dumps(MINIMAL_LABELS), encoding="utf-8")

    labels = load_clip_labels(path)

    assert labels.clip == "test_01.mp4"


def test_load_clip_labels_reports_invalid_json_with_path(tmp_path):
    path = tmp_path / "bozuk.json"
    path.write_text("{ bu json degil", encoding="utf-8")

    with pytest.raises(ValueError, match="geçersiz JSON"):
        load_clip_labels(path)


def test_load_label_directory_is_sorted(tmp_path):
    for name in ("b_clip", "a_clip"):
        (tmp_path / f"{name}.json").write_text(
            json.dumps({**MINIMAL_LABELS, "clip": f"{name}.mp4"}), encoding="utf-8"
        )

    labels = load_label_directory(tmp_path)

    assert [lbl.clip for lbl in labels] == ["a_clip.mp4", "b_clip.mp4"]


def test_load_label_directory_rejects_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="etiket dosyası yok"):
        load_label_directory(tmp_path)


# --- saat enjeksiyonu (regresyon) -------------------------------------------


def test_replay_advances_the_injected_clock():
    """REGRESYON: replay kendi saatini yaratıyordu. Gerçek dedektörün
    içindeki DrowsinessService/AlarmService süreyi ENJEKTE EDİLEN
    saatten okuduğu için, onlarınki hiç ilerlemiyor ve closed_duration
    sonsuza kadar 0 kalıyordu — yani hiçbir alarm üretilemiyordu."""
    clock = FrameClock(fps=30.0)
    seen_times = []

    def detect_fn(frame):
        seen_times.append(clock())
        return dict(AWAKE_RESULT)

    replay(range(9), detect_fn, source_fps=30.0, frame_step=3, clock=clock)

    assert seen_times == pytest.approx([0.0, 0.1, 0.2])


def test_replay_resets_injected_clock_between_clips():
    """Aynı saat birden çok klipte kullanılır; ikinci klip sıfırdan
    başlamalı, yoksa seek_to_frame'in geri gitme yasağı koşuyu patlatır."""
    clock = FrameClock(fps=30.0)

    replay(range(30), constant_detect_fn(AWAKE_RESULT), source_fps=30.0, clock=clock)
    second = replay(range(30), constant_detect_fn(AWAKE_RESULT), source_fps=30.0, clock=clock)

    assert second.entries[0].time_s == pytest.approx(0.0)


def test_replay_applies_new_fps_when_reusing_a_clock():
    clock = FrameClock(fps=30.0)

    result = replay(range(10), constant_detect_fn(AWAKE_RESULT), source_fps=10.0, clock=clock)

    assert result.entries[5].time_s == pytest.approx(0.5)
