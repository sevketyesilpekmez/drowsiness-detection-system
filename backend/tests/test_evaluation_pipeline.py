"""
Değerlendirme hattının uçtan uca entegrasyon testi.

Diğer değerlendirme testleri parçaları ayrı ayrı doğruluyor; buradaki
soru farklı: gerçek bir video dosyası okunduğunda video_frames ->
replay -> metrics zinciri DOĞRU ZAMAN DAMGALARINI üretiyor ve bilinen
bir olayı yakalıyor mu?

Gerçek dedektör yerine, senaryosu önceden bilinen sahte bir detect_fn
kullanılıyor: amaç modelin başarımını değil, ÖLÇÜM ARACININ kendisinin
doğruluğunu sınamak. Ölçüm aracı yanlışsa modelin ne kadar iyi olduğu
da bilinemez.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from evaluation.clock import frame_step_for_target_fps
from evaluation.ground_truth import parse_clip_labels
from evaluation.metrics import clip_metrics
from evaluation.report import build_markdown_report, group_by_condition
from evaluation.runner import replay
from evaluation.video_frames import probe_video, read_frames

FPS = 30.0
DURATION_S = 6.0
FRAME_COUNT = int(FPS * DURATION_S)
THRESHOLD = 1.0

# Gözlerin kapalı olduğu aralık — hem videoyu üreten sahte dedektör hem
# de ground truth bunu kullanır.
CLOSED_START_S = 2.0
CLOSED_END_S = 5.0


@pytest.fixture
def synthetic_clip(tmp_path):
    """Kare indeksi görüntüye gömülü küçük bir video üretir."""
    path = tmp_path / "sentetik.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 48))

    if not writer.isOpened():
        pytest.skip("Bu ortamda mp4v codec'i yok, sentetik video üretilemiyor.")

    for _ in range(FRAME_COUNT):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()

    if not path.exists() or path.stat().st_size == 0:
        pytest.skip("Sentetik video yazılamadı.")

    return path


class ScriptedDetector:
    """
    Belirlenen zaman aralığında "gözler kapalı" diyen sahte dedektör.

    Zamanı kendi çağrı sayacından türetir: `frame_step` kadar aralıklarla
    çağrıldığını bilir, bu yüzden kare atlansa da gözlediği zaman
    replay'in verdiği zaman damgasıyla aynı kalır.

    Alarm, kapalı serisi eşiği aştıktan sonraki İLK karede bir kez
    tetiklenir (gerçek AlarmService'in cooldown'ı burada taklit
    edilmiyor — amaç ölçüm aracını sınamak, dedektörü değil).
    """

    def __init__(self, *, fps, frame_step, closed_start_s, closed_end_s, threshold_s):
        self._fps = fps
        self._frame_step = frame_step
        self._closed_start_s = closed_start_s
        self._closed_end_s = closed_end_s
        self._threshold_s = threshold_s
        self._call_count = 0
        self._closed_since = None
        self._alarm_fired = False

    def __call__(self, frame):
        time_s = (self._call_count * self._frame_step) / self._fps
        self._call_count += 1

        eyes_closed = self._closed_start_s <= time_s < self._closed_end_s

        if not eyes_closed:
            self._closed_since = None
            self._alarm_fired = False
            return {
                "state": "AWAKE",
                "eyes_closed": False,
                "closed_duration": 0.0,
                "alarm_triggered": False,
            }

        if self._closed_since is None:
            self._closed_since = time_s

        closed_duration = time_s - self._closed_since
        is_drowsy = closed_duration >= self._threshold_s
        triggered = is_drowsy and not self._alarm_fired

        if triggered:
            self._alarm_fired = True

        return {
            "state": "DROWSY" if is_drowsy else "AWAKE",
            "eyes_closed": True,
            "closed_duration": closed_duration,
            "alarm_triggered": triggered,
        }


def run_pipeline(clip_path, *, target_fps=None):
    """video -> replay -> alarm zaman damgaları."""
    info = probe_video(clip_path)
    step = frame_step_for_target_fps(info.fps, target_fps)
    detect_fn = ScriptedDetector(
        fps=info.fps,
        frame_step=step,
        closed_start_s=CLOSED_START_S,
        closed_end_s=CLOSED_END_S,
        threshold_s=THRESHOLD,
    )

    with read_frames(clip_path) as frames:
        return info, replay(frames, detect_fn, source_fps=info.fps, frame_step=step)


def test_probe_reads_video_properties(synthetic_clip):
    info = probe_video(synthetic_clip)

    assert info.fps == pytest.approx(FPS, abs=0.5)
    assert info.frame_count == pytest.approx(FRAME_COUNT, abs=2)


def test_probe_rejects_missing_file(tmp_path):
    with pytest.raises(ValueError, match="bulunamadı"):
        probe_video(tmp_path / "yok.mp4")


def test_read_frames_yields_every_frame_and_stops_at_end(synthetic_clip):
    """VideoSource'un aksine dosya sonunda BAŞA SARMAMALI — sarsaydı
    değerlendirme koşusu sonsuz döngüye girerdi."""
    with read_frames(synthetic_clip) as frames:
        count = sum(1 for _ in frames)

    assert count == pytest.approx(FRAME_COUNT, abs=2)


def test_pipeline_detects_known_event_with_expected_latency(synthetic_clip):
    info, result = run_pipeline(synthetic_clip)

    labels = parse_clip_labels(
        {
            "clip": synthetic_clip.name,
            "fps": info.fps,
            "conditions": ["sentetik"],
            "events": [{"start_s": CLOSED_START_S, "end_s": CLOSED_END_S}],
        }
    )

    metrics = clip_metrics(
        events=labels.events,
        alarm_times_s=result.alarm_times_s,
        clip_duration_s=result.duration_s,
        drowsiness_seconds=THRESHOLD,
        states=result.states,
    )

    assert metrics["eligible_event_count"] == 1
    assert metrics["hit_count"] == 1
    assert metrics["false_alarm_count"] == 0
    # Alarm, eşik dolduktan sonraki ilk karede gelmeli: bir kare payı.
    assert metrics["alarm_latency_mean_s"] == pytest.approx(0.0, abs=1.5 / FPS)


def test_pipeline_result_is_identical_across_runs(synthetic_clip):
    """Determinizm, bu aracın en temel gereksinimi."""
    _, first = run_pipeline(synthetic_clip)
    _, second = run_pipeline(synthetic_clip)

    assert first.alarm_times_s == second.alarm_times_s
    assert first.states == second.states


def test_frame_skipping_changes_resolution_but_not_event_timing(synthetic_clip):
    """Kare atlamak alarm zamanını bir kare adımı kadar geciktirebilir
    ama olayı kaçırtmamalı — zaman damgaları gerçek video zamanı
    olduğu için ölçüm hâlâ karşılaştırılabilir kalır."""
    _, full = run_pipeline(synthetic_clip)
    _, decimated = run_pipeline(synthetic_clip, target_fps=6.0)

    assert decimated.processed_frame_count < full.processed_frame_count
    assert len(decimated.alarm_times_s) == 1
    assert decimated.alarm_times_s[0] == pytest.approx(full.alarm_times_s[0], abs=0.35)


def test_pipeline_output_renders_a_report(synthetic_clip):
    info, result = run_pipeline(synthetic_clip)

    metrics = clip_metrics(
        events=[],
        alarm_times_s=result.alarm_times_s,
        clip_duration_s=result.duration_s,
        drowsiness_seconds=THRESHOLD,
        states=result.states,
    )
    rows = [{"clip": synthetic_clip.name, "conditions": ["sentetik"], "metrics": metrics}]

    report = build_markdown_report(
        clip_rows=rows,
        condition_rows=group_by_condition(rows),
        run_info={"target_fps": None, "drowsiness_seconds": THRESHOLD},
    )

    assert "Uçtan Uca Değerlendirme" in report
    assert "sentetik" in report
