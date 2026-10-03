from __future__ import annotations

import pytest

from app.services.eye_state_smoother import EyeStateSmoother


class FakeClock:
    def __init__(self, initial_time: float = 0.0) -> None:
        self._time = initial_time

    def __call__(self) -> float:
        return self._time

    def advance(self, seconds: float) -> None:
        self._time += seconds


def make_smoother(clock, **kwargs) -> EyeStateSmoother:
    defaults = dict(
        window_seconds=0.9,
        closed_ratio_threshold=0.6,
        missing_tolerance_seconds=0.25,
        minimum_valid_seconds=0.35,
    )
    defaults.update(kwargs)
    return EyeStateSmoother(clock=clock, **defaults)


def feed(smoother, clock, *, closed, count, step=0.125, valid=True, confidence=0.9):
    """`count` ölçüm besler, her ölçümden sonra `step` saniye geçirir."""
    result = False
    for _ in range(count):
        result = smoother.update(valid=valid, eyes_closed=closed, confidence=confidence)
        clock.advance(step)
    return result


# --- doğrulama --------------------------------------------------------------


def test_rejects_invalid_window_seconds() -> None:
    with pytest.raises(ValueError):
        EyeStateSmoother(window_seconds=0)


def test_rejects_invalid_closed_ratio_threshold() -> None:
    with pytest.raises(ValueError):
        EyeStateSmoother(closed_ratio_threshold=0)
    with pytest.raises(ValueError):
        EyeStateSmoother(closed_ratio_threshold=1.5)


def test_rejects_negative_missing_tolerance() -> None:
    with pytest.raises(ValueError):
        EyeStateSmoother(missing_tolerance_seconds=-0.1)


def test_rejects_negative_minimum_valid_seconds() -> None:
    with pytest.raises(ValueError):
        EyeStateSmoother(minimum_valid_seconds=-0.1)


def test_zero_minimum_valid_seconds_disables_the_startup_guard() -> None:
    """0.0 açık bir "korumayı kapat" değeri: test double'ları ham kararı
    ilk ölçümde yansıtabilsin diye."""
    clock = FakeClock()
    smoother = make_smoother(clock, minimum_valid_seconds=0.0)

    assert smoother.update(valid=True, eyes_closed=True, confidence=0.9) is True


def test_rejects_minimum_valid_seconds_greater_than_window() -> None:
    with pytest.raises(ValueError):
        EyeStateSmoother(window_seconds=0.5, minimum_valid_seconds=0.9)


def test_rejects_non_callable_clock() -> None:
    with pytest.raises(TypeError):
        EyeStateSmoother(clock="saat değil")


# --- temel davranış ---------------------------------------------------------


def test_starts_as_not_closed() -> None:
    clock = FakeClock()
    smoother = make_smoother(clock)

    assert smoother.update(valid=True, eyes_closed=True, confidence=0.9) is False


def test_requires_minimum_valid_span_before_reporting_closed() -> None:
    """Sistem başlarken tek bir ölçüm 'kapalı' dedi diye karar
    vermemeli; en az `minimum_valid_seconds` kadar ölçüm birikmeli."""
    clock = FakeClock()
    smoother = make_smoother(clock, minimum_valid_seconds=0.35)

    # 3 ölçüm = 0.25 sn aralık, eşiğin altında.
    assert feed(smoother, clock, closed=True, count=3, step=0.125) is False

    # 4. ölçümle aralık 0.375 sn'ye çıkar.
    assert smoother.update(valid=True, eyes_closed=True, confidence=0.9) is True


def test_majority_closed_within_window_reports_closed() -> None:
    clock = FakeClock()
    smoother = make_smoother(clock)

    assert feed(smoother, clock, closed=True, count=7, step=0.125) is True


def test_open_eyes_majority_reports_not_closed() -> None:
    clock = FakeClock()
    smoother = make_smoother(clock)

    assert feed(smoother, clock, closed=False, count=8, step=0.125) is False


def test_single_bad_frame_does_not_flip_the_decision() -> None:
    """Tek karelik yanlış sınıflandırma kararı devirmemeli — sınıfın
    varlık sebebi bu."""
    clock = FakeClock()
    smoother = make_smoother(clock)

    feed(smoother, clock, closed=True, count=8, step=0.125)

    assert smoother.update(valid=True, eyes_closed=False, confidence=0.9) is True


# --- zaman penceresi (değişikliğin özü) -------------------------------------


def test_window_is_time_based_not_frame_based() -> None:
    """DEĞİŞİKLİĞİN ÖZÜ: aynı SÜREYİ farklı kare hızlarıyla doldurmak
    aynı kararı vermeli. Kare bazlı pencerede 30 FPS'te 7 kare 0.23
    saniyeye, 5 FPS'te 1.4 saniyeye denk geliyordu."""
    for step in (0.033, 0.125, 0.2):  # ~30, 8, 5 FPS
        clock = FakeClock()
        smoother = make_smoother(clock, window_seconds=0.9, minimum_valid_seconds=0.35)

        count = int(0.6 / step) + 1  # 0.6 saniyelik kapalı serisi
        assert feed(smoother, clock, closed=True, count=count, step=step) is True, (
            f"step={step} için karar farklı çıktı"
        )


def test_time_to_decision_does_not_scale_with_frame_rate() -> None:
    """Karara varma SÜRESİ kare hızıyla orantılı kaymamalı. Kare bazlı
    pencerede bu süre 5→30 FPS aralığında 6 kat değişiyordu."""
    sureler = {}
    for step in (0.05, 0.1, 0.25):
        clock = FakeClock()
        smoother = make_smoother(clock, window_seconds=0.9, minimum_valid_seconds=0.35)
        gecen = 0.0
        while gecen <= 3.0:
            if smoother.update(valid=True, eyes_closed=True, confidence=0.9):
                break
            clock.advance(step)
            gecen += step
        sureler[step] = gecen

    # Tek bir örnekleme aralığı kadar fark normal; FPS ile ORANTILI
    # bir kayma olmamalı.
    assert max(sureler.values()) - min(sureler.values()) <= 0.25


def test_old_votes_leave_the_window() -> None:
    """Pencereden düşen oylar karara katılmamalı: yeterince eski bir
    'kapalı' geçmişi, yeni 'açık' serisinin kararı çevirmesini
    engellememeli."""
    clock = FakeClock()
    smoother = make_smoother(clock, window_seconds=0.5, minimum_valid_seconds=0.2)

    assert feed(smoother, clock, closed=True, count=6, step=0.1) is True
    assert feed(smoother, clock, closed=False, count=7, step=0.1) is False


def test_window_keeps_at_least_one_vote_at_very_low_frame_rates() -> None:
    """Kare aralığı pencereden UZUNSA (ör. 0.5 FPS, 0.9 sn pencere) her
    yeni oy tek başına pencereyi aşar; boş pencere sıfıra bölme demek."""
    clock = FakeClock()
    smoother = make_smoother(clock, window_seconds=0.9)

    for _ in range(3):
        smoother.update(valid=True, eyes_closed=True, confidence=0.9)
        clock.advance(2.0)

    assert smoother.valid_vote_count == 1


# --- geçersiz ölçüm toleransı -----------------------------------------------


def test_missing_measurement_within_tolerance_holds_previous_decision() -> None:
    clock = FakeClock()
    smoother = make_smoother(clock, missing_tolerance_seconds=0.25)

    feed(smoother, clock, closed=True, count=8, step=0.125)

    clock.advance(0.1)
    assert smoother.update(valid=False, eyes_closed=False, confidence=0.0) is True


def test_missing_measurement_beyond_tolerance_resets() -> None:
    clock = FakeClock()
    smoother = make_smoother(clock, missing_tolerance_seconds=0.25)

    feed(smoother, clock, closed=True, count=8, step=0.125)

    smoother.update(valid=False, eyes_closed=False, confidence=0.0)
    clock.advance(0.4)

    assert smoother.update(valid=False, eyes_closed=False, confidence=0.0) is False
    assert smoother.valid_vote_count == 0


def test_valid_measurement_clears_the_missing_timer() -> None:
    clock = FakeClock()
    smoother = make_smoother(clock, missing_tolerance_seconds=0.25)

    feed(smoother, clock, closed=True, count=8, step=0.125)

    smoother.update(valid=False, eyes_closed=False, confidence=0.0)
    clock.advance(0.2)
    smoother.update(valid=True, eyes_closed=True, confidence=0.9)
    clock.advance(0.2)

    # Sayaç sıfırlandığı için bu ölçüm hâlâ tolerans içinde.
    assert smoother.update(valid=False, eyes_closed=False, confidence=0.0) is True


# --- eşik ve gözlemlenebilirlik ---------------------------------------------


def test_closed_ratio_threshold_boundary_is_inclusive() -> None:
    clock = FakeClock()
    smoother = make_smoother(
        clock, window_seconds=1.0, minimum_valid_seconds=0.2, closed_ratio_threshold=0.6
    )

    sonuc = False
    for kapali in (True, True, True, False, False):  # 3/5 = 0.6, eşiğe tam eşit
        sonuc = smoother.update(valid=True, eyes_closed=kapali, confidence=0.9)
        clock.advance(0.1)

    assert sonuc is True


def test_last_confidence_tracks_most_recent_valid_vote() -> None:
    clock = FakeClock()
    smoother = make_smoother(clock)

    smoother.update(valid=True, eyes_closed=True, confidence=0.71)
    clock.advance(0.125)
    smoother.update(valid=False, eyes_closed=True, confidence=0.99)  # geçersiz: sayılmaz

    assert smoother.last_confidence == pytest.approx(0.71)


def test_reset_clears_state_and_requires_fresh_span() -> None:
    clock = FakeClock()
    smoother = make_smoother(clock)

    feed(smoother, clock, closed=True, count=8, step=0.125)
    smoother.reset()

    assert smoother.valid_vote_count == 0
    assert smoother.update(valid=True, eyes_closed=True, confidence=0.9) is False
