from __future__ import annotations

import json

import pytest

from evaluation.ground_truth import parse_clip_labels
from evaluation.label_editor import EditableEvent, LabelEditor


@pytest.fixture
def editor():
    return LabelEditor(clip="test_01.mp4", fps=30.0, duration_s=60.0)


# --- aralık işaretleme ------------------------------------------------------


def test_begin_then_end_creates_an_event(editor):
    editor.begin_event(10.0)
    editor.end_event(15.0)

    assert len(editor.events) == 1
    assert editor.events[0].start_s == 10.0
    assert editor.events[0].end_s == 15.0
    assert editor.events[0].expect_alarm is True
    assert editor.has_pending is False


def test_second_begin_updates_the_start_instead_of_failing(editor):
    """Etiketleme sırasında en sık yapılan düzeltme: 'yanlış yerde
    başlattım, geri sarıp tekrar basayım'. Hata vermek yerine
    başlangıcı güncellemeli."""
    editor.begin_event(10.0)
    message = editor.begin_event(12.0)

    assert editor.pending_start_s == 12.0
    assert "güncellendi" in message


def test_end_without_begin_is_reported_not_raised(editor):
    """Yanlış tuşa basmak normaldir; program çökmemeli."""
    message = editor.end_event(10.0)

    assert editor.events == []
    assert "başlangıç" in message.lower()


def test_end_before_start_is_rejected(editor):
    editor.begin_event(10.0)
    message = editor.end_event(8.0)

    assert editor.events == []
    assert editor.has_pending is True  # bekleyen korunur, kullanıcı düzeltebilir
    assert "sonra olmalı" in message


def test_end_beyond_clip_duration_is_rejected(editor):
    editor.begin_event(55.0)
    message = editor.end_event(70.0)

    assert editor.events == []
    assert "klip süresini" in message


def test_negative_time_is_rejected(editor):
    message = editor.begin_event(-1.0)

    assert editor.has_pending is False
    assert "negatif" in message


# --- çakışma ----------------------------------------------------------------


def test_overlapping_expected_event_is_rejected(editor):
    """Çakışan olaylar bir alarmın hangi olaya ait olduğunu belirsiz
    kılar; dosyaya yazılmadan önce engellenmeli."""
    editor.begin_event(10.0)
    editor.end_event(15.0)

    editor.begin_event(14.0)
    message = editor.end_event(20.0)

    assert len(editor.events) == 1
    assert "çakışıyor" in message


def test_adjacent_events_are_allowed(editor):
    """Bitişik (uç uca) olaylar çakışma değildir."""
    editor.begin_event(10.0)
    editor.end_event(15.0)
    editor.begin_event(15.0)
    editor.end_event(20.0)

    assert len(editor.events) == 2


def test_distractor_may_overlap_a_real_event(editor):
    """Uzun bir kapanma içinde göz kırpması işaretlemek anlamlıdır."""
    editor.begin_event(10.0)
    editor.end_event(20.0)
    editor.toggle_last_distractor()  # ilkini çeldirici yap

    editor.begin_event(12.0)
    message = editor.end_event(18.0)

    assert len(editor.events) == 2
    assert "çakışıyor" not in message


# --- çeldirici --------------------------------------------------------------


def test_toggle_marks_last_event_as_distractor(editor):
    editor.begin_event(10.0)
    editor.end_event(10.4)
    editor.toggle_last_distractor()

    assert editor.events[-1].expect_alarm is False
    assert editor.events[-1].label == "blink"
    assert editor.expected_event_count == 0
    assert editor.distractor_count == 1


def test_toggle_is_reversible(editor):
    editor.begin_event(10.0)
    editor.end_event(15.0)
    editor.toggle_last_distractor()
    editor.toggle_last_distractor()

    assert editor.events[-1].expect_alarm is True
    assert editor.events[-1].label == "eyes_closed"


def test_toggle_back_is_refused_when_it_would_create_an_overlap(editor):
    """Değişmez kural: alarm beklenen iki olay çakışamaz. Çeldiriciyi
    gerçek olaya çevirmek bu kuralı bozabilir.

    Durum doğrudan kuruluyor çünkü etkileşimli akıştan bu duruma
    ULAŞILAMIYOR: çakışan bir olay zaten `end_event` tarafından
    reddediliyor ve `toggle_last_distractor` yalnızca SON olayı
    değiştiriyor. Koruma yine de duruyor — kural gerçek ve ileride
    "istediğin olayı seç" özelliği eklenirse tek savunma bu olacak."""
    editor.events.append(EditableEvent(start_s=12.0, end_s=14.0, expect_alarm=True))
    editor.events.append(
        EditableEvent(start_s=10.0, end_s=20.0, expect_alarm=False, label="blink")
    )

    message = editor.toggle_last_distractor()

    assert editor.events[-1].expect_alarm is False
    assert "çevrilemez" in message


def test_toggle_with_no_events_is_reported(editor):
    assert "olay yok" in editor.toggle_last_distractor()


# --- geri alma --------------------------------------------------------------


def test_undo_cancels_pending_start_first(editor):
    """Kullanıcının en son yaptığı iş bekleyen başlangıçtır;
    tamamlanmış bir olayı silmek beklenmedik olurdu."""
    editor.begin_event(10.0)
    editor.end_event(15.0)
    editor.begin_event(20.0)

    editor.undo()

    assert editor.has_pending is False
    assert len(editor.events) == 1


def test_undo_removes_last_event_when_nothing_pending(editor):
    editor.begin_event(10.0)
    editor.end_event(15.0)

    editor.undo()

    assert editor.events == []


def test_undo_on_empty_editor_is_reported(editor):
    assert "geri alınacak" in editor.undo().lower()


def test_clear_removes_everything(editor):
    editor.begin_event(10.0)
    editor.end_event(15.0)
    editor.begin_event(20.0)

    editor.clear()

    assert editor.events == []
    assert editor.has_pending is False


# --- JSON çıktısı -----------------------------------------------------------


def test_output_is_accepted_by_the_real_loader(editor):
    """ARACIN EN ÖNEMLİ GÜVENCESİ: üretilen dosya, değerlendirme
    koşusunun kullanacağı doğrulamadan birebir geçmeli. Yoksa
    'etiketledim ama koşu patlıyor' durumu ortaya çıkar."""
    editor.begin_event(10.0)
    editor.end_event(15.0)
    editor.begin_event(20.0)
    editor.end_event(20.3)
    editor.toggle_last_distractor()

    labels = parse_clip_labels(editor.to_dict())

    assert labels.clip == "test_01.mp4"
    assert len(labels.events) == 2
    assert labels.expected_event_count == 1


def test_validate_returns_none_when_clean(editor):
    editor.begin_event(10.0)
    editor.end_event(15.0)

    assert editor.validate() is None


def test_output_is_json_serialisable(editor):
    editor.begin_event(10.0)
    editor.end_event(15.0)

    data = editor.to_dict()

    assert json.loads(json.dumps(data)) == data


def test_events_are_sorted_by_time_in_output(editor):
    """Kullanıcı sondan başa doğru etiketleyebilir; dosya yine de
    okunabilir sırada olmalı."""
    editor.begin_event(30.0)
    editor.end_event(35.0)
    editor.begin_event(10.0)
    editor.end_event(15.0)

    starts = [e["start_s"] for e in editor.to_dict()["events"]]

    assert starts == [10.0, 30.0]


def test_default_fields_are_omitted_for_readability(editor):
    """Etiket dosyaları elle de okunuyor; expect_alarm=True gibi
    varsayılanlar gürültü yapar."""
    editor.begin_event(10.0)
    editor.end_event(15.0)

    event = editor.to_dict()["events"][0]

    assert "expect_alarm" not in event
    assert "label" not in event


def test_distractor_fields_are_written(editor):
    editor.begin_event(10.0)
    editor.end_event(10.4)
    editor.toggle_last_distractor()

    event = editor.to_dict()["events"][0]

    assert event["expect_alarm"] is False
    assert event["label"] == "blink"


def test_optional_metadata_is_included_when_set():
    editor = LabelEditor(
        clip="a.mp4",
        fps=30.0,
        duration_s=60.0,
        subject="S03",
        conditions=("glasses", "lowlight"),
        notes="yandan ışık",
    )

    data = editor.to_dict()

    assert data["subject"] == "S03"
    assert data["conditions"] == ["glasses", "lowlight"]
    assert data["notes"] == "yandan ışık"


# --- uyarılar ---------------------------------------------------------------


def test_warns_about_events_shorter_than_threshold(editor):
    editor.begin_event(10.0)
    editor.end_event(11.0)

    warnings = editor.warnings(drowsiness_seconds=2.5)

    assert any("eşiğinden kısa" in w for w in warnings)


def test_warns_when_there_are_no_distractors(editor):
    editor.begin_event(10.0)
    editor.end_event(15.0)

    warnings = editor.warnings(drowsiness_seconds=2.5)

    assert any("çeldirici yok" in w for w in warnings)


def test_warns_when_there_are_no_real_events(editor):
    warnings = editor.warnings(drowsiness_seconds=2.5)

    assert any("gerçek olay yok" in w for w in warnings)


def test_no_warnings_for_a_well_formed_clip(editor):
    editor.begin_event(10.0)
    editor.end_event(15.0)
    editor.begin_event(30.0)
    editor.end_event(30.3)
    editor.toggle_last_distractor()

    assert editor.warnings(drowsiness_seconds=2.5) == []


def test_status_line_reflects_pending_state(editor):
    assert "hazır" in editor.status_line()

    editor.begin_event(10.0)

    assert "kaydediliyor" in editor.status_line()
