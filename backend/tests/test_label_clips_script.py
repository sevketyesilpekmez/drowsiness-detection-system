"""
`scripts/label_clips.py`'nin GUI GEREKTİRMEYEN kısımlarının testleri.

Tuş döngüsü ve pencere elle denenmek zorunda, ama diske yazan ve
görüntü üreten fonksiyonlar başsız (headless) sınanabilir. En önemlisi:
BOZUK BİR ETİKET DOSYASININ DİSKE HİÇ YAZILMAMASI.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.label_editor import EditableEvent, LabelEditor

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "label_clips.py"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("label_clips", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def editor():
    return LabelEditor(clip="test_01.mp4", fps=30.0, duration_s=60.0)


# --- kaydetme güvenliği -----------------------------------------------------


def test_save_writes_a_file_the_real_loader_accepts(script, editor, tmp_path):
    from evaluation.ground_truth import load_clip_labels

    editor.begin_event(10.0)
    editor.end_event(15.0)
    target = tmp_path / "test_01.json"

    message = script.save(editor, target)

    assert "Kaydedildi" in message
    assert load_clip_labels(target).expected_event_count == 1


def test_save_refuses_to_write_an_invalid_file(script, editor, tmp_path):
    """EN ÖNEMLİ GÜVENCE: bozuk etiket dosyası diske hiç yazılmamalı.
    Yazılsaydı hata, günler sonra değerlendirme koşusunda ortaya
    çıkardı."""
    editor.events.append(EditableEvent(start_s=10.0, end_s=15.0))
    editor.events.append(EditableEvent(start_s=14.0, end_s=20.0))  # çakışma
    target = tmp_path / "test_01.json"

    message = script.save(editor, target)

    assert "KAYDEDİLMEDİ" in message
    assert not target.exists()


def test_save_does_not_clobber_a_good_file_with_a_bad_one(script, editor, tmp_path):
    """Doğrulama başarısızsa ÖNCEKİ iyi dosya olduğu gibi kalmalı."""
    target = tmp_path / "test_01.json"
    editor.begin_event(10.0)
    editor.end_event(15.0)
    script.save(editor, target)
    good = target.read_text(encoding="utf-8")

    editor.events.append(EditableEvent(start_s=14.0, end_s=20.0))  # çakışma
    script.save(editor, target)

    assert target.read_text(encoding="utf-8") == good


def test_save_creates_missing_directories(script, editor, tmp_path):
    editor.begin_event(10.0)
    editor.end_event(15.0)
    target = tmp_path / "yeni" / "klasor" / "test_01.json"

    script.save(editor, target)

    assert target.exists()


def test_saved_json_is_human_readable(script, editor, tmp_path):
    """Etiket dosyaları elle de gözden geçiriliyor."""
    editor.begin_event(10.0)
    editor.end_event(15.0)
    target = tmp_path / "test_01.json"

    script.save(editor, target)
    text = target.read_text(encoding="utf-8")

    assert text.endswith("\n")
    assert "\n  " in text  # girintili
    assert json.loads(text)["clip"] == "test_01.mp4"


# --- yol mantığı ------------------------------------------------------------


def test_label_path_mirrors_clip_name(script):
    path = script.label_path_for(Path("/x/clips/glasses_01.mp4"), Path("/x/labels"))

    assert path == Path("/x/labels/glasses_01.json")


def test_default_labels_dir_is_sibling_of_clips(script):
    assert script.default_labels_dir(Path("/x/clips/a.mp4")) == Path("/x/labels")


# --- görüntü üretimi --------------------------------------------------------


def test_format_time_is_minutes_and_seconds(script):
    assert script.format_time(125.4) == "02:05.40"
    assert script.format_time(0.0) == "00:00.00"


def test_format_time_clamps_negative(script):
    assert script.format_time(-5.0) == "00:00.00"


def test_render_downscales_large_frames(script, editor):
    frame = np.zeros((1080, 1620, 3), dtype=np.uint8)

    canvas = script.render(
        frame, editor, frame_index=0, time_s=0.0, message="", playing=False
    )

    assert canvas.shape[1] == script.MAX_DISPLAY_WIDTH


def test_render_handles_events_pending_state_and_small_frames(script, editor):
    """Çizim kodu, etiketleme sırasında oluşabilecek her durumu
    patlamadan çizebilmeli — bekleyen aralık, çeldirici, klip sonu."""
    editor.begin_event(10.0)
    editor.end_event(15.0)
    editor.begin_event(30.0)
    editor.end_event(30.3)
    editor.toggle_last_distractor()
    editor.begin_event(59.9)

    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    canvas = script.render(
        frame, editor, frame_index=1797, time_s=59.9, message="uzun bir mesaj", playing=True
    )

    assert canvas.shape[0] == script.HEADER_HEIGHT + 48 + script.TIMELINE_HEIGHT


def test_render_does_not_mutate_the_editor(script, editor):
    editor.begin_event(10.0)
    before = (len(editor.events), editor.pending_start_s)

    script.render(
        np.zeros((48, 64, 3), dtype=np.uint8),
        editor,
        frame_index=0,
        time_s=0.0,
        message="",
        playing=False,
    )

    assert (len(editor.events), editor.pending_start_s) == before
