import cv2
import numpy as np
import pytest

from app.services import video_source as video_source_module
from app.services.video_source import VideoSource, VideoSourceError

FRAME_COUNT = 5
FRAME_SIZE = (64, 48)  # (width, height)


@pytest.fixture
def tiny_video_path(tmp_path):
    """5 karelik minik bir mp4 üretir (gerçek kamera gerektirmeden test için)."""
    path = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        FRAME_SIZE,
    )

    for i in range(FRAME_COUNT):
        frame = np.full((FRAME_SIZE[1], FRAME_SIZE[0], 3), i * 40, dtype=np.uint8)
        writer.write(frame)

    writer.release()

    return str(path)


def test_parse_source_numeric_string_becomes_camera_index():
    source = VideoSource("0")
    assert source.source == 0
    assert source.is_camera is True


def test_parse_source_int_stays_camera_index():
    source = VideoSource(0)
    assert source.source == 0
    assert source.is_camera is True


def test_parse_source_file_path_stays_string():
    source = VideoSource("sample_data/drowsy/test.mp4")
    assert source.source == "sample_data/drowsy/test.mp4"
    assert source.is_camera is False


def test_read_before_open_raises():
    source = VideoSource("sample_data/drowsy/test.mp4")

    with pytest.raises(VideoSourceError):
        source.read()


def test_open_missing_file_raises():
    source = VideoSource("sample_data/does_not_exist.mp4")

    with pytest.raises(VideoSourceError):
        source.open()


def test_open_and_read_all_frames(tiny_video_path):
    source = VideoSource(tiny_video_path)
    source.open()

    try:
        frames = [source.read() for _ in range(FRAME_COUNT)]
        assert len(frames) == FRAME_COUNT
        assert all(frame is not None for frame in frames)
    finally:
        source.release()


def test_file_source_loops_after_last_frame(tiny_video_path):
    source = VideoSource(tiny_video_path)
    source.open()

    try:
        for _ in range(FRAME_COUNT):
            source.read()

        # FRAME_COUNT. karede video bitti; bir sonraki read() başa sarıp
        # tekrar 1. kareyi döndürmeli, hata fırlatmamalı.
        looped_frame = source.read()
        assert looped_frame is not None
    finally:
        source.release()


def test_release_is_safe_to_call_twice(tiny_video_path):
    source = VideoSource(tiny_video_path)
    source.open()
    source.release()
    source.release()  # ikinci çağrı patlamamalı

    assert source.is_opened is False


def test_open_is_idempotent(tiny_video_path):
    """Art arda open() çağırmak önceki handle'ı sızdırmamalı/yeniden
    açmaya çalışmamalı — ikinci çağrı no-op olmalı."""
    source = VideoSource(tiny_video_path)

    try:
        source.open()
        first_capture = source._capture
        source.open()
        second_capture = source._capture

        assert first_capture is second_capture
        assert source.is_opened is True
    finally:
        source.release()


def test_get_properties_before_open_returns_zeros():
    source = VideoSource("sample_data/drowsy/test.mp4")
    props = source.get_properties()

    assert props["is_opened"] is False
    assert props["width"] == 0
    assert props["height"] == 0
    assert props["frame_count"] == 0


def test_get_properties_after_open_reflects_real_video(tiny_video_path):
    source = VideoSource(tiny_video_path)
    source.open()

    try:
        props = source.get_properties()
        assert props["is_opened"] is True
        assert props["is_camera"] is False
        assert props["width"] == FRAME_SIZE[0]
        assert props["height"] == FRAME_SIZE[1]
        assert props["frame_count"] >= FRAME_COUNT
    finally:
        source.release()


def test_context_manager_opens_and_releases(tiny_video_path):
    with VideoSource(tiny_video_path) as source:
        assert source.is_opened is True
        frame = source.read()
        assert frame is not None

    assert source.is_opened is False


class _FakeCapture:
    """cv2.VideoCapture'ın gerçek bir kamera olmadan test edilebilmesi için
    minik sahte sürümü — yalnızca bu testlerin ihtiyaç duyduğu metodları
    (isOpened/set/get/read/release) taklit eder."""

    def __init__(self, source):
        self.source = source
        self._opened = True
        self.set_calls = []

    def isOpened(self):
        return self._opened

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        return True

    def get(self, prop):
        return 0.0

    def read(self):
        return True, np.zeros((10, 10, 3), dtype=np.uint8)

    def release(self):
        self._opened = False


def test_requested_resolution_is_applied_for_camera_source(monkeypatch):
    """GERÇEK BUG'IN REGRESYON TESTİ: CAMERA_WIDTH/CAMERA_HEIGHT önceden
    Config'de tanımlıydı ama hiçbir yerde kameraya uygulanmıyordu. Şimdi
    open(), yalnızca kamera kaynakları için, CAP_PROP_FRAME_WIDTH/HEIGHT'ı
    set() ile uygulamayı denemeli."""
    captured = {}

    def fake_video_capture(source):
        capture = _FakeCapture(source)
        captured["capture"] = capture
        return capture

    monkeypatch.setattr(video_source_module.cv2, "VideoCapture", fake_video_capture)

    source = VideoSource(0, requested_width=1280, requested_height=720)
    source.open()

    assert captured["capture"].set_calls == [
        (cv2.CAP_PROP_FRAME_WIDTH, 1280),
        (cv2.CAP_PROP_FRAME_HEIGHT, 720),
    ]


def test_no_requested_resolution_means_no_set_calls(monkeypatch):
    captured = {}

    def fake_video_capture(source):
        capture = _FakeCapture(source)
        captured["capture"] = capture
        return capture

    monkeypatch.setattr(video_source_module.cv2, "VideoCapture", fake_video_capture)

    source = VideoSource(0)
    source.open()

    assert captured["capture"].set_calls == []


def test_requested_resolution_is_ignored_for_file_source(tiny_video_path):
    """Bir video dosyasının "istenen" bir çözünürlüğü olmaz (zaten sabit
    kodlanmış) — requested_width/height verilse bile dosya kaynakları için
    hiç set() çağrılmamalı. Burada gerçek cv2.VideoCapture kullanılıyor
    (tiny_video_path fixture'ı), sahte capture'a gerek yok: dosya
    kaynağında is_camera False olduğu için set() hiç tetiklenmeyecek ve
    dosya normal şekilde okunabilir olmaya devam edecek."""
    source = VideoSource(tiny_video_path, requested_width=1280, requested_height=720)

    try:
        source.open()
        assert source.is_opened is True
        frame = source.read()
        assert frame is not None
    finally:
        source.release()
