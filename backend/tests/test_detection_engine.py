import time

import cv2
import numpy as np
import pytest

from app.core.detection_engine import DetectionEngine, REQUIRED_DETECT_FN_KEYS, dummy_detect_fn
from app.core.shared_state import SharedState
from app.core.states import DetectionState
from app.services.video_source import VideoSource, VideoSourceError


class FakeVideoSource:
    """Gerçek cv2.VideoCapture yerine testte kullanılan sahte kaynak.
    video_source.py'nin kendi doğruluğu test_video_source.py'de zaten
    kanıtlandı; burada sadece DetectionEngine'in open/read/release
    arayüzünü doğru kullandığını test ediyoruz. Gerçek VideoSource gibi
    VideoSourceError fırlatır ki DetectionEngine'in `except VideoSourceError`
    dalı testte de gerçekten tetiklensin."""

    def __init__(self, fail_on_open: bool = False, fail_on_read: bool = False):
        self.opened = False
        self.released = False
        self.open_calls = 0
        self.fail_on_open = fail_on_open
        self.fail_on_read = fail_on_read
        self.source = "fake-source"
        self._frame = np.zeros((48, 64, 3), dtype=np.uint8)

    def open(self):
        self.open_calls += 1
        if self.fail_on_open:
            raise VideoSourceError("sahte kamera açılamadı")
        self.opened = True

    def read(self):
        if self.fail_on_read:
            raise VideoSourceError("sahte kamera okuma hatası")
        return self._frame

    def release(self):
        self.released = True


def wait_until(condition, timeout=2.0, interval=0.01) -> bool:
    """Sabit sleep() yerine polling: thread'in gerçekten beklenen duruma
    gelmesini bekler, flaky testlerden kaçınmak için."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(interval)
    return False


@pytest.fixture
def engine_factory():
    """Testte oluşturulan her engine'i, test bitiminde otomatik durdurur.
    Bir assert patlarsa bile arkada thread bırakmamak için gerekli."""
    created = []

    def factory(*, video_source=None, shared_state=None, detect_fn=dummy_detect_fn, **kwargs):
        engine = DetectionEngine(
            video_source=video_source or FakeVideoSource(),
            shared_state=shared_state or SharedState(),
            detect_fn=detect_fn,
            retry_delay=kwargs.pop("retry_delay", 0.02),
            **kwargs,
        )
        created.append(engine)
        return engine

    yield factory

    for engine in created:
        engine.stop()


def test_start_opens_source_and_writes_results(engine_factory):
    video_source = FakeVideoSource()
    shared_state = SharedState()
    engine = engine_factory(video_source=video_source, shared_state=shared_state)

    engine.start()

    assert wait_until(lambda: shared_state.get_result()["state"] == DetectionState.AWAKE)
    assert video_source.opened is True
    assert shared_state.get_frame() is not None


def test_stop_releases_video_source_and_stops_thread(engine_factory):
    video_source = FakeVideoSource()
    engine = engine_factory(video_source=video_source)

    engine.start()
    assert wait_until(lambda: video_source.opened)

    engine.stop()

    assert video_source.released is True
    assert engine.is_running is False


def test_stop_without_start_is_safe():
    video_source = FakeVideoSource()
    engine = DetectionEngine(
        video_source=video_source,
        shared_state=SharedState(),
        detect_fn=dummy_detect_fn,
    )

    engine.stop()  # exception atmamalı

    assert video_source.released is True


def test_double_start_does_not_reopen_source(engine_factory):
    video_source = FakeVideoSource()
    engine = engine_factory(video_source=video_source)

    engine.start()
    engine.start()

    assert video_source.open_calls == 1


def test_open_failure_sets_camera_error_and_does_not_start_thread():
    video_source = FakeVideoSource(fail_on_open=True)
    shared_state = SharedState()
    engine = DetectionEngine(
        video_source=video_source,
        shared_state=shared_state,
        detect_fn=dummy_detect_fn,
    )

    engine.start()

    assert engine.is_running is False
    result = shared_state.get_result()
    assert result["state"] == DetectionState.CAMERA_ERROR
    assert result["error"] is not None


def test_read_failure_sets_camera_error(engine_factory):
    video_source = FakeVideoSource(fail_on_read=True)
    shared_state = SharedState()
    engine = engine_factory(video_source=video_source, shared_state=shared_state)

    engine.start()

    assert wait_until(lambda: shared_state.get_result()["state"] == DetectionState.CAMERA_ERROR)


def test_detect_fn_failure_sets_model_error(engine_factory):
    def failing_detect_fn(frame):
        raise ValueError("model patladı")

    shared_state = SharedState()
    engine = engine_factory(shared_state=shared_state, detect_fn=failing_detect_fn)

    engine.start()

    assert wait_until(lambda: shared_state.get_result()["state"] == DetectionState.MODEL_ERROR)


def test_fps_field_is_populated(engine_factory):
    shared_state = SharedState()
    engine = engine_factory(shared_state=shared_state)

    engine.start()

    assert wait_until(lambda: "fps" in shared_state.get_result())
    assert shared_state.get_result()["fps"] >= 0


def test_camera_read_and_jpeg_encode_timing_fields_are_populated(engine_factory):
    """FPS düşüklüğü araştırmasının bir parçası: engine, backend'den
    (yolo_pytorch/mediapipe) bağımsız olarak camera_read_ms/jpeg_encode_ms
    ölçüp result'a eklemeli — böylece darboğazın kamera okuma mı, JPEG
    encode mu, yoksa detect_fn'in kendisi mi olduğu ayırt edilebilir."""
    shared_state = SharedState()
    engine = engine_factory(shared_state=shared_state)

    engine.start()

    assert wait_until(lambda: "camera_read_ms" in shared_state.get_result())
    result = shared_state.get_result()
    assert result["camera_read_ms"] >= 0.0
    assert result["jpeg_encode_ms"] >= 0.0


def test_average_fps_uses_rolling_window_of_frame_durations(engine_factory):
    """_record_frame_and_get_average_fps, tek bir karenin anlık 1/elapsed'i
    yerine son N karenin süresinin ortalamasını kullanmalı. Burada thread
    hiç başlatılmıyor — metod doğrudan, deterministik elapsed değerleriyle
    çağrılıyor (flaky timing riski yok)."""
    engine = engine_factory()

    # 10 kare, her biri 0.1s sürmüş varsayalım -> ortalama süre 0.1s ->
    # ortalama fps = 1/0.1 = 10.0.
    fps = 0.0
    for _ in range(10):
        fps = engine._record_frame_and_get_average_fps(0.1)

    assert fps == 10.0


def test_average_fps_is_not_skewed_by_one_slow_frame(engine_factory):
    """Mentor'un gözlemlediği türden bir senaryo: 9 hızlı kare (0.05s,
    yani anlık 20 fps) arasına TEK bir yavaş kare (1.0s, yani anlık 1 fps)
    girerse, ortalama fps'in o tek kareye göre absürt düşmemesi lazım —
    süreler ortalanıp SONRA tersi alınıyor, yani gerçek toplam verim
    yansıtılıyor (yaklaşık 10 kare / (9*0.05 + 1.0) saniye ≈ 6.9 fps),
    aritmetik ortalama fps yöntemiyle çıkacak daha düşük bir sayı değil."""
    engine = engine_factory()

    fps = 0.0
    for _ in range(9):
        fps = engine._record_frame_and_get_average_fps(0.05)
    fps = engine._record_frame_and_get_average_fps(1.0)

    total_elapsed = 9 * 0.05 + 1.0
    expected_fps = round(10 / total_elapsed, 1)
    assert fps == expected_fps
    assert fps > 5.0  # aritmetik ortalama fps yöntemiyle bu çok daha düşük çıkardı


def test_average_fps_window_is_bounded(engine_factory):
    """FPS_AVERAGE_WINDOW'dan (30) fazla kare geçtiğinde en eski süreler
    deque'den düşmeli — sonsuza kadar büyüyen bir pencere olmamalı."""
    engine = engine_factory()

    for _ in range(100):
        engine._record_frame_and_get_average_fps(0.1)

    assert len(engine._frame_durations) == 30


def test_average_fps_resets_on_restart(engine_factory):
    """start() tekrar çağrıldığında (aynı örnek stop() sonrası yeniden
    başlatılırsa) önceki oturumdan kalan süreler yeni ortalamayı
    bozmamalı. start() thread'i hemen canlandırıp gerçek (küçük) süreler
    eklemeye başladığı için uzunluğun tam 0 olmasını bekleyemeyiz (yarış
    durumu) — bunun yerine eski oturumun yavaş (1.0s) kayıtlarının
    KALMADIĞINI doğruluyoruz."""
    engine = engine_factory()

    for _ in range(5):
        engine._record_frame_and_get_average_fps(1.0)  # yavaş "eski oturum"

    assert len(engine._frame_durations) == 5

    engine.start()
    assert wait_until(lambda: all(duration < 0.5 for duration in engine._frame_durations))


def test_engine_recovers_after_transient_read_error(engine_factory):
    """Bir okuma hatasından sonra kaynak tekrar çalışırsa (fail_on_read
    False'a döndürülürse) engine kendiliğinden düzelip AWAKE'e dönebilmeli
    — thread'i yeniden başlatmaya gerek yok."""
    video_source = FakeVideoSource(fail_on_read=True)
    shared_state = SharedState()
    engine = engine_factory(video_source=video_source, shared_state=shared_state)

    engine.start()
    assert wait_until(lambda: shared_state.get_result()["state"] == DetectionState.CAMERA_ERROR)

    video_source.fail_on_read = False

    assert wait_until(lambda: shared_state.get_result()["state"] == DetectionState.AWAKE)


def test_engine_works_with_real_video_source(tmp_path, engine_factory):
    """FakeVideoSource'un yerini gerçek VideoSource'un aldığı tek entegrasyon
    testi — video_source.py'nin open/read/release arayüzü değişince
    DetectionEngine'in hâlâ doğru çalıştığını kanıtlar (mock'ların
    yakalayamayacağı arayüz uyumsuzluklarına karşı). Ayrıca _draw_overlay'in
    gerçek bir frame üzerinde patlamadığını da dolaylı olarak doğrular."""
    video_path = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (64, 48),
    )
    for i in range(5):
        writer.write(np.full((48, 64, 3), i * 40, dtype=np.uint8))
    writer.release()

    real_video_source = VideoSource(str(video_path))
    shared_state = SharedState()
    engine = engine_factory(video_source=real_video_source, shared_state=shared_state)

    engine.start()

    assert wait_until(lambda: shared_state.get_result()["state"] == DetectionState.AWAKE)
    frame_bytes = shared_state.get_frame()
    assert frame_bytes is not None
    assert frame_bytes[:2] == b"\xff\xd8"  # JPEG magic bytes

    engine.stop()
    assert real_video_source.is_opened is False


def test_invalid_jpeg_quality_raises():
    with pytest.raises(ValueError):
        DetectionEngine(
            video_source=FakeVideoSource(),
            shared_state=SharedState(),
            detect_fn=dummy_detect_fn,
            jpeg_quality=101,
        )


def test_invalid_retry_delay_raises():
    with pytest.raises(ValueError):
        DetectionEngine(
            video_source=FakeVideoSource(),
            shared_state=SharedState(),
            detect_fn=dummy_detect_fn,
            retry_delay=-1,
        )


def test_stop_is_responsive_during_error_backoff(engine_factory):
    """threading.Event.wait() kullanıldığı için stop(), hata sonrası uzun
    bir bekleme sırasında bile ANINDA dönmeli — time.sleep() olsaydı tüm
    retry_delay süresini beklemek zorunda kalırdı."""
    video_source = FakeVideoSource(fail_on_read=True)
    shared_state = SharedState()
    engine = engine_factory(
        video_source=video_source,
        shared_state=shared_state,
        retry_delay=5.0,  # kasıtlı olarak uzun
    )

    engine.start()
    assert wait_until(lambda: shared_state.get_result()["state"] == DetectionState.CAMERA_ERROR)

    stopped_at = time.monotonic()
    engine.stop(timeout=2.0)
    elapsed = time.monotonic() - stopped_at

    assert elapsed < 1.0  # 5 saniyelik retry_delay'i beklememeli


def test_detect_fn_missing_required_key_sets_model_error(engine_factory):
    """detect_fn sözleşmeyi bozup (örn. confidence'ı unutup) eksik bir
    dict döndürürse, bu bozuk veri sessizce shared_state'e sızmamalı —
    MODEL_ERROR olarak yakalanmalı."""

    def incomplete_detect_fn(frame):
        result = {
            "state": DetectionState.AWAKE,
            "is_drowsy": False,
            "eyes_closed": False,
            # "confidence" ve "closed_duration" kasıtlı olarak eksik
        }
        return result

    shared_state = SharedState()
    engine = engine_factory(shared_state=shared_state, detect_fn=incomplete_detect_fn)

    engine.start()

    assert wait_until(lambda: shared_state.get_result()["state"] == DetectionState.MODEL_ERROR)
    assert "confidence" in shared_state.get_result()["error"]


def test_detect_fn_with_extra_keys_is_allowed(engine_factory):
    """Sözleşme bir MİNİMUM'dur — detect_fn ekstra alanlar (örn. bbox)
    döndürürse bunlar reddedilmemeli, olduğu gibi shared_state'e geçmeli."""

    def detect_fn_with_extras(frame):
        result = dict(dummy_detect_fn(frame))
        result["bbox"] = [10, 10, 50, 50]
        return result

    shared_state = SharedState()
    engine = engine_factory(shared_state=shared_state, detect_fn=detect_fn_with_extras)

    engine.start()

    assert wait_until(lambda: shared_state.get_result().get("bbox") == [10, 10, 50, 50])


def test_engine_always_overwrites_fps_from_detect_fn(engine_factory):
    """fps, detect_fn'in değil engine'in sorumluluğunda — detect_fn yanlışlıkla
    fps döndürse bile engine'in kendi hesapladığı değer geçerli olmalı."""

    def detect_fn_with_bogus_fps(frame):
        result = dict(dummy_detect_fn(frame))
        result["fps"] = 999.0
        return result

    shared_state = SharedState()
    engine = engine_factory(shared_state=shared_state, detect_fn=detect_fn_with_bogus_fps)

    engine.start()

    assert wait_until(lambda: shared_state.get_result().get("fps", 999.0) != 999.0)


class MockDetector:
    """DrowsinessDetector'ın gerçek MediaPipe/DrowsinessService'e ihtiyaç
    duymadan sahte bir sürümü — burada yalnızca DetectionEngine.stop()'un
    detect_fn.close()'u gerçekten çağırdığını (getattr + callable kontrolü
    üzerinden) doğrulamak için kullanılıyor."""

    def __init__(self, *, raise_on_close: bool = False):
        self.close_called = False
        self.raise_on_close = raise_on_close

    def __call__(self, frame):
        return dict(dummy_detect_fn(frame))

    def close(self):
        self.close_called = True
        if self.raise_on_close:
            raise RuntimeError("close() patladı")


def test_stop_closes_detect_fn_if_closeable(engine_factory):
    """DetectionEngine.stop() tek başına detect_fn'in kaynaklarını (örn.
    DrowsinessDetector -> LandmarkService -> MediaPipe FaceMesh) serbest
    bırakabilmeli — create_app()'teki atexit'e bağımlı kalmamalı, çünkü
    engine.stop() süreç sonlanmadan da (örn. testlerde, ya da ileride bir
    "yeniden başlat" özelliğinde) çağrılabilir olmalı."""
    detector = MockDetector()
    engine = engine_factory(detect_fn=detector)

    engine.start()
    assert wait_until(lambda: engine.is_running)

    engine.stop()

    assert detector.close_called is True


def test_stop_is_safe_when_detect_fn_has_no_close(engine_factory):
    """dummy_detect_fn gibi sade fonksiyonların close() metodu yok —
    getattr(None) + callable kontrolü sayesinde stop() bunda patlamamalı
    (bu proje boyunca zaten dummy_detect_fn ile onlarca test stop()
    çağırıyor, ama niyeti açıkça belgelemek için ayrı bir test)."""
    engine = engine_factory(detect_fn=dummy_detect_fn)

    engine.start()
    engine.stop()  # exception atmamalı


def test_stop_swallows_detect_fn_close_exception(engine_factory):
    """detect_fn.close() patlarsa (örn. MediaPipe kaynak serbest bırakma
    hatası), bu stop()'un geri kalanını (thread/video_source temizliği
    zaten tamamlanmış) veya çağıranı ETKİLEMEMELİ — loglanıp yutulmalı."""
    detector = MockDetector(raise_on_close=True)
    engine = engine_factory(detect_fn=detector)

    engine.start()
    assert wait_until(lambda: engine.is_running)

    engine.stop()  # exception atmamalı

    assert detector.close_called is True


def test_dummy_detect_fn_satisfies_contract():
    """dummy_detect_fn'in kendisi de REQUIRED_DETECT_FN_KEYS'i tam
    karşılamalı — şablon olarak referans gösterildiği için bu garanti
    olmadan diğer her şey anlamsızlaşır."""
    result = dummy_detect_fn(np.zeros((4, 4, 3), dtype=np.uint8))

    assert REQUIRED_DETECT_FN_KEYS <= result.keys()
