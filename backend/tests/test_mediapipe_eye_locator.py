from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.services.eye_region_locator import EyeBox
from app.services.landmark_service import LEFT_EYE_INDICES, RIGHT_EYE_INDICES
from app.services.mediapipe_eye_locator import MediaPipeEyeLocator
from app.services.person_detector import PersonBox

FRAME_WIDTH = 640
FRAME_HEIGHT = 480


class FakeFaceMesh:
    """test_landmark_service.py'deki ile birebir aynı sahte FaceMesh —
    MediaPipeEyeLocator da aynı FaceMeshLike kontratını kullanıyor."""

    def __init__(self, *, landmarks=None) -> None:
        self.closed = False
        self.received_image = None

        if landmarks is None:
            self.output = SimpleNamespace(multi_face_landmarks=None)
        else:
            self.output = SimpleNamespace(multi_face_landmarks=[SimpleNamespace(landmark=landmarks)])

    def process(self, image):
        self.received_image = image
        return self.output

    def close(self) -> None:
        self.closed = True


def make_frame(width=FRAME_WIDTH, height=FRAME_HEIGHT) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def make_landmarks(*, num_landmarks=478) -> list[SimpleNamespace]:
    """Tüm noktaları merkeze (0.5, 0.5) koyan bir iskelet üretir — testler
    yalnızca ihtiyaç duydukları göz indekslerini ayrıca override eder."""
    return [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(num_landmarks)]


def set_eye_points(landmarks, indices, *, x_range, y_range):
    """Verilen indekslere, x_range/y_range sınırlarını (piksel) tam olarak
    kapsayan 6 nokta atar — böylece beklenen bounding box önceden
    hesaplanabilir. Basitlik için ilk iki nokta min/max köşeleri, kalan
    dördü aradaki bir noktayı tekrarlar (yalnızca min/max'a bakan
    bounding-box mantığı için yeterli)."""
    min_x, max_x = x_range
    min_y, max_y = y_range
    pixel_points = (
        (min_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
        (max_x, min_y),
        ((min_x + max_x) / 2, min_y),
        ((min_x + max_x) / 2, max_y),
    )
    for index, (x, y) in zip(indices, pixel_points, strict=True):
        landmarks[index] = SimpleNamespace(x=x / FRAME_WIDTH, y=y / FRAME_HEIGHT, z=0.0)


def test_locate_returns_empty_tuple_when_no_face():
    locator = MediaPipeEyeLocator(face_mesh=FakeFaceMesh(landmarks=None))

    boxes = locator.locate(make_frame())

    assert boxes == ()


def test_locate_returns_square_padded_boxes_for_both_eyes():
    """Crop KARE olmalı (bkz. _landmark_bounding_box'taki gerçek bug notu
    — yassı bir dikdörtgen, TorchEyeClassifier'ın sabit kare resize'ında
    gerilip bozulurdu, kapalı gözler açık sanılıyordu)."""
    landmarks = make_landmarks()
    set_eye_points(landmarks, LEFT_EYE_INDICES, x_range=(100, 140), y_range=(90, 110))
    set_eye_points(landmarks, RIGHT_EYE_INDICES, x_range=(400, 440), y_range=(90, 110))
    locator = MediaPipeEyeLocator(face_mesh=FakeFaceMesh(landmarks=landmarks), eye_padding_ratio=0.5)

    boxes = locator.locate(make_frame())

    # Sol göz: x[100,140] (genişlik 40), y[90,110] (yükseklik 20), merkez (120,100).
    # side = max(40,20) * (1+2*0.5) = 40*2 = 80 -> x[80,160], y[60,140] (80x80 KARE).
    # Sağ göz: merkez (420,100), aynı side=80 -> x[380,460], y[60,140].
    assert boxes == (
        EyeBox(x1=80, y1=60, x2=160, y2=140),
        EyeBox(x1=380, y1=60, x2=460, y2=140),
    )
    left_box = boxes[0]
    assert (left_box.x2 - left_box.x1) == (left_box.y2 - left_box.y1)  # kare


def test_locate_with_zero_padding_still_returns_square_box():
    """padding=0 olsa bile crop hâlâ KARE'dir — yalnızca ekstra büyütme
    (padding) sıfırlanır, kare-zorlama mantığı (max(genişlik,yükseklik))
    devam eder."""
    landmarks = make_landmarks()
    set_eye_points(landmarks, LEFT_EYE_INDICES, x_range=(100, 140), y_range=(90, 110))
    set_eye_points(landmarks, RIGHT_EYE_INDICES, x_range=(400, 440), y_range=(90, 110))
    locator = MediaPipeEyeLocator(face_mesh=FakeFaceMesh(landmarks=landmarks), eye_padding_ratio=0.0)

    boxes = locator.locate(make_frame())

    # side = max(40,20) = 40, merkez (120,100) -> x[100,140], y[80,120].
    assert boxes[0] == EyeBox(x1=100, y1=80, x2=140, y2=120)


def test_locate_square_box_stays_anchored_to_width_when_eyes_close():
    """Göz KAPANDIĞINDA yükseklik küçülür ama crop'un kare kenarı hâlâ
    GENİŞLİĞE göre belirlenmeli (yüksekliğe göre değil) — aksi halde
    kapalı bir göz için çok küçük/anlamsız bir kare üretilirdi."""
    landmarks = make_landmarks()
    # Açıkken y[90,110] (yükseklik 20) idi; kapanınca yükseklik ~2'ye
    # düşüyor (neredeyse çizgi), genişlik (40) aynı kalıyor.
    set_eye_points(landmarks, LEFT_EYE_INDICES, x_range=(100, 140), y_range=(99, 101))
    locator = MediaPipeEyeLocator(face_mesh=FakeFaceMesh(landmarks=landmarks), eye_padding_ratio=0.0)

    boxes = locator.locate(make_frame())

    # side = max(40, 2) = 40 (yüksekliğe göre değil genişliğe göre) ->
    # merkez (120,100) -> x[100,140], y[80,120] -- hâlâ 40x40 kare.
    left_box = boxes[0]
    assert (left_box.x2 - left_box.x1) == 40
    assert (left_box.y2 - left_box.y1) == 40


def test_locate_clips_padded_box_to_frame_bounds():
    """Göz frame'in kenarına çok yakınsa, padding frame dışına taşabilir
    — sonuç kutusu frame sınırlarına kırpılmalı (negatif/aşırı koordinat
    olmamalı)."""
    landmarks = make_landmarks()
    set_eye_points(landmarks, LEFT_EYE_INDICES, x_range=(2, 10), y_range=(2, 10))
    set_eye_points(landmarks, RIGHT_EYE_INDICES, x_range=(400, 440), y_range=(90, 110))
    locator = MediaPipeEyeLocator(face_mesh=FakeFaceMesh(landmarks=landmarks), eye_padding_ratio=5.0)

    boxes = locator.locate(make_frame())

    left_box = boxes[0]
    assert left_box.x1 == 0
    assert left_box.y1 == 0


def test_locate_ignores_search_region():
    """EyeRegionLocator'ın aksine, search_region frame'i kırpmaz —
    face_mesh'e her zaman TAM frame gönderilir."""
    fake = FakeFaceMesh(landmarks=None)
    locator = MediaPipeEyeLocator(face_mesh=fake)
    frame = make_frame()
    region = PersonBox(x1=50, y1=50, x2=200, y2=200, confidence=0.9)

    locator.locate(frame, search_region=region)

    assert fake.received_image.shape[:2] == (FRAME_HEIGHT, FRAME_WIDTH)


def test_locate_skips_eye_with_out_of_range_landmark_index():
    """MediaPipe beklenmedik şekilde eksik/kısa bir landmark listesi
    döndürürse (gerçek hayatta olmamalı ama savunmacı programlama), o göz
    None döner ve atlanır — tüm frame MODEL_ERROR'a düşmez."""
    short_landmarks = make_landmarks(num_landmarks=200)  # RIGHT_EYE_INDICES (362+) bunun dışında kalır
    set_eye_points(short_landmarks, LEFT_EYE_INDICES, x_range=(10, 20), y_range=(10, 20))
    locator = MediaPipeEyeLocator(face_mesh=FakeFaceMesh(landmarks=short_landmarks), eye_padding_ratio=0.0)

    boxes = locator.locate(make_frame())

    assert len(boxes) == 1
    assert boxes[0] == EyeBox(x1=10, y1=10, x2=20, y2=20)


def test_rejects_invalid_eye_padding_ratio():
    with pytest.raises(ValueError):
        MediaPipeEyeLocator(face_mesh=FakeFaceMesh(), eye_padding_ratio=-0.1)


def test_rejects_invalid_max_num_faces():
    with pytest.raises(ValueError):
        MediaPipeEyeLocator(face_mesh=FakeFaceMesh(), max_num_faces=0)


def test_close_does_not_close_injected_face_mesh():
    fake = FakeFaceMesh(landmarks=None)
    locator = MediaPipeEyeLocator(face_mesh=fake)

    locator.close()

    assert fake.closed is False


def test_close_is_idempotent():
    locator = MediaPipeEyeLocator(face_mesh=FakeFaceMesh(landmarks=None))

    locator.close()
    locator.close()  # ikinci çağrı patlamamalı


def test_locate_raises_after_close():
    locator = MediaPipeEyeLocator(face_mesh=FakeFaceMesh(landmarks=None))
    locator.close()

    with pytest.raises(RuntimeError):
        locator.locate(make_frame())


def test_context_manager_closes_owned_face_mesh():
    """face_mesh enjekte edilmeden (gerçek MediaPipe kurulacak şekilde)
    context manager kullanılırsa __exit__ close()'u tetiklemeli — burada
    gerçek FaceMesh'in var olup olmadığını değil, yalnızca akışın
    çökmediğini doğruluyoruz (bkz. aşağıdaki gerçek-mediapipe smoke
    testi, orada gerçek kaynak yönetimi de kanıtlanıyor)."""
    with MediaPipeEyeLocator(face_mesh=FakeFaceMesh(landmarks=None)) as locator:
        assert locator.locate(make_frame()) == ()


def test_locate_raises_typeerror_for_non_ndarray_frame():
    locator = MediaPipeEyeLocator(face_mesh=FakeFaceMesh(landmarks=None))

    with pytest.raises(TypeError):
        locator.locate("not a frame")


def test_real_mediapipe_on_blank_frame_returns_empty_tuple():
    """opencv-contrib gibi mediapipe'ın da gerçekten kurulu/çalışır
    olduğunu doğrular (bkz. test_eye_region_locator.py'deki
    test_loads_real_bundled_haar_cascade ile aynı amaç) — boş/siyah bir
    frame'de gerçekçi olarak hiçbir yüz bulunmaz, bu da beklenen sonuçtur.
    Gerçek göz tespiti/doğruluğu burada test EDİLMİYOR (sentetik
    frame'lerde mümkün değil, gerçek bir yüz gerektirir) — yalnızca
    gerçek MediaPipe FaceMesh'in kurulup çökmeden çalıştığı kanıtlanıyor.
    """
    with MediaPipeEyeLocator() as locator:  # face_mesh=None -> gerçek FaceMesh
        result = locator.locate(make_frame())

    assert result == ()
