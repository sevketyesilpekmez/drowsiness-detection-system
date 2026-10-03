from __future__ import annotations

import numpy as np
import pytest

from app.services.eye_region_locator import EyeBox, EyeRegionLocator
from app.services.person_detector import PersonBox


class FakeCascade:
    """cv2.CascadeClassifier'ın detectMultiScale() imzasını taklit eden
    sahte bir cascade — gerçek Haar cascade, sentetik/boş frame'lerde
    deterministik sonuç üretmediği için (gerçek göz benzeri piksel
    desenleri gerekir) testler bunu kullanır."""

    def __init__(self, *, detections=()):
        self.detections = detections
        self.received_image = None
        self.received_kwargs = None

    def detectMultiScale(self, image, scaleFactor, minNeighbors, minSize):
        self.received_image = image
        self.received_kwargs = {
            "scaleFactor": scaleFactor,
            "minNeighbors": minNeighbors,
            "minSize": minSize,
        }
        return self.detections


def make_frame(width=200, height=150) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_locate_returns_boxes_in_full_frame_coordinates():
    cascade = FakeCascade(detections=[(10, 20, 30, 30)])
    locator = EyeRegionLocator(cascade=cascade)

    boxes = locator.locate(make_frame())

    assert boxes == (EyeBox(x1=10, y1=20, x2=40, y2=50),)


def test_locate_with_search_region_offsets_coordinates():
    """search_region verildiğinde cascade yalnızca o bölgede çalışır, ama
    dönen kutular yine TAM FRAME koordinatlarında olmalı (offset eklenmiş)."""
    cascade = FakeCascade(detections=[(5, 5, 10, 10)])
    locator = EyeRegionLocator(cascade=cascade)
    region = PersonBox(x1=50, y1=60, x2=150, y2=140, confidence=0.9)

    boxes = locator.locate(make_frame(), search_region=region)

    assert boxes == (EyeBox(x1=55, y1=65, x2=65, y2=75),)


def test_locate_passes_search_region_crop_not_full_frame():
    """search_region 80x100 (h,w) olsa da, varsayılan
    eye_search_height_ratio=0.70 yüzünden cascade'e yalnızca ÜST %70'i
    (56 satır) gönderilir — çene/boyun/gövde bölgesi hiç taranmaz (bkz.
    sınıf docstring'i, 'ağzı göz sanma' sorununun düzeltmesi)."""
    cascade = FakeCascade(detections=())
    locator = EyeRegionLocator(cascade=cascade)
    frame = make_frame(width=200, height=150)
    region = PersonBox(x1=50, y1=60, x2=150, y2=140, confidence=0.9)

    locator.locate(frame, search_region=region)

    assert cascade.received_image.shape == (56, 100)  # gray, int(80*0.70)=56, x2-x1=100


def test_locate_searches_full_region_height_when_ratio_is_1():
    cascade = FakeCascade(detections=())
    locator = EyeRegionLocator(cascade=cascade, eye_search_height_ratio=1.0)
    frame = make_frame(width=200, height=150)
    region = PersonBox(x1=50, y1=60, x2=150, y2=140, confidence=0.9)

    locator.locate(frame, search_region=region)

    assert cascade.received_image.shape == (80, 100)


def test_locate_does_not_restrict_height_when_no_search_region():
    """search_region=None (tüm frame'de arama) iken eye_search_height_ratio
    UYGULANMAZ — frame'in 'person kutusu' anatomisi olmadığı için bu
    kısıtlama anlamsız olurdu."""
    cascade = FakeCascade(detections=())
    locator = EyeRegionLocator(cascade=cascade, eye_search_height_ratio=0.1)
    frame = make_frame(width=200, height=150)

    locator.locate(frame)

    assert cascade.received_image.shape == (150, 200)


def test_locate_filters_out_wide_mouth_like_box():
    """Ağız açıkken Haar'ın ürettiği tipik kutu, gözden çok daha geniş/
    basıktır (en/boy oranı > 1.8) — bu kutu elenmeli."""
    # w=40, h=15 -> ratio=2.67, [0.65, 1.8] aralığının dışında.
    cascade = FakeCascade(detections=[(10, 5, 40, 15)])
    locator = EyeRegionLocator(cascade=cascade)

    boxes = locator.locate(make_frame())

    assert boxes == ()


def test_locate_keeps_eye_shaped_box_within_aspect_ratio() -> None:
    # w=20, h=18 -> ratio=1.11, aralık içinde.
    cascade = FakeCascade(detections=[(10, 5, 20, 18)])
    locator = EyeRegionLocator(cascade=cascade)

    boxes = locator.locate(make_frame())

    assert boxes == (EyeBox(x1=10, y1=5, x2=30, y2=23),)


def test_locate_filters_out_box_in_bottom_quarter_of_search_area():
    """Arama bölgesinin (tüm frame, height=150) alt %25'inde kalan bir
    kutu (merkez y > 112.5) elenmeli — bu genelde çene/ağız bölgesidir.
    Kutu boyutu bilerek geçerli bir en/boy oranında (1.11) seçildi ki
    yalnızca DİKEY konum filtresi test edilsin, aspect ratio filtresiyle
    karışmasın."""
    # w=20, h=18 (ratio=1.11, geçerli), y=125 -> center_y=134 > 0.75*150=112.5 -> elenmeli
    cascade = FakeCascade(detections=[(10, 125, 20, 18)])
    locator = EyeRegionLocator(cascade=cascade)

    boxes = locator.locate(make_frame())

    assert boxes == ()


def test_locate_selects_topmost_two_of_three_candidates() -> None:
    """Üç eş boyutlu (göz-benzeri) aday verildiğinde, en üstteki (en küçük
    y) ikisi seçilmeli — üçüncüsü (en alttaki, örn. ağız bölgesi) elenmeli."""
    top_left = (10, 10, 20, 18)  # y=10
    top_right = (60, 12, 20, 18)  # y=12
    bottom_mouth = (35, 40, 20, 18)  # y=40, en altta

    cascade = FakeCascade(detections=[bottom_mouth, top_left, top_right])
    locator = EyeRegionLocator(cascade=cascade)

    boxes = locator.locate(make_frame())

    assert len(boxes) == 2
    assert boxes == (
        EyeBox(x1=10, y1=10, x2=30, y2=28),
        EyeBox(x1=60, y1=12, x2=80, y2=30),
    )


def test_locate_merges_overlapping_detections_keeping_larger() -> None:
    """Haar'ın aynı gözü iki farklı ölçekte bulması yaygındır (çakışan
    kutular) — yalnızca daha büyük olan tutulmalı."""
    small = (10, 10, 16, 16)  # alan=256
    large_overlapping = (8, 8, 22, 22)  # alan=484, small ile yüksek IoU

    cascade = FakeCascade(detections=[small, large_overlapping])
    locator = EyeRegionLocator(cascade=cascade)

    boxes = locator.locate(make_frame())

    assert boxes == (EyeBox(x1=8, y1=8, x2=30, y2=30),)


def test_locate_keeps_non_overlapping_boxes_of_similar_size() -> None:
    """Çakışmayan iki gerçek göz kutusu, overlap filtresinden ikisi de
    sağ çıkmalı (yanlışlıkla birbirini elemesin)."""
    left_eye = (10, 10, 20, 18)
    right_eye = (60, 10, 20, 18)

    cascade = FakeCascade(detections=[left_eye, right_eye])
    locator = EyeRegionLocator(cascade=cascade)

    boxes = locator.locate(make_frame())

    assert len(boxes) == 2


def test_locate_caps_result_at_max_eyes() -> None:
    detections = [(x, 10, 20, 18) for x in (10, 40, 70, 100)]
    cascade = FakeCascade(detections=detections)
    locator = EyeRegionLocator(cascade=cascade, max_eyes=2)

    boxes = locator.locate(make_frame(width=200, height=150))

    assert len(boxes) == 2


def test_rejects_invalid_eye_search_height_ratio() -> None:
    with pytest.raises(ValueError):
        EyeRegionLocator(eye_search_height_ratio=0.0, cascade=FakeCascade())
    with pytest.raises(ValueError):
        EyeRegionLocator(eye_search_height_ratio=1.1, cascade=FakeCascade())


def test_rejects_invalid_aspect_ratio_bounds() -> None:
    with pytest.raises(ValueError):
        EyeRegionLocator(min_aspect_ratio=0.0, cascade=FakeCascade())
    with pytest.raises(ValueError):
        EyeRegionLocator(min_aspect_ratio=1.5, max_aspect_ratio=1.0, cascade=FakeCascade())


def test_rejects_invalid_bottom_exclusion_ratio() -> None:
    with pytest.raises(ValueError):
        EyeRegionLocator(bottom_exclusion_ratio=1.0, cascade=FakeCascade())
    with pytest.raises(ValueError):
        EyeRegionLocator(bottom_exclusion_ratio=-0.1, cascade=FakeCascade())


def test_rejects_invalid_overlap_iou_threshold() -> None:
    with pytest.raises(ValueError):
        EyeRegionLocator(overlap_iou_threshold=0.0, cascade=FakeCascade())


def test_rejects_invalid_max_eyes() -> None:
    with pytest.raises(ValueError):
        EyeRegionLocator(max_eyes=0, cascade=FakeCascade())


def test_locate_returns_empty_tuple_when_no_detections():
    cascade = FakeCascade(detections=())
    locator = EyeRegionLocator(cascade=cascade)

    assert locator.locate(make_frame()) == ()


def test_locate_returns_empty_tuple_for_empty_search_region():
    cascade = FakeCascade(detections=[(1, 1, 5, 5)])
    locator = EyeRegionLocator(cascade=cascade)
    empty_region = PersonBox(x1=10, y1=10, x2=10, y2=10, confidence=0.9)  # sıfır alan

    assert locator.locate(make_frame(), search_region=empty_region) == ()


def test_locate_forwards_configured_parameters_to_cascade():
    cascade = FakeCascade(detections=())
    locator = EyeRegionLocator(cascade=cascade, scale_factor=1.2, min_neighbors=5, min_size=(15, 15))

    locator.locate(make_frame())

    assert cascade.received_kwargs == {"scaleFactor": 1.2, "minNeighbors": 5, "minSize": (15, 15)}


def test_rejects_invalid_scale_factor():
    with pytest.raises(ValueError):
        EyeRegionLocator(scale_factor=1.0, cascade=FakeCascade())


def test_rejects_invalid_min_neighbors():
    with pytest.raises(ValueError):
        EyeRegionLocator(min_neighbors=0, cascade=FakeCascade())


def test_loads_real_bundled_haar_cascade():
    """opencv-contrib-python'la birlikte gelen gerçek haarcascade_eye.xml
    dosyasının başarıyla yüklendiğini doğrular (ek bir indirmeye gerek
    olmadığını kanıtlar) — gerçek tespit sentetik frame'lerde test
    edilmiyor (yukarıdaki testler bunu FakeCascade ile kapsıyor), sadece
    dosyanın var olup doğru şekilde parse edildiği doğrulanıyor."""
    locator = EyeRegionLocator()  # cascade=None -> gerçek dosyayı yükler

    # Gerçek cascade'le boş/siyah bir frame'de arama yapmak exception
    # atmamalı (muhtemelen hiçbir şey bulunmaz, ki bu da beklenen).
    result = locator.locate(make_frame())

    assert isinstance(result, tuple)


def test_raises_when_cascade_file_missing():
    with pytest.raises(RuntimeError):
        EyeRegionLocator(cascade_path="/nonexistent/path/to/cascade.xml")


# ---------------------------------------------------------------------------
# adaptive_min_size_ratio (FPS düşüklüğü araştırması sırasında eklendi —
# kamera çözünürlüğü artık gerçekten uygulandığı için (bkz. README "FPS
# düşüklüğü"), sabit piksel minSize'ın farklı çözünürlüklerde tutarsız
# davranma riski gerçekleşti; kullanıcı 640x360'ta göz tespitinin
# istikrarsızlaştığını bildirdi).
# ---------------------------------------------------------------------------


def test_adaptive_min_size_is_disabled_by_default():
    """adaptive_min_size_ratio verilmezse minSize eskisi gibi SABİT kalır
    — geriye dönük uyumluluk regresyon testi."""
    cascade = FakeCascade(detections=())
    locator = EyeRegionLocator(cascade=cascade, min_size=(20, 20))
    region = PersonBox(x1=0, y1=0, x2=200, y2=200, confidence=0.9)

    locator.locate(make_frame(width=200, height=200), search_region=region)

    assert cascade.received_kwargs["minSize"] == (20, 20)


def test_adaptive_min_size_scales_up_for_large_region():
    cascade = FakeCascade(detections=())
    locator = EyeRegionLocator(
        cascade=cascade, min_size=(20, 20), adaptive_min_size_ratio=(0.12, 0.08)
    )
    # search_region 300x300 -> eye_search_height_ratio=0.70 (varsayılan)
    # sonrası region 300x210. adaptive_w = max(20, 300*0.12=36) = 36.
    # adaptive_h = max(20, 210*0.08=16.8->16) = 20 (taban altında kalır).
    region = PersonBox(x1=0, y1=0, x2=300, y2=300, confidence=0.9)

    locator.locate(make_frame(width=300, height=300), search_region=region)

    assert cascade.received_kwargs["minSize"] == (36, 20)


def test_adaptive_min_size_never_goes_below_floor():
    """Küçük bir arama bölgesinde oran tabana (min_size) ulaşmıyorsa,
    min_size TABAN olarak korunmalı — asla altına inilmemeli."""
    cascade = FakeCascade(detections=())
    locator = EyeRegionLocator(
        cascade=cascade, min_size=(20, 20), adaptive_min_size_ratio=(0.05, 0.05)
    )
    region = PersonBox(x1=0, y1=0, x2=100, y2=100, confidence=0.9)

    locator.locate(make_frame(width=100, height=100), search_region=region)

    assert cascade.received_kwargs["minSize"] == (20, 20)


def test_adaptive_min_size_ignored_without_search_region():
    """search_region=None iken (tüm frame'de arama) ORANTILI minSize
    hesaplanmaz — sabit min_size kullanılır (frame boyutu 'göz/yüz oranı'
    anlamına gelmez, bkz. eye_search_height_ratio'nun aynı kısıtlaması)."""
    cascade = FakeCascade(detections=())
    locator = EyeRegionLocator(
        cascade=cascade, min_size=(20, 20), adaptive_min_size_ratio=(0.5, 0.5)
    )

    locator.locate(make_frame(width=200, height=150))

    assert cascade.received_kwargs["minSize"] == (20, 20)


def test_rejects_invalid_adaptive_min_size_ratio():
    with pytest.raises(ValueError):
        EyeRegionLocator(cascade=FakeCascade(), adaptive_min_size_ratio=(0.0, 0.1))
    with pytest.raises(ValueError):
        EyeRegionLocator(cascade=FakeCascade(), adaptive_min_size_ratio=(0.1, -0.1))


# ---------------------------------------------------------------------------
# split_left_right (mentor önerisi: göz bölgesini sol/sağ diye ikiye bölüp
# Haar'ı ayrı ayrı çalıştırmak, kaş/burun köprüsü civarındaki yanlış/
# istikrarsız adayları azaltmak için).
# ---------------------------------------------------------------------------


class RecordingSplitCascade:
    """detectMultiScale'in kaç kez ve hangi görüntü genişlikleriyle
    çağrıldığını kaydeden sahte cascade — split_left_right'ın gerçekten
    İKİ ayrı çağrı yaptığını doğrulamak için."""

    def __init__(self, *, responses):
        # responses: her çağrıda sırayla döndürülecek detection listeleri.
        self._responses = list(responses)
        self.calls = []

    def detectMultiScale(self, image, scaleFactor, minNeighbors, minSize):
        self.calls.append({"width": image.shape[1], "minSize": minSize})
        return self._responses.pop(0) if self._responses else ()


def test_split_left_right_calls_cascade_twice_with_half_width():
    cascade = RecordingSplitCascade(responses=[(), ()])
    locator = EyeRegionLocator(cascade=cascade, split_left_right=True, eye_search_height_ratio=1.0)
    region = PersonBox(x1=0, y1=0, x2=100, y2=50, confidence=0.9)

    locator.locate(make_frame(width=100, height=50), search_region=region)

    assert len(cascade.calls) == 2
    assert cascade.calls[0]["width"] == 50
    assert cascade.calls[1]["width"] == 50


def test_split_left_right_offsets_right_half_detection():
    """Sağ yarıda bulunan bir kutunun x koordinatı, tam bölgeye göre
    (yarım genişlik kadar) offsetlenmeli."""
    # Sol yarı: hiçbir şey. Sağ yarı: (5, 5, 20, 18) -> tam bölgede x=55.
    cascade = RecordingSplitCascade(responses=[(), [(5, 5, 20, 18)]])
    locator = EyeRegionLocator(cascade=cascade, split_left_right=True, eye_search_height_ratio=1.0)
    region = PersonBox(x1=0, y1=0, x2=100, y2=50, confidence=0.9)

    boxes = locator.locate(make_frame(width=100, height=50), search_region=region)

    assert boxes == (EyeBox(x1=55, y1=5, x2=75, y2=23),)


def test_split_left_right_combines_both_halves_and_applies_existing_filters():
    """Sol yarıda bir göz, sağ yarıda bir göz bulunursa ikisi de sonuçta
    olmalı — split, mevcut aspect-ratio/overlap filtrelerini bypass
    etmemeli."""
    left_eye = (10, 5, 20, 18)  # sol yarı içinde (yarı genişlik=50)
    right_eye_local = (5, 5, 20, 18)  # sağ yarı içinde, offsetlenince x=55
    cascade = RecordingSplitCascade(responses=[[left_eye], [right_eye_local]])
    locator = EyeRegionLocator(cascade=cascade, split_left_right=True, eye_search_height_ratio=1.0)
    region = PersonBox(x1=0, y1=0, x2=100, y2=50, confidence=0.9)

    boxes = locator.locate(make_frame(width=100, height=50), search_region=region)

    assert len(boxes) == 2


def test_split_left_right_disabled_by_default_calls_cascade_once():
    cascade = RecordingSplitCascade(responses=[()])
    locator = EyeRegionLocator(cascade=cascade)  # split_left_right varsayılan False

    locator.locate(make_frame())

    assert len(cascade.calls) == 1
