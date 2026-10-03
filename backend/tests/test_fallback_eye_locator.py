from __future__ import annotations

import numpy as np

from app.services.eye_region_locator import EyeBox
from app.services.fallback_eye_locator import FallbackEyeLocator
from app.services.person_detector import PersonBox


class FakeEyeLocator:
    """EyeLocatorLike'a uyan minimal sahte bulucu — kaç kez ve hangi
    argümanlarla çağrıldığını kaydeder."""

    def __init__(self, *, boxes=()):
        self.boxes = boxes
        self.call_count = 0
        self.closed = False
        self.received_search_region = "not-called"

    def locate(self, frame, *, search_region=None):
        self.call_count += 1
        self.received_search_region = search_region
        return self.boxes

    def close(self):
        self.closed = True


class FakeEyeLocatorWithoutClose:
    """close() metodu OLMAYAN bir bulucu — FallbackEyeLocator.close()'un
    getattr ile güvenli şekilde atlaması gerektiğini doğrulamak için."""

    def __init__(self, *, boxes=()):
        self.boxes = boxes

    def locate(self, frame, *, search_region=None):
        return self.boxes


def make_frame() -> np.ndarray:
    return np.zeros((100, 100, 3), dtype=np.uint8)


LEFT_BOX = EyeBox(x1=10, y1=10, x2=30, y2=30)
RIGHT_BOX = EyeBox(x1=50, y1=10, x2=70, y2=30)


def test_uses_primary_result_when_non_empty():
    primary = FakeEyeLocator(boxes=(LEFT_BOX, RIGHT_BOX))
    fallback = FakeEyeLocator(boxes=(LEFT_BOX,))
    locator = FallbackEyeLocator(primary=primary, fallback=fallback)

    boxes = locator.locate(make_frame())

    assert boxes == (LEFT_BOX, RIGHT_BOX)
    assert primary.call_count == 1
    assert fallback.call_count == 0


def test_falls_back_when_primary_returns_empty():
    primary = FakeEyeLocator(boxes=())
    fallback = FakeEyeLocator(boxes=(LEFT_BOX,))
    locator = FallbackEyeLocator(primary=primary, fallback=fallback)

    boxes = locator.locate(make_frame())

    assert boxes == (LEFT_BOX,)
    assert primary.call_count == 1
    assert fallback.call_count == 1


def test_returns_empty_when_both_return_empty():
    primary = FakeEyeLocator(boxes=())
    fallback = FakeEyeLocator(boxes=())
    locator = FallbackEyeLocator(primary=primary, fallback=fallback)

    assert locator.locate(make_frame()) == ()


def test_search_region_is_forwarded_to_both():
    primary = FakeEyeLocator(boxes=())
    fallback = FakeEyeLocator(boxes=())
    locator = FallbackEyeLocator(primary=primary, fallback=fallback)
    region = PersonBox(x1=0, y1=0, x2=50, y2=50, confidence=0.9)

    locator.locate(make_frame(), search_region=region)

    assert primary.received_search_region == region
    assert fallback.received_search_region == region


def test_close_closes_both_locators():
    primary = FakeEyeLocator()
    fallback = FakeEyeLocator()
    locator = FallbackEyeLocator(primary=primary, fallback=fallback)

    locator.close()

    assert primary.closed is True
    assert fallback.closed is True


def test_close_skips_locators_without_close_method():
    primary = FakeEyeLocatorWithoutClose()
    fallback = FakeEyeLocatorWithoutClose()
    locator = FallbackEyeLocator(primary=primary, fallback=fallback)

    locator.close()  # patlamamalı
