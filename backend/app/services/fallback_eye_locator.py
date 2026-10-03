from __future__ import annotations

import numpy as np

from app.services.eye_region_locator import EyeBox, EyeLocatorLike
from app.services.person_detector import PersonBox


class FallbackEyeLocator:
    """
    Birincil bir göz bulucuyu (`primary`) dener; hiç kutu döndürmezse
    (örn. MediaPipe yüz/landmark bulamadıysa) yedek bir bulucuya
    (`fallback`) düşer.

    `EyeLocatorLike` sözleşmesine uyar — `YoloDrowsinessDetector`'a
    doğrudan enjekte edilebilir, hangi bulucunun devrede olduğunu bilmez.

    MediaPipeEyeLocator baş pozuna Haar'dan çok daha dayanıklı, ama kendi
    yüz tespitçisi de her koşulda (karanlık, aşırı yakın kırpma, profil
    açı) başarısız olabilir. Bu durumda Haar'a (EyeRegionLocator) düşmek,
    o karede hiç göz kutusu üretmemekten (EyeStateSmoother'ın kareyi
    geçersiz saymasından) daha iyi.

    `primary`/`fallback`'in ne olduğunu bilmez, yalnızca `EyeLocatorLike`
    sözleşmesine uyduklarını varsayar — testlerde iki sahte bulucuyla
    kolayca doğrulanabilir.
    """

    def __init__(self, *, primary: EyeLocatorLike, fallback: EyeLocatorLike) -> None:
        self._primary = primary
        self._fallback = fallback

    def locate(self, frame: np.ndarray, *, search_region: PersonBox | None = None) -> tuple[EyeBox, ...]:
        boxes = self._primary.locate(frame, search_region=search_region)
        if boxes:
            return boxes
        return self._fallback.locate(frame, search_region=search_region)

    def close(self) -> None:
        """`primary`/`fallback`'in `close()` metodu varsa çağırır, yoksa
        (örn. bir test double'ı) sessizce atlar."""
        for locator in (self._primary, self._fallback):
            close_fn = getattr(locator, "close", None)
            if callable(close_fn):
                close_fn()
