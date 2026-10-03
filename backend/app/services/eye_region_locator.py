from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import cv2
import numpy as np

from app.services.person_detector import PersonBox


@dataclass(frozen=True)
class EyeBox:
    """Tam frame koordinatlarında bir göz bölgesinin sınırlayıcı kutusu."""

    x1: int
    y1: int
    x2: int
    y2: int


class CascadeLike(Protocol):
    """cv2.CascadeClassifier'ı testlerde sahte bir cascade ile değiştirmek
    için kontrat (gerçek Haar, sentetik/boş frame'lerde deterministik
    tespit üretmez)."""

    def detectMultiScale(  # noqa: N802 - OpenCV'nin kendi metod adı
        self, image: np.ndarray, scaleFactor: float, minNeighbors: int, minSize: tuple[int, int]
    ) -> Any: ...


class EyeLocatorLike(Protocol):
    """Göz bölgesi bulucularının ortak sözleşmesi. `EyeRegionLocator`
    (Haar), `MediaPipeEyeLocator` (Face Mesh) ve `FallbackEyeLocator`
    (ikisini birleştiren wrapper) hepsi bunu uygular — YoloDrowsinessDetector
    hangisinin enjekte edildiğini bilmeden tek bir `locate()` çağırır."""

    def locate(self, frame: np.ndarray, *, search_region: PersonBox | None = None) -> tuple[EyeBox, ...]: ...


class EyeRegionLocator:
    """
    OpenCV'nin yerleşik Haar cascade'iyle (haarcascade_eye.xml, ekstra
    indirme gerekmez) bir kişi kutusu içinde göz bölgelerini bulur.

    YOLO'nun bulduğu kaba "person" kutusunu daraltıp yalnızca o bölgede
    arar (bkz. locate()'in search_region parametresi) — hem performansı
    artırır hem yanlış pozitifleri azaltır.

    Haar, kişi kutusunun tamamında arandığında bazen ağzı göz sanıyordu.
    Bunu azaltmak için dört filtre var (varsayılanlar gerçek kamerayla
    denenmiş, ince ayar gerekebilir, bkz. README):
      1. Arama yalnızca kutunun üst `eye_search_height_ratio`'sunda
         yapılır (varsayılan 0.70 — 0.55'ten belirgin şekilde daha iyi
         sonuç verdi).
      2. En/boy oranı aralığı dışındaki kutular elenir (ağız açıkken
         daha geniş/basık bir kutu üretir).
      3. Arama bölgesinin alt `bottom_exclusion_ratio`'sunda kalan
         kutular elenir (ek güvenlik payı).
      4. Yüksek oranda çakışan (IoU > `overlap_iou_threshold`) kutulardan
         yalnızca büyüğü tutulur (Haar aynı gözü farklı ölçeklerde birden
         fazla kez bulabilir).
    Sonuç kutuları y sonra x'e göre sıralanıp en fazla `max_eyes`
    (varsayılan 2) tanesi döndürülür.
    """

    def __init__(
        self,
        *,
        cascade_path: str | None = None,
        scale_factor: float = 1.1,
        min_neighbors: int = 8,
        min_size: tuple[int, int] = (20, 20),
        eye_search_height_ratio: float = 0.70,
        min_aspect_ratio: float = 0.65,
        max_aspect_ratio: float = 1.8,
        bottom_exclusion_ratio: float = 0.25,
        overlap_iou_threshold: float = 0.3,
        max_eyes: int = 2,
        adaptive_min_size_ratio: tuple[float, float] | None = None,
        split_left_right: bool = False,
        cascade: CascadeLike | None = None,
    ) -> None:
        """
        Args:
            cascade_path: Haar cascade XML yolu. None ise OpenCV'nin
                paketiyle gelen `haarcascade_eye.xml` kullanılır.
            scale_factor: Her ölçekte görüntünün küçültülme oranı (> 1.0).
            min_neighbors: Bir bölgenin "göz" sayılması için gereken
                minimum komşu tespit sayısı.
            min_size: Kabul edilecek minimum göz bölgesi boyutu (piksel).
                `adaptive_min_size_ratio` verilmemişse sabit kullanılır,
                verilmişse bir taban (floor) olarak kullanılır.
            eye_search_height_ratio: `search_region` verildiğinde kutunun
                yalnızca üstten bu orana kadarı taranır (0, 1]).
            min_aspect_ratio / max_aspect_ratio: Kabul edilecek kutuların
                genişlik/yükseklik oranı aralığı.
            bottom_exclusion_ratio: Arama bölgesinin alt bu oranında kalan
                kutular elenir ([0, 1)).
            overlap_iou_threshold: Bu değerden yüksek IoU'ya sahip kutu
                çiftlerinden yalnızca büyüğü tutulur ((0, 1]).
            max_eyes: locate()'in döndüreceği en fazla kutu sayısı.
            adaptive_min_size_ratio: (width_ratio, height_ratio) verilirse
                minSize sabit yerine arama bölgesinin boyutuna orantılı
                hesaplanır (min_size yine taban olarak kalır) — düşük/yüksek
                çözünürlükte sabit minSize'ın gerçek gözü reddetmesi ya da
                gereğinden küçük yanlış pozitif kabul etmesi riskini azaltır.
                None (varsayılan) iken eski sabit davranış korunur.
            split_left_right: True ise arama bölgesi dikeyde ikiye bölünüp
                Haar sol/sağ yarıda ayrı çalıştırılır, sonuçlar birleştirilip
                aynı filtrelerden geçirilir — amaç Haar'ın kaş/burun köprüsünü
                yanlış aday seçmesini azaltmak. Varsayılan False.
            cascade: Testlerde sahte bir cascade enjekte etmek için (bkz.
                CascadeLike).
        """
        if scale_factor <= 1.0:
            raise ValueError("scale_factor 1.0'dan büyük olmalıdır.")
        if min_neighbors <= 0:
            raise ValueError("min_neighbors sıfırdan büyük olmalıdır.")
        if not 0.0 < eye_search_height_ratio <= 1.0:
            raise ValueError("eye_search_height_ratio 0 ile 1 arasında olmalıdır (0 hariç, 1 dahil).")
        if min_aspect_ratio <= 0:
            raise ValueError("min_aspect_ratio sıfırdan büyük olmalıdır.")
        if max_aspect_ratio <= min_aspect_ratio:
            raise ValueError("max_aspect_ratio min_aspect_ratio'dan büyük olmalıdır.")
        if not 0.0 <= bottom_exclusion_ratio < 1.0:
            raise ValueError("bottom_exclusion_ratio 0 ile 1 arasında olmalıdır (1 hariç).")
        if not 0.0 < overlap_iou_threshold <= 1.0:
            raise ValueError("overlap_iou_threshold 0 ile 1 arasında olmalıdır (0 hariç, 1 dahil).")
        if max_eyes <= 0:
            raise ValueError("max_eyes sıfırdan büyük olmalıdır.")
        if adaptive_min_size_ratio is not None:
            ratio_w, ratio_h = adaptive_min_size_ratio
            if ratio_w <= 0 or ratio_h <= 0:
                raise ValueError("adaptive_min_size_ratio'nun her iki bileşeni de sıfırdan büyük olmalıdır.")

        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        self._min_size = min_size
        self._eye_search_height_ratio = eye_search_height_ratio
        self._min_aspect_ratio = min_aspect_ratio
        self._max_aspect_ratio = max_aspect_ratio
        self._bottom_exclusion_ratio = bottom_exclusion_ratio
        self._overlap_iou_threshold = overlap_iou_threshold
        self._max_eyes = max_eyes
        self._adaptive_min_size_ratio = adaptive_min_size_ratio
        self._split_left_right = split_left_right

        if cascade is not None:
            self._cascade = cascade
        else:
            path = cascade_path or (cv2.data.haarcascades + "haarcascade_eye.xml")
            loaded = cv2.CascadeClassifier(path)
            if loaded.empty():
                raise RuntimeError(f"Haar cascade yüklenemedi: {path}")
            self._cascade = loaded

    def locate(self, frame: np.ndarray, *, search_region: PersonBox | None = None) -> tuple[EyeBox, ...]:
        """
        Verilen frame içinde (varsa yalnızca search_region sınırları içinde,
        üst `eye_search_height_ratio` kısmında) göz bölgelerini arar.

        Returns:
            Tam frame koordinatlarında, y sonra x'e göre sıralanmış, en
            fazla `max_eyes` tane EyeBox. Bulunamazsa boş tuple döner.
        """
        if search_region is not None:
            if search_region.is_empty:
                # Boş bölge: "aranacak bir şey yok" demek, tüm frame'e dönmüyoruz.
                return ()

            offset_x, offset_y = search_region.x1, search_region.y1
            region = frame[search_region.y1 : search_region.y2, search_region.x1 : search_region.x2]

            # Gözler person kutusunun üst kısmındadır; alt kısım (çene/
            # boyun) ağzı göz sanma riskini artırıyordu (bkz. sınıf
            # docstring'i). search_region=None iken bu kısıt uygulanmaz.
            if region.size > 0:
                restricted_height = max(1, int(region.shape[0] * self._eye_search_height_ratio))
                region = region[:restricted_height, :]
        else:
            offset_x, offset_y = 0, 0
            region = frame

        if region.size == 0:
            return ()

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region
        # Adaptif minSize yalnızca search_region verildiğinde uygulanır —
        # tüm frame'de arama yapılırken (search_region=None) frame'in
        # "yüz/göz oranı" anlamı yoktur (eye_search_height_ratio'nun aynı
        # kısıtlamasıyla tutarlı, bkz. sınıf docstring'i).
        effective_min_size = self._compute_min_size(region) if search_region is not None else self._min_size

        if self._split_left_right and gray.shape[1] >= 2:
            detections = self._detect_split_left_right(gray, effective_min_size)
        else:
            detections = self._cascade.detectMultiScale(
                gray,
                scaleFactor=self._scale_factor,
                minNeighbors=self._min_neighbors,
                minSize=effective_min_size,
            )

        region_height = region.shape[0]
        candidates: list[tuple[int, int, int, int]] = []
        for x, y, w, h in detections:
            x, y, w, h = int(x), int(y), int(w), int(h)
            if not self._passes_aspect_ratio(w, h):
                continue
            if not self._passes_vertical_position(y, h, region_height):
                continue
            candidates.append((x, y, w, h))

        kept = self._merge_overlapping(candidates)
        kept.sort(key=lambda box: (box[1], box[0]))  # önce y, sonra x
        limited = kept[: self._max_eyes]

        return tuple(
            EyeBox(x1=x + offset_x, y1=y + offset_y, x2=x + w + offset_x, y2=y + h + offset_y)
            for x, y, w, h in limited
        )

    def _compute_min_size(self, region: np.ndarray) -> tuple[int, int]:
        """`adaptive_min_size_ratio` verilmemişse sabit `min_size`'ı aynen
        döndürür (eski davranış). Verilmişse, arama bölgesinin (split
        edilmemiş, TAM) boyutuna orantılı bir minSize hesaplar — `min_size`
        hâlâ bir taban (floor) olarak uygulanır, asla altına inilmez."""
        if self._adaptive_min_size_ratio is None:
            return self._min_size

        ratio_w, ratio_h = self._adaptive_min_size_ratio
        region_height, region_width = region.shape[:2]
        adaptive_w = max(self._min_size[0], int(region_width * ratio_w))
        adaptive_h = max(self._min_size[1], int(region_height * ratio_h))
        return (adaptive_w, adaptive_h)

    def _detect_split_left_right(
        self, gray: np.ndarray, min_size: tuple[int, int]
    ) -> list[tuple[int, int, int, int]]:
        """Gri arama bölgesini dikeyde ikiye bölüp Haar cascade'i SOL ve
        SAĞ yarıda AYRI AYRI çalıştırır, sağ yarının x koordinatlarını
        tam bölgeye göre offsetleyip birleştirir. Amaç: Haar'ın kaş/burun
        köprüsü gibi iki göz ARASINDAKİ bölgeyi yanlışlıkla tek/yanlış bir
        aday olarak bulma riskini azaltmak (bkz. sınıf docstring'i,
        split_left_right notu)."""
        width = gray.shape[1]
        half = width // 2

        left_detections = self._cascade.detectMultiScale(
            gray[:, :half],
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=min_size,
        )
        right_detections = self._cascade.detectMultiScale(
            gray[:, half:],
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=min_size,
        )

        combined = [tuple(int(v) for v in box) for box in left_detections]
        combined += [(int(x) + half, int(y), int(w), int(h)) for x, y, w, h in right_detections]
        return combined

    def _passes_aspect_ratio(self, width: int, height: int) -> bool:
        if height <= 0:
            return False
        ratio = width / height
        return self._min_aspect_ratio <= ratio <= self._max_aspect_ratio

    def _passes_vertical_position(self, y: int, height: int, region_height: int) -> bool:
        if region_height <= 0:
            return True
        center_y = y + height / 2
        return center_y < (1.0 - self._bottom_exclusion_ratio) * region_height

    def _merge_overlapping(
        self, candidates: list[tuple[int, int, int, int]]
    ) -> list[tuple[int, int, int, int]]:
        """Birbiriyle yüksek oranda çakışan (IoU > overlap_iou_threshold)
        kutulardan yalnızca alanı en büyük olanı tutar. Haar'ın kendi
        API'si bir güven skoru döndürmediği için (bkz. CascadeLike), "daha
        güvenilir" kutunun ne olduğuna dair elimizdeki tek nesnel sinyal
        kutunun boyutudur."""
        ordered = sorted(candidates, key=lambda box: box[2] * box[3], reverse=True)
        kept: list[tuple[int, int, int, int]] = []
        for candidate in ordered:
            if not any(self._iou(candidate, existing) > self._overlap_iou_threshold for existing in kept):
                kept.append(candidate)
        return kept

    @staticmethod
    def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh

        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        union_area = aw * ah + bw * bh - inter_area
        if union_area <= 0:
            return 0.0

        return inter_area / union_area
