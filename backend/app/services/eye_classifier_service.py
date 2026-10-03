from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

OPEN_EYES = "OPEN_EYES"
CLOSED_EYES = "CLOSED_EYES"
VALID_LABELS = frozenset({OPEN_EYES, CLOSED_EYES})


@dataclass(frozen=True)
class EyeClassification:
    """Tek bir göz kırpıntısının (crop) sınıflandırma sonucu."""

    label: str
    confidence: float

    @property
    def is_closed(self) -> bool:
        return self.label == CLOSED_EYES


class EyeClassifierModelLike(Protocol):
    """EyeClassifierService'in enjekte edebileceği modellerin kontratı.

    numpy içeri alır, (label, confidence) tuple'ı döndürür — torch.Tensor
    değil. numpy<->tensor dönüşümü gerçek modelin sorumluluğunda (bkz.
    TorchEyeClassifier), bu sayede EyeClassifierService hiç torch import
    etmez ve torch kurulu olmayan bir ortamda da test edilebilir.
    """

    def __call__(self, eye_crop: np.ndarray) -> tuple[str, float]: ...


class EyeClassifierService:
    """
    Kırpılmış bir göz görüntüsünü OPEN_EYES/CLOSED_EYES olarak sınıflandıran
    modeli sarmalar (wrapper).

    Bu sınıf PyTorch'un ne olduğunu bilmez — yalnızca EyeClassifierModelLike
    sözleşmesine uyan bir model bekler. Gerçek çalışmada bu TorchEyeClassifier
    olur (bkz. torch_eye_classifier.py); testlerde sahte bir model enjekte
    edilir.
    """

    def __init__(self, *, model: EyeClassifierModelLike, min_crop_size: int = 8) -> None:
        """
        Args:
            model: EyeClassifierModelLike sözleşmesine uyan, gerçek veya
                sahte bir sınıflandırıcı.
            min_crop_size: Kabul edilecek minimum crop genişliği/yüksekliği
                (piksel) — bundan küçük crop'lar muhtemelen EyeRegionLocator'ın
                yanlış pozitif bir tespitidir, modele hiç gönderilmez.
        """
        if model is None or not callable(model):
            raise TypeError("model çağrılabilir bir nesne olmalıdır (None olamaz).")

        if min_crop_size <= 0:
            raise ValueError("min_crop_size sıfırdan büyük olmalıdır.")

        self._model = model
        self._min_crop_size = min_crop_size

    def classify(self, eye_crop: np.ndarray) -> EyeClassification:
        """
        Bir göz kırpıntısını sınıflandırır.

        Raises:
            TypeError: eye_crop bir numpy.ndarray değilse.
            ValueError: eye_crop çok küçükse, ya da model sözleşmeyi
                bozan bir değer döndürürse (bilinmeyen etiket / [0,1]
                dışında confidence).
        """
        self._validate_crop(eye_crop)

        label, confidence = self._model(eye_crop)

        if label not in VALID_LABELS:
            raise ValueError(f"Model geçersiz bir etiket döndürdü: {label!r} (beklenen: {sorted(VALID_LABELS)})")

        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise TypeError("confidence sayısal olmalıdır.")

        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("confidence 0.0 ile 1.0 arasında olmalıdır.")

        return EyeClassification(label=label, confidence=round(float(confidence), 4))

    def _validate_crop(self, eye_crop: np.ndarray) -> None:
        if not isinstance(eye_crop, np.ndarray):
            raise TypeError("eye_crop bir numpy.ndarray olmalıdır.")

        if eye_crop.ndim not in (2, 3):
            raise ValueError("eye_crop 2 (gri) veya 3 (renkli) boyutlu olmalıdır.")

        height, width = eye_crop.shape[:2]
        if height < self._min_crop_size or width < self._min_crop_size:
            raise ValueError(
                f"eye_crop çok küçük ({width}x{height}), minimum {self._min_crop_size}x{self._min_crop_size} olmalı."
            )
