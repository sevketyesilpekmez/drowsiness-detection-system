from __future__ import annotations

import cv2
import numpy as np

from app.services.eye_classifier_service import CLOSED_EYES, OPEN_EYES

# EyeStateCNN'nin 0/1 çıktı indekslerinin karşılığı. Bu sıralama
# torchvision.datasets.ImageFolder'ın klasör isimlerini ALFABETİK
# sıralamasından gelir ("closed" < "open") — bkz. train_eye_classifier.py.
CLASS_NAMES = (CLOSED_EYES, OPEN_EYES)


class TorchEyeClassifier:
    """
    EyeStateCNN'i sarmalayan, EyeClassifierModelLike sözleşmesine uyan
    (numpy içeri, (label, confidence) dışarı) gerçek PyTorch inference
    adaptörü.

    EyeClassifierService bu sınıfı hiç bilmez, yalnızca çağrılabilir
    olduğunu bilir — bu ayrım sayesinde EyeClassifierService torch kurulu
    olmayan bir ortamda bile test edilebiliyor (bkz.
    eye_classifier_service.py'deki tasarım notu).
    """

    def __init__(
        self,
        *,
        weights_path: str = "weights/eye_classifier.pth",
        device: str = "cpu",
        crop_size: int = 64,
    ) -> None:
        """
        Args:
            weights_path: train_eye_classifier.py'nin ürettiği checkpoint yolu.
            device: "cpu" veya "cuda:0" gibi bir torch cihaz string'i.
            crop_size: Modelin eğitildiği kare girdi boyutu (piksel) —
                train_eye_classifier.py'deki EYE_CROP_SIZE ile aynı olmalı,
                aksi halde model tanımadığı bir çözünürlükle karşılaşır
                (hata vermez, sessizce performans düşürür).
        """
        # Lazy import: torch kurulu olmayan ortamda bile geri kalan kod
        # tabanı import edilebilsin diye.
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "torch kurulu değil. Gerçek YOLO/PyTorch backend'ini kullanmak için: "
                "pip install torch --index-url https://download.pytorch.org/whl/cpu"
            ) from exc

        from app.models.eye_state_cnn import EyeStateCNN

        self._torch = torch
        self._device = torch.device(device)
        self._crop_size = crop_size

        model = EyeStateCNN()
        state_dict = torch.load(weights_path, map_location=self._device)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self._device)

        self._model = model

    def __call__(self, eye_crop: np.ndarray) -> tuple[str, float]:
        torch = self._torch

        tensor = self._preprocess(eye_crop)

        with torch.no_grad():
            logits = self._model(tensor)
            probabilities = torch.softmax(logits, dim=1)

        predicted_index = int(torch.argmax(probabilities, dim=1).item())
        confidence = float(probabilities[0, predicted_index].item())

        return CLASS_NAMES[predicted_index], confidence

    def _preprocess(self, eye_crop: np.ndarray):
        torch = self._torch

        grayscale = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY) if eye_crop.ndim == 3 else eye_crop
        resized = cv2.resize(grayscale, (self._crop_size, self._crop_size), interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0

        # (H, W) -> (1, 1, H, W): batch boyutu 1, kanal boyutu 1 (gri tonlama)
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
        return tensor.to(self._device)
