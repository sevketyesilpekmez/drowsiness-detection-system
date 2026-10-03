from __future__ import annotations

from typing import Any

import numpy as np

from app.services.person_detector import PersonBox

# `ultralytics` (ve zorunlu bağımlılığı `torch`) modül seviyesinde import
# edilmiyor — __init__ içine ertelenerek geri kalan kod tabanı torch kurulu
# olmadan da import edilebiliyor. Gerçek YOLO nesnesi yalnızca biri
# `UltralyticsPersonDetector()` oluşturduğunda (DETECTION_BACKEND=yolo_pytorch) lazım olur.


class UltralyticsPersonDetector:
    """
    Ultralytics YOLOv8 ile kare içinde "person" (COCO sınıf 0) tespiti
    yapar ve en yüksek güvenli kutuyu döndürür.

    Neden özel bir "face" modeli değil: resmi Ultralytics ağırlıkları
    (yolov8n.pt) COCO üzerinde eğitilmiştir ve "face" diye bir sınıfı
    yoktur. Gayri resmi yüz-özel YOLO ağırlıkları kaynağı/lisansı
    doğrulanamayan üçüncü parti dosyalar olduğu için kullanılmadı (bkz.
    proje kararı). Bunun yerine YOLO kabaca "kişi" kutusunu bulur,
    EyeRegionLocator (Haar cascade) o kutu içinde gözleri arar.
    """

    PERSON_CLASS_ID = 0

    def __init__(
        self,
        *,
        weights_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        device: str = "cpu",
        imgsz: int = 640,
        model: Any = None,
    ) -> None:
        """
        Args:
            weights_path: Ultralytics ağırlık dosyasının yolu ya da resmi
                bir model adı (örn. "yolov8n.pt"). Yerel bulunamazsa
                ultralytics kendi sunucularından otomatik indirir.
            confidence_threshold: Bu değerin altındaki tespitler yok sayılır.
            device: "cpu" veya "cuda:0" gibi ultralytics'in kabul ettiği
                cihaz string'i.
            imgsz: Inference öncesi kareyi ne boyuta letterbox'layacağı
                (piksel). Varsayılan 640 — ultralytics'in kendi varsayılanı.
                Düşürmek (416/320) CPU'da hız kazandırabilir ama küçük/uzak
                kişilerin kaçırılma riskini artırır.
            model: Testlerde ultralytics'in ham `YOLO` API'sini taklit eden
                sahte bir nesne enjekte etmek için.
        """
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold 0.0 ile 1.0 arasında olmalıdır.")
        if imgsz <= 0:
            raise ValueError("imgsz sıfırdan büyük olmalıdır.")

        self._confidence_threshold = confidence_threshold
        self._device = device
        self._imgsz = imgsz

        if model is not None:
            self._model = model
        else:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "ultralytics kurulu değil. Gerçek YOLO backend'ini kullanmak için: "
                    "pip install ultralytics torch --index-url https://download.pytorch.org/whl/cpu"
                ) from exc

            self._model = YOLO(weights_path)

    def __call__(self, frame: np.ndarray) -> PersonBox | None:
        """Ultralytics'in .predict()/çağrılabilir API'si üzerinden kişi
        tespiti yapar. Sonuç yoksa veya eşik altındaysa None döner."""
        results = self._model(
            frame,
            classes=[self.PERSON_CLASS_ID],
            device=self._device,
            imgsz=self._imgsz,
            verbose=False,
        )

        if not results:
            return None

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        confidences = boxes.conf.tolist()
        best_index = max(range(len(confidences)), key=lambda i: confidences[i])
        best_confidence = confidences[best_index]

        if best_confidence < self._confidence_threshold:
            return None

        x1, y1, x2, y2 = boxes.xyxy[best_index].tolist()

        height, width = frame.shape[:2]
        return PersonBox(
            x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2), confidence=float(best_confidence)
        ).clip(width=width, height=height)
