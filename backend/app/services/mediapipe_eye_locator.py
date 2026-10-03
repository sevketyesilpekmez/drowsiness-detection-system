from __future__ import annotations

from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from app.services.eye_region_locator import EyeBox
from app.services.landmark_service import LEFT_EYE_INDICES, RIGHT_EYE_INDICES, FaceMeshLike
from app.services.person_detector import PersonBox


class MediaPipeEyeLocator:
    """
    MediaPipe Face Mesh landmark'larından sol/sağ göz bölgelerini (EyeBox)
    çıkarır. `EyeRegionLocator` (Haar) ile aynı `locate()` sözleşmesine
    uyar (bkz. `EyeLocatorLike`) — onun yerine ya da `FallbackEyeLocator`
    içinde birincil bulucu olarak enjekte edilebilir.

    Haar cascade, kullanıcı kameraya yaklaşıp başını öne eğdiğinde
    (uyuklarken tipik poz) göz bulmakta zorlanıyordu — frontal/dik açıya
    göre eğitilmiş, kafa pozu değiştikçe kırılganlaşıyor (bkz. README
    "Baş eğikliği"). MediaPipe Face Mesh 3D yüz landmark tahmini yaptığı
    için baş açısına çok daha dayanıklı — zaten mediapipe backend'inin
    (landmark_service.py) kullandığı aynı teknoloji, yeni bir bağımlılık
    değil.

    `landmark_service.py`'deki `LEFT_EYE_INDICES`/`RIGHT_EYE_INDICES`
    aynen yeniden kullanılıyor — iki ayrı indeks seti tutmamak için.

    Önemli fark: `search_region` (YOLO'nun kişi kutusu) burada sessizce
    yok sayılır — MediaPipe kendi yüz tespitçisiyle tüm frame'de çalışır,
    bir alt-bölgeye kırpmak beklediği bağlamı bozabilir. Parametre
    yalnızca `EyeLocatorLike` sözleşmesine uymak için burada duruyor.

    Not: bu, her karede üçüncü bir model (YOLO + PyTorch sınıflandırıcıya
    ek olarak Face Mesh) çalıştırır, FPS'i bir miktar düşürür (bkz. README
    "FPS"). Baş pozuna dayanıklılık için kabul edilen bir trade-off;
    gerçek maliyeti `eye_pipeline_ms` alanından izlenebilir.
    """

    def __init__(
        self,
        *,
        eye_padding_ratio: float = 0.4,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        refine_landmarks: bool = True,
        face_mesh: FaceMeshLike | None = None,
    ) -> None:
        """
        Args:
            eye_padding_ratio: Crop kare (bkz. `_landmark_bounding_box` —
                büyük olan boyut temel alınır, TorchEyeClassifier'ın sabit
                64x64 resize'ı görüntüyü germesin diye). Kare kenarı
                `max(genişlik, yükseklik) * (1 + 2*eye_padding_ratio)`
                kadar büyütülür. 0.4 makul bir başlangıç değeri; doğruluk
                düşükse ilk denenecek ayar burası.
            max_num_faces / min_detection_confidence /
                min_tracking_confidence / refine_landmarks: LandmarkService
                ile aynı anlama gelir, MediaPipe FaceMesh'e aynen geçirilir.
            face_mesh: Testlerde sahte bir nesne enjekte etmek için (bkz.
                `FaceMeshLike`, landmark_service.py'de tanımlı). Enjekte
                edilirse ömrü bu sınıfa ait değildir, close() onu kapatmaz.
        """
        if eye_padding_ratio < 0:
            raise ValueError("eye_padding_ratio negatif olamaz.")
        if max_num_faces <= 0:
            raise ValueError("max_num_faces sıfırdan büyük olmalıdır.")
        self._validate_probability("min_detection_confidence", min_detection_confidence)
        self._validate_probability("min_tracking_confidence", min_tracking_confidence)

        self._eye_padding_ratio = eye_padding_ratio
        self._owns_face_mesh = face_mesh is None
        self._closed = False

        if face_mesh is not None:
            self._face_mesh = face_mesh
        else:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=max_num_faces,
                refine_landmarks=refine_landmarks,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )

    def locate(self, frame: np.ndarray, *, search_region: PersonBox | None = None) -> tuple[EyeBox, ...]:
        """
        MediaPipe Face Mesh ile frame'de bir yüz arar, bulunursa sol/sağ
        göz bölgelerini `EyeBox` olarak döndürür.

        `search_region` kasıtlı olarak yok sayılır (bkz. sınıf docstring'i).

        Returns:
            Yüz bulunamazsa boş tuple, bulunursa (sol, sağ) en fazla 2
            `EyeBox` — `EyeRegionLocator.locate()` ile tutarlı olsun diye.
        """
        if self._closed:
            raise RuntimeError("Kapatılmış MediaPipeEyeLocator tekrar kullanılamaz.")

        self._validate_frame(frame)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        output = self._face_mesh.process(rgb_frame)
        face_landmarks_collection = getattr(output, "multi_face_landmarks", None)

        if not face_landmarks_collection:
            return ()

        landmarks = face_landmarks_collection[0].landmark
        height, width = frame.shape[:2]

        left_box = self._landmark_bounding_box(landmarks, LEFT_EYE_INDICES, width, height)
        right_box = self._landmark_bounding_box(landmarks, RIGHT_EYE_INDICES, width, height)

        return tuple(box for box in (left_box, right_box) if box is not None)

    def close(self) -> None:
        """MediaPipe kaynaklarını serbest bırakır. Enjekte edilmiş bir
        face_mesh'i kapatmaz — o kaynağın sahibi bu sınıf değildir."""
        if self._closed:
            return
        if self._owns_face_mesh:
            self._face_mesh.close()
        self._closed = True

    def __enter__(self) -> MediaPipeEyeLocator:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def _landmark_bounding_box(
        self, landmarks: Any, indices: tuple[int, ...], frame_width: int, frame_height: int
    ) -> EyeBox | None:
        """Landmark indekslerinin piksel sınır kutusundan kare bir crop
        bölgesi hesaplar. Landmark indeksi mevcut değilse None döner —
        tüm frame'i MODEL_ERROR'a düşürmek yerine o gözü atlar.

        İlk sürüm x/y eksenlerini aynı oranla padliyordu, ama göz
        landmark'larının ham kutusu yataydır (genişlik >> yükseklik) ve
        göz kapanınca yükseklik neredeyse sıfırlanır. Sabit 64x64 resize
        (TorchEyeClassifier) böyle yassı bir dikdörtgeni kareye zorlayınca
        görüntü dikeyde ciddi gerilip kapalı gözler açık sanılıyordu.
        Düzeltme: iki boyuttan büyük olanı temel alıp göz merkezinde kare
        bir bölge kesiyoruz — açık/kapalı fark etmeksizin crop şekli
        tutarlı kalıyor."""
        xs: list[float] = []
        ys: list[float] = []

        for index in indices:
            if index >= len(landmarks):
                return None
            landmark = landmarks[index]
            xs.append(float(landmark.x) * frame_width)
            ys.append(float(landmark.y) * frame_height)

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        raw_width = max_x - min_x
        raw_height = max_y - min_y

        side = max(raw_width, raw_height) * (1.0 + 2.0 * self._eye_padding_ratio)

        x1 = max(0, int(center_x - side / 2.0))
        y1 = max(0, int(center_y - side / 2.0))
        x2 = min(frame_width, int(center_x + side / 2.0))
        y2 = min(frame_height, int(center_y + side / 2.0))

        if x2 <= x1 or y2 <= y1:
            return None

        return EyeBox(x1=x1, y1=y1, x2=x2, y2=y2)

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame bir numpy.ndarray olmalıdır.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame üç kanallı (BGR) bir görüntü olmalıdır.")
        if frame.size == 0:
            raise ValueError("frame boş olamaz.")

    @staticmethod
    def _validate_probability(name: str, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} sayısal olmalıdır.")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} 0.0 ile 1.0 arasında olmalıdır.")
