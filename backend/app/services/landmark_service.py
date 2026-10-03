from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import cv2
import mediapipe as mp
import numpy as np

# EAR hesabında kullanılan altı göz noktası (standart MediaPipe Face Mesh
# 468 nokta topolojisi — refine_landmarks=True ile eklenen iris noktaları
# 468-477 aralığında sona eklenir, bu indeksleri etkilemez).
#
# Sıralama:
# p1: gözün sol köşesi
# p2: üst-sol
# p3: üst-sağ
# p4: sağ köşe
# p5: alt-sağ
# p6: alt-sol
LEFT_EYE_INDICES = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_INDICES = (362, 385, 387, 263, 373, 380)


class FaceMeshLike(Protocol):
    """MediaPipe FaceMesh ile testlerde kullanılan sahte nesnelerin kontratı."""

    def process(self, image: np.ndarray) -> Any: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class Point:
    """Görüntü üzerindeki iki boyutlu piksel koordinatı."""

    x: float
    y: float


@dataclass(frozen=True)
class LandmarkResult:
    """Tek frame için landmark analizinin sonucu."""

    face_detected: bool
    eyes_closed: bool
    confidence: float
    ear: float | None
    left_ear: float | None
    right_ear: float | None
    left_eye_points: tuple[Point, ...]
    right_eye_points: tuple[Point, ...]

    def as_dict(self) -> dict[str, Any]:
        """Sonucu DetectionEngine tarafından kullanılabilecek sözlüğe dönüştürür."""
        return {
            "face_detected": self.face_detected,
            "eyes_closed": self.eyes_closed,
            "confidence": self.confidence,
            "ear": self.ear,
            "left_ear": self.left_ear,
            "right_ear": self.right_ear,
        }


class LandmarkService:
    """
    MediaPipe FaceMesh kullanarak göz landmark noktalarını ve EAR (Eye
    Aspect Ratio) değerini hesaplar.

    Bu sınıf yalnızca görüntüden ölçüm çıkarır. Kullanıcının DROWSY olup
    olmadığına karar vermez; süreye bağlı karar DrowsinessService'e aittir
    (bkz. drowsiness_detector.py, ikisini birleştiren yer).

    `confidence`, MediaPipe'ın model güven skoru DEĞİLDİR — FaceMesh bunu
    sağlamaz. EAR'ın eşikten ne kadar uzakta olduğuna göre hesaplanan bir
    karar güvenidir (eşiğe yakın ölçümlerde düşük, uzak ölçümlerde yüksek).
    """

    def __init__(
        self,
        *,
        ear_threshold: float = 0.22,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        refine_landmarks: bool = True,
        face_mesh: FaceMeshLike | None = None,
    ) -> None:
        """
        Args:
            ear_threshold: Ortalama EAR bu değerin ALTINDAYSA (strictly
                less than) gözler kapalı kabul edilir.
            max_num_faces: Bir frame içinde işlenecek maksimum yüz sayısı.
            min_detection_confidence: MediaPipe yüz tespit güven eşiği.
            min_tracking_confidence: MediaPipe landmark takip güven eşiği.
            refine_landmarks: Göz/iris çevresindeki landmark hassasiyetini artırır.
            face_mesh: Testlerde sahte FaceMesh enjekte etmek için kullanılır.
                Enjekte edilirse ömrü LandmarkService'e ait DEĞİLDİR — close()
                çağrıldığında kapatılmaz (çağıran taraf kendi yönetir).
        """
        if not 0.0 < ear_threshold < 1.0:
            raise ValueError("ear_threshold 0.0 ile 1.0 arasında olmalıdır.")

        if max_num_faces <= 0:
            raise ValueError("max_num_faces sıfırdan büyük olmalıdır.")

        self._validate_probability("min_detection_confidence", min_detection_confidence)
        self._validate_probability("min_tracking_confidence", min_tracking_confidence)

        self._ear_threshold = float(ear_threshold)
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

    @property
    def ear_threshold(self) -> float:
        return self._ear_threshold

    @property
    def is_closed(self) -> bool:
        return self._closed

    def analyze(self, frame: np.ndarray) -> LandmarkResult:
        """
        Bir OpenCV frame'ini analiz eder.

        Args:
            frame: OpenCV tarafından BGR formatında sağlanan görüntü.

        Returns:
            Yüz, göz ve EAR ölçümlerini içeren LandmarkResult.
        """
        self._validate_frame(frame)

        if self._closed:
            raise RuntimeError("Kapatılmış LandmarkService tekrar kullanılamaz.")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # MediaPipe bazı sürümlerde yazılabilir olmayan görüntüyle daha
        # verimli çalışabilir (kendi örnek kodlarında önerilen pattern).
        rgb_frame.flags.writeable = False

        output = self._face_mesh.process(rgb_frame)
        face_landmarks_collection = getattr(output, "multi_face_landmarks", None)

        if not face_landmarks_collection:
            return self._no_face_result()

        landmarks = face_landmarks_collection[0].landmark
        height, width = frame.shape[:2]

        left_eye_points = self._extract_eye_points(
            landmarks=landmarks, indices=LEFT_EYE_INDICES, frame_width=width, frame_height=height
        )
        right_eye_points = self._extract_eye_points(
            landmarks=landmarks, indices=RIGHT_EYE_INDICES, frame_width=width, frame_height=height
        )

        left_ear = self.calculate_ear(left_eye_points)
        right_ear = self.calculate_ear(right_eye_points)
        average_ear = (left_ear + right_ear) / 2.0

        eyes_closed = average_ear < self._ear_threshold
        confidence = self._calculate_eye_state_confidence(ear=average_ear, eyes_closed=eyes_closed)

        return LandmarkResult(
            face_detected=True,
            eyes_closed=eyes_closed,
            confidence=round(confidence, 4),
            ear=round(average_ear, 4),
            left_ear=round(left_ear, 4),
            right_ear=round(right_ear, 4),
            left_eye_points=left_eye_points,
            right_eye_points=right_eye_points,
        )

    @staticmethod
    def calculate_ear(eye_points: Sequence[Point]) -> float:
        """
        Altı göz noktası üzerinden Eye Aspect Ratio hesaplar.

            EAR = (|p2-p6| + |p3-p5|) / (2 x |p1-p4|)

        Göz kapandıkça dikey mesafeler küçülür ve EAR düşer.
        """
        if len(eye_points) != 6:
            raise ValueError("EAR hesabı için tam olarak 6 nokta gereklidir.")

        p1, p2, p3, p4, p5, p6 = eye_points

        vertical_1 = LandmarkService._distance(p2, p6)
        vertical_2 = LandmarkService._distance(p3, p5)
        horizontal = LandmarkService._distance(p1, p4)

        if horizontal <= 1e-8:
            raise ValueError("Gözün yatay mesafesi sıfır olamaz.")

        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def draw_eye_landmarks(self, frame: np.ndarray, result: LandmarkResult) -> np.ndarray:
        """Göz noktalarını ve EAR değerini frame üzerine çizer (yerinde
        değiştirir, aynı frame nesnesini döndürür)."""
        self._validate_frame(frame)

        if not result.face_detected:
            return frame

        for point in (*result.left_eye_points, *result.right_eye_points):
            cv2.circle(frame, (int(point.x), int(point.y)), 2, (0, 255, 255), -1, cv2.LINE_AA)

        if result.ear is not None:
            cv2.putText(
                frame,
                f"EAR: {result.ear:.3f}",
                (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return frame

    def close(self) -> None:
        """MediaPipe kaynaklarını serbest bırakır. Enjekte edilmiş (dışarıdan
        verilmiş) bir face_mesh'i KAPATMAZ — o kaynağın sahibi bu sınıf
        değildir."""
        if self._closed:
            return

        if self._owns_face_mesh:
            self._face_mesh.close()

        self._closed = True

    def __enter__(self) -> LandmarkService:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def _calculate_eye_state_confidence(self, *, ear: float, eyes_closed: bool) -> float:
        """
        EAR değerinin eşikten uzaklığına göre yaklaşık karar güveni üretir.
        MediaPipe'ın model confidence skoru değildir.
        """
        if eyes_closed:
            distance_ratio = (self._ear_threshold - ear) / self._ear_threshold
        else:
            distance_ratio = (ear - self._ear_threshold) / self._ear_threshold

        return min(1.0, max(0.0, distance_ratio))

    @staticmethod
    def _extract_eye_points(
        *, landmarks: Sequence[Any], indices: Sequence[int], frame_width: int, frame_height: int
    ) -> tuple[Point, ...]:
        """Normalize landmark koordinatlarını piksel koordinatlarına çevirir."""
        points: list[Point] = []

        for index in indices:
            if index >= len(landmarks):
                raise ValueError(f"Landmark index'i mevcut değil: {index}")

            landmark = landmarks[index]
            points.append(Point(x=float(landmark.x) * frame_width, y=float(landmark.y) * frame_height))

        return tuple(points)

    @staticmethod
    def _distance(first: Point, second: Point) -> float:
        return math.hypot(first.x - second.x, first.y - second.y)

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame bir numpy.ndarray olmalıdır.")

        if frame.ndim != 3:
            raise ValueError("frame üç boyutlu BGR görüntüsü olmalıdır.")

        if frame.shape[2] != 3:
            raise ValueError("frame tam olarak 3 renk kanalına sahip olmalıdır.")

        if frame.size == 0:
            raise ValueError("frame boş olamaz.")

    @staticmethod
    def _validate_probability(name: str, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} sayısal olmalıdır.")

        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} 0.0 ile 1.0 arasında olmalıdır.")

    @staticmethod
    def _no_face_result() -> LandmarkResult:
        return LandmarkResult(
            face_detected=False,
            eyes_closed=False,
            confidence=0.0,
            ear=None,
            left_ear=None,
            right_ear=None,
            left_eye_points=(),
            right_eye_points=(),
        )
