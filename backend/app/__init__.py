import atexit
import logging
import time
from collections.abc import Callable
from typing import Any

from flask import Flask
from flask_cors import CORS

from app.api.detection_routes import detection_bp
from app.api.health_routes import health_bp
from app.api.stream_routes import stream_bp
from app.config import Config
from app.core.detection_engine import DetectFn, DetectionEngine, dummy_detect_fn
from app.core.shared_state import SharedState
from app.services.alarm_service import AlarmService
from app.services.drowsiness_detector import DrowsinessDetector
from app.services.drowsiness_service import DrowsinessService
from app.services.eye_classifier_service import EyeClassifierService
from app.services.eye_region_locator import EyeRegionLocator
from app.services.eye_state_smoother import EyeStateSmoother
from app.services.fallback_eye_locator import FallbackEyeLocator
from app.services.landmark_service import LandmarkService
from app.services.mediapipe_eye_locator import MediaPipeEyeLocator
from app.services.video_source import VideoSource
from app.services.yolo_drowsiness_detector import YoloDrowsinessDetector

logger = logging.getLogger(__name__)


Clock = Callable[[], float]


def _build_mediapipe_detect_fn(app: Flask, *, clock: Clock = time.monotonic) -> DetectFn:
    """MediaPipe Face Mesh + EAR tabanlı yedek/geliştirme backend'i.
    Ağırlık eğitimi gerektirmez, her zaman çalışır."""
    landmark_service = LandmarkService(ear_threshold=app.config["EAR_THRESHOLD"])
    drowsiness_service = DrowsinessService(
        drowsiness_seconds=app.config["DROWSINESS_SECONDS"],
        closed_grace_period=app.config["CLOSED_EYES_GRACE_PERIOD_SECONDS"],
        clock=clock,
    )
    alarm_service = AlarmService(cooldown_seconds=app.config["ALARM_COOLDOWN_SECONDS"], clock=clock)
    return DrowsinessDetector(
        landmark_service=landmark_service,
        drowsiness_service=drowsiness_service,
        alarm_service=alarm_service,
        draw_landmarks=app.config["DRAW_LANDMARKS"],
    )


def _try_build_yolo_pytorch_detect_fn(
    app: Flask, *, clock: Clock = time.monotonic
) -> tuple[DetectFn | None, str | None]:
    """
    YOLO (kişi) + Haar/MediaPipe (göz bölgesi) + PyTorch (göz sınıflandırma)
    backend'ini kurmayı dener.

    Kurulamazsa (bağımlılık eksik ya da eğitilmiş ağırlık yok) (None,
    hata_mesajı) döner — sonraki adımda ne yapılacağına (mediapipe'a
    düşmek mi, MODEL_ERROR'da kalmak mı) burada karar verilmiyor, bkz.
    _build_detect_fn.
    """
    try:
        # Lazy import: torch/ultralytics kurulu olmayan bir ortamda bile
        # (mediapipe'a düşüldüğünde) bu modül sorunsuz import edilebilsin.
        from app.services.torch_eye_classifier import TorchEyeClassifier
        from app.services.ultralytics_person_detector import UltralyticsPersonDetector

        person_detector = UltralyticsPersonDetector(
            weights_path=app.config["YOLO_MODEL_PATH"],
            confidence_threshold=app.config["YOLO_CONFIDENCE_THRESHOLD"],
            device=app.config["TORCH_DEVICE"],
            imgsz=app.config["YOLO_IMGSZ"],
        )
        logger.info(
            "YOLO model loaded: %s (device=%s, imgsz=%s)",
            app.config["YOLO_MODEL_PATH"],
            app.config["TORCH_DEVICE"],
            app.config["YOLO_IMGSZ"],
        )

        eye_classifier_model = TorchEyeClassifier(
            weights_path=app.config["EYE_CLASSIFIER_WEIGHTS_PATH"],
            device=app.config["TORCH_DEVICE"],
        )
        logger.info("Eye classifier weights loaded: %s", app.config["EYE_CLASSIFIER_WEIGHTS_PATH"])
    except (ImportError, RuntimeError, FileNotFoundError, OSError) as exc:
        error_message = (
            f"{type(exc).__name__}: {exc}. Çözüm için: 1) requirements.txt'teki "
            f"ultralytics/torch'u kur, 2) scripts/train_eye_classifier.py ile bir "
            f"göz sınıflandırıcı eğitip {app.config['EYE_CLASSIFIER_WEIGHTS_PATH']} "
            f"yoluna kaydet."
        )
        logger.error("DETECTION_BACKEND=yolo_pytorch istendi ama kurulamadı: %s", error_message)
        return None, error_message

    haar_eye_locator = EyeRegionLocator(
        eye_search_height_ratio=app.config["EYE_SEARCH_HEIGHT_RATIO"],
        adaptive_min_size_ratio=(
            (
                app.config["EYE_ADAPTIVE_MIN_SIZE_WIDTH_RATIO"],
                app.config["EYE_ADAPTIVE_MIN_SIZE_HEIGHT_RATIO"],
            )
            if app.config["EYE_ADAPTIVE_MIN_SIZE_ENABLED"]
            else None
        ),
        split_left_right=app.config["EYE_SPLIT_LEFT_RIGHT"],
    )

    if app.config["EYE_LOCATOR_BACKEND"] == "mediapipe":
        # MediaPipe birincil, Haar yedek — YOLO ve PyTorch sınıflandırıcı
        # bundan etkilenmiyor, yalnızca "göz nerede" adımı değişiyor.
        eye_locator = FallbackEyeLocator(
            primary=MediaPipeEyeLocator(eye_padding_ratio=app.config["EYE_LANDMARK_PADDING_RATIO"]),
            fallback=haar_eye_locator,
        )
        logger.info(
            "Göz bulucu: mediapipe (Face Mesh, Haar yedek) — padding=%s",
            app.config["EYE_LANDMARK_PADDING_RATIO"],
        )
    else:
        eye_locator = haar_eye_locator
        logger.info("Göz bulucu: haar (yalnızca Haar cascade)")

    eye_classifier = EyeClassifierService(model=eye_classifier_model)
    drowsiness_service = DrowsinessService(
        drowsiness_seconds=app.config["DROWSINESS_SECONDS"],
        closed_grace_period=app.config["CLOSED_EYES_GRACE_PERIOD_SECONDS"],
        # Yalnızca bu backend measurement_valid raporluyor; mediapipe/EAR
        # backend'i her zaman ölçüm aldığını varsayar (bkz. LandmarkService).
        measurement_grace_seconds=app.config["EYE_MEASUREMENT_GRACE_SECONDS"],
        clock=clock,
    )
    alarm_service = AlarmService(cooldown_seconds=app.config["ALARM_COOLDOWN_SECONDS"], clock=clock)
    eye_state_smoother = EyeStateSmoother(
        window_seconds=app.config["EYE_SMOOTHER_WINDOW_SECONDS"],
        closed_ratio_threshold=app.config["EYE_SMOOTHER_CLOSED_RATIO_THRESHOLD"],
        missing_tolerance_seconds=app.config["EYE_SMOOTHER_MISSING_TOLERANCE_SECONDS"],
        minimum_valid_seconds=app.config["EYE_SMOOTHER_MINIMUM_VALID_SECONDS"],
        # Smoother artık zaman tabanlı, yani DrowsinessService/AlarmService
        # ile AYNI saati kullanmak zorunda — offline değerlendirmede sanal
        # saat enjekte edildiğinde üçü de aynı zaman ekseninde olmalı.
        clock=clock,
    )

    detector = YoloDrowsinessDetector(
        person_detector=person_detector,
        eye_locator=eye_locator,
        eye_classifier=eye_classifier,
        drowsiness_service=drowsiness_service,
        eye_state_smoother=eye_state_smoother,
        alarm_service=alarm_service,
        draw_boxes=app.config["DRAW_LANDMARKS"],
        single_eye_confidence_threshold=app.config["SINGLE_EYE_CONFIDENCE_THRESHOLD"],
    )
    return detector, None


class _StrictBackendFailure:
    """ALLOW_DETECTION_FALLBACK=false iken yolo_pytorch kurulamazsa
    kullanılan detect_fn — mediapipe'a sessizce düşmek yerine her
    çağrıda aynı hatayı fırlatır, DetectionEngine bunu MODEL_ERROR'a
    çevirir. Sistem farklı bir teknolojiyle "çalışıyormuş gibi" görünmez."""

    def __init__(self, message: str) -> None:
        self._message = message

    def __call__(self, frame: Any) -> dict[str, Any]:
        raise RuntimeError(self._message)

    def close(self) -> None:  # DetectionEngine.stop() varsa çağırır, no-op.
        return None


def _build_detect_fn(app: Flask, *, clock: Clock = time.monotonic) -> DetectFn:
    """
    Gerçek detect_fn'i yalnızca ENABLE_DETECTION_ENGINE=True iken kurar —
    aksi halde her testte (create_app() sık çağrılıyor) ağır modeller
    boşuna yüklenirdi.

    DETECTION_BACKEND=yolo_pytorch kurulamazsa Config.ALLOW_DETECTION_FALLBACK
    devreye girer: true iken mediapipe'a otomatik düşülür, false iken
    (teslim/sunum modu) düşülmez, MODEL_ERROR kalıcı olarak gösterilir.

    AlarmService'e callback verilmiyor — alarm sesi backend'de değil,
    tarayıcıda React tarafında çalıyor.
    """
    if not app.config["ENABLE_DETECTION_ENGINE"]:
        return dummy_detect_fn

    logger.info("Detection backend: %s", app.config["DETECTION_BACKEND"])

    if app.config["DETECTION_BACKEND"] == "mediapipe":
        detector: DetectFn = _build_mediapipe_detect_fn(app, clock=clock)
    else:
        # Config zaten yalnızca yolo_pytorch/mediapipe'a izin veriyor.
        yolo_detector, yolo_error = _try_build_yolo_pytorch_detect_fn(app, clock=clock)
        if yolo_detector is not None:
            detector = yolo_detector
        elif app.config["ALLOW_DETECTION_FALLBACK"]:
            logger.warning(
                "ALLOW_DETECTION_FALLBACK=true olduğu için mediapipe backend'ine "
                "düşülüyor. TESLİM/SUNUM modunda bu İSTENMEZ — "
                "ALLOW_DETECTION_FALLBACK=false yapıp MODEL_ERROR'ın açıkça "
                "gösterilmesini sağlayın (bkz. README 7. bölüm)."
            )
            detector = _build_mediapipe_detect_fn(app, clock=clock)
        else:
            logger.error(
                "ALLOW_DETECTION_FALLBACK=false: mediapipe'a düşülmüyor, "
                "/api/detection/status kalıcı olarak MODEL_ERROR gösterecek. Sebep: %s",
                yolo_error,
            )
            detector = _StrictBackendFailure(
                f"YOLO/PyTorch backend'i başlatılamadı (ALLOW_DETECTION_FALLBACK=false): {yolo_error}"
            )

    app.extensions["drowsiness_detector"] = detector
    # Ayrıca atexit kaydı YOK: DetectionEngine.stop() zaten detect_fn.close()'u
    # (varsa) kendisi çağırıyor, engine.stop() aşağıda tek noktadan atexit'e
    # kaydediliyor.

    return detector


def build_detect_fn_for_evaluation(*, clock: Clock = time.monotonic) -> DetectFn:
    """
    Offline değerlendirme için detect_fn kurar (bkz.
    scripts/evaluate_system.py).

    Flask uygulamasını yalnızca CONFIG KAYNAĞI olarak kullanır: blueprint
    kaydetmez, SharedState oluşturmaz, DetectionEngine kurmaz, kamerayı
    açmaz. Amaç, değerlendirmenin gerçek çalışma zamanıyla AYNI kurulum
    kodunu (YOLO + göz bulucu + sınıflandırıcı + smoother + eşikler)
    kullanması — ayrı bir kurulum yazılsaydı ikisi zamanla birbirinden
    ayrışır ve ölçüm, ölçtüğünü iddia ettiği sistemi ölçmemeye başlardı.

    `clock` ile sanal saat enjekte edilir; böylece süre eşikleri ve alarm
    cooldown'ı video zamanına göre işler, makinenin hızına göre değil.
    """
    app = Flask(__name__)
    app.config.from_object(Config)
    # _build_detect_fn bu bayrağa bakıp gerçek dedektörü kuruyor; kamera
    # yalnızca create_app() içinde açılır, burada değil.
    app.config["ENABLE_DETECTION_ENGINE"] = True

    return _build_detect_fn(app, clock=clock)


def create_app() -> Flask:
    # logging.basicConfig() olmadan logger.info(...) çağrıları (Detection
    # backend/YOLO model loaded/Eye classifier loaded dahil) gunicorn
    # altında tamamen sessiz kalıyordu — Flask dev server'da fark
    # edilmiyordu çünkü debug modu kendi handler'ını otomatik ekliyor.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    app = Flask(__name__)
    app.config.from_object(Config)

    CORS(app, origins=app.config["CORS_ORIGINS"].split(","))

    app.register_blueprint(health_bp, url_prefix="/api")
    app.register_blueprint(detection_bp, url_prefix="/api")
    app.register_blueprint(stream_bp, url_prefix="/api")

    # Global singleton DEĞİL — her create_app() kendi izole SharedState'ini
    # alır, testler birbirini kirletmesin diye.
    shared_state = SharedState()
    app.extensions["shared_state"] = shared_state

    video_source = VideoSource(
        app.config["VIDEO_SOURCE"],
        requested_width=app.config["CAMERA_WIDTH"],
        requested_height=app.config["CAMERA_HEIGHT"],
    )
    detection_engine = DetectionEngine(
        video_source=video_source,
        shared_state=shared_state,
        detect_fn=_build_detect_fn(app),
    )
    app.extensions["detection_engine"] = detection_engine

    # Engine'i koşulsuz başlatmıyoruz — ENABLE_DETECTION_ENGINE varsayılan
    # false, testler bunu hiç açmadığı için create_app() testlerde her
    # zaman kamerasız ve hızlı kalır.
    if app.config["ENABLE_DETECTION_ENGINE"]:
        detection_engine.start()
        atexit.register(detection_engine.stop)

    return app
