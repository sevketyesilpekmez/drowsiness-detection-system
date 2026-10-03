import os

from dotenv import load_dotenv

# .env'i Config sınıfı okunmadan ÖNCE yüklemek lazım, yoksa değerler
# hiç okunmaz.
load_dotenv()


class Config:
    """Ortam değişkenlerinden okunan uygulama ayarları."""

    VIDEO_SOURCE = os.environ.get("VIDEO_SOURCE", "0")
    # Kamera çözünürlüğü — VideoSource.open() bunu gerçek kamera
    # kaynaklarına uygular (video dosyalarına değil).
    CAMERA_WIDTH = int(os.environ.get("CAMERA_WIDTH", "640"))
    CAMERA_HEIGHT = int(os.environ.get("CAMERA_HEIGHT", "480"))

    # torch/ultralytics'in çalışacağı cihaz: cpu / mps (Apple Silicon) /
    # cuda:0.
    TORCH_DEVICE = os.environ.get("TORCH_DEVICE", "cpu")

    # yolo_pytorch: YOLO + Haar/MediaPipe + PyTorch (varsayılan).
    # mediapipe: Face Mesh + EAR, ağırlık eğitimi gerektirmeyen yedek.
    DETECTION_BACKEND = os.environ.get("DETECTION_BACKEND", "yolo_pytorch")
    if DETECTION_BACKEND not in ("yolo_pytorch", "mediapipe"):
        raise ValueError('DETECTION_BACKEND "yolo_pytorch" veya "mediapipe" olmalıdır.')

    YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "yolov8n.pt")
    YOLO_CONFIDENCE_THRESHOLD = float(os.environ.get("YOLO_CONFIDENCE_THRESHOLD", "0.5"))
    # YOLO'nun inference öncesi kareyi ölçeklendireceği kare boyutu.
    YOLO_IMGSZ = int(os.environ.get("YOLO_IMGSZ", "640"))
    if YOLO_IMGSZ <= 0:
        raise ValueError("YOLO_IMGSZ sıfırdan büyük olmalıdır.")
    # scripts/train_eye_classifier.py'nin ürettiği checkpoint'in yolu.
    EYE_CLASSIFIER_WEIGHTS_PATH = os.environ.get("EYE_CLASSIFIER_WEIGHTS_PATH", "weights/eye_classifier.pth")

    # true: yolo_pytorch kurulamazsa mediapipe'a otomatik düşer (geliştirme
    # için). false: düşülmez, MODEL_ERROR gösterilir (teslim/sunum modu).
    ALLOW_DETECTION_FALLBACK = os.environ.get("ALLOW_DETECTION_FALLBACK", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    # Gözler kaç saniye kapalı kalırsa DROWSY olacağı — gerçek kamerayla
    # test edilip 2.5'e ayarlandı.
    DROWSINESS_SECONDS = float(os.environ.get("DROWSINESS_SECONDS", "2.5"))

    # Kapalı-göz serisi sürerken tek karelik bir hatalı "açık"/"yüz yok"
    # okuması bu süre boyunca birikimi sıfırlamaz. 0.0 = eski davranış.
    CLOSED_EYES_GRACE_PERIOD_SECONDS = float(os.environ.get("CLOSED_EYES_GRACE_PERIOD_SECONDS", "0.3"))
    if CLOSED_EYES_GRACE_PERIOD_SECONDS < 0:
        raise ValueError("CLOSED_EYES_GRACE_PERIOD_SECONDS negatif olamaz.")

    # EyeStateSmoother ayarları (yalnızca yolo_pytorch) — tek bir hatalı
    # karenin değil, bir zaman penceresindeki çoğunluğun esas alınması için.
    #
    # Bu ayarlar önce KARE cinsindendi (window_size=7 vb.). Uçtan uca
    # ölçüm, kare biriminin sistemin davranışını makinenin hızına
    # bağladığını gösterdi: 5→30 FPS aralığında alarm gecikmesi 0.79 sn
    # kayıyordu (smoother kapalıyken 0.06 sn). Varsayılanlar, eski kare
    # değerlerinin ~8 FPS'teki (gerçekçi işleme hızı) karşılıklarıdır —
    # yani mevcut davranışı korur, benchmark'a fit edilmiş değildir.
    EYE_SMOOTHER_WINDOW_SECONDS = float(os.environ.get("EYE_SMOOTHER_WINDOW_SECONDS", "0.9"))
    EYE_SMOOTHER_CLOSED_RATIO_THRESHOLD = float(os.environ.get("EYE_SMOOTHER_CLOSED_RATIO_THRESHOLD", "0.6"))
    EYE_SMOOTHER_MISSING_TOLERANCE_SECONDS = float(
        os.environ.get("EYE_SMOOTHER_MISSING_TOLERANCE_SECONDS", "0.25")
    )
    EYE_SMOOTHER_MINIMUM_VALID_SECONDS = float(
        os.environ.get("EYE_SMOOTHER_MINIMUM_VALID_SECONDS", "0.35")
    )

    # Tek göz bulunduğunda, o gözün oyunun geçerli sayılması için gereken
    # minimum güven.
    SINGLE_EYE_CONFIDENCE_THRESHOLD = float(os.environ.get("SINGLE_EYE_CONFIDENCE_THRESHOLD", "0.80"))

    # Kadrajda kişi varken göz durumunun ÖLÇÜLEMEDİĞİ bir serinin kaç
    # saniyeye kadar tolere edileceği. Aşılırsa state UNVERIFIABLE olur
    # (bkz. drowsiness_service.py). Sürücü yüzünü çevirdiğinde ya da göz
    # crop'ları sınıflandırılamadığında sistemin "uyanık" demeye devam
    # etmesini engelleyen ayar.
    #
    # Eski adı EYE_LOCATOR_MISS_TOLERANCE'tı ve KARE sayıyordu; bir
    # karenin süresi FPS'e bağlı olduğu için aynı değer 30 FPS'te 67 ms,
    # 5 FPS'te 400 ms anlamına geliyordu. Saniye cinsinden tolerans
    # makineden bağımsız aynı davranışı verir.
    EYE_MEASUREMENT_GRACE_SECONDS = float(os.environ.get("EYE_MEASUREMENT_GRACE_SECONDS", "0.35"))
    if EYE_MEASUREMENT_GRACE_SECONDS < 0:
        raise ValueError("EYE_MEASUREMENT_GRACE_SECONDS negatif olamaz.")

    # Haar cascade'in kişi kutusunun yalnızca üst bu oranını taraması —
    # ağzın göz sanılma riskini azaltır.
    EYE_SEARCH_HEIGHT_RATIO = float(os.environ.get("EYE_SEARCH_HEIGHT_RATIO", "0.70"))

    # Düşük çözünürlükte göz tespitini stabilize eden iki opsiyonel ayar —
    # ikisi de varsayılan kapalı.
    EYE_ADAPTIVE_MIN_SIZE_ENABLED = os.environ.get("EYE_ADAPTIVE_MIN_SIZE_ENABLED", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    # minSize'ı sabit piksel yerine arama bölgesinin oranı olarak hesaplar.
    EYE_ADAPTIVE_MIN_SIZE_WIDTH_RATIO = float(os.environ.get("EYE_ADAPTIVE_MIN_SIZE_WIDTH_RATIO", "0.12"))
    EYE_ADAPTIVE_MIN_SIZE_HEIGHT_RATIO = float(os.environ.get("EYE_ADAPTIVE_MIN_SIZE_HEIGHT_RATIO", "0.08"))
    # Arama bölgesini sol/sağ diye ikiye bölüp Haar'ı ayrı çalıştırır.
    EYE_SPLIT_LEFT_RIGHT = os.environ.get("EYE_SPLIT_LEFT_RIGHT", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    # Göz bölgesini hangi yöntemin bulacağı. mediapipe (varsayılan): Face
    # Mesh landmark'ları, baş pozuna dayanıklı, başarısız olursa Haar'a
    # düşer. haar: eski davranış, yalnızca Haar cascade.
    EYE_LOCATOR_BACKEND = os.environ.get("EYE_LOCATOR_BACKEND", "mediapipe")
    if EYE_LOCATOR_BACKEND not in ("mediapipe", "haar"):
        raise ValueError('EYE_LOCATOR_BACKEND "mediapipe" veya "haar" olmalıdır.')

    # MediaPipe'ın ham landmark kutusunu her yöne ne kadar genişleteceği.
    EYE_LANDMARK_PADDING_RATIO = float(os.environ.get("EYE_LANDMARK_PADDING_RATIO", "0.4"))
    if EYE_LANDMARK_PADDING_RATIO < 0:
        raise ValueError("EYE_LANDMARK_PADDING_RATIO negatif olamaz.")

    if EYE_SMOOTHER_WINDOW_SECONDS <= 0:
        raise ValueError("EYE_SMOOTHER_WINDOW_SECONDS sıfırdan büyük olmalıdır.")
    if EYE_SMOOTHER_MISSING_TOLERANCE_SECONDS < 0:
        raise ValueError("EYE_SMOOTHER_MISSING_TOLERANCE_SECONDS negatif olamaz.")
    if EYE_SMOOTHER_MINIMUM_VALID_SECONDS <= 0:
        raise ValueError("EYE_SMOOTHER_MINIMUM_VALID_SECONDS sıfırdan büyük olmalıdır.")
    if EYE_SMOOTHER_MINIMUM_VALID_SECONDS > EYE_SMOOTHER_WINDOW_SECONDS:
        raise ValueError(
            "EYE_SMOOTHER_MINIMUM_VALID_SECONDS, EYE_SMOOTHER_WINDOW_SECONDS'tan büyük olamaz."
        )
    if not 0.0 < EYE_SMOOTHER_CLOSED_RATIO_THRESHOLD <= 1.0:
        raise ValueError("EYE_SMOOTHER_CLOSED_RATIO_THRESHOLD 0 ile 1 arasında olmalıdır (0 hariç).")
    if not 0.0 <= SINGLE_EYE_CONFIDENCE_THRESHOLD <= 1.0:
        raise ValueError("SINGLE_EYE_CONFIDENCE_THRESHOLD 0.0 ile 1.0 arasında olmalıdır.")
    if not 0.0 < EYE_SEARCH_HEIGHT_RATIO <= 1.0:
        raise ValueError("EYE_SEARCH_HEIGHT_RATIO 0 ile 1 arasında olmalıdır (0 hariç, 1 dahil).")

    # mediapipe backend'inin EAR eşiği — CONFIDENCE_THRESHOLD'dan ayrı,
    # farklı bir anlamı var.
    EAR_THRESHOLD = float(os.environ.get("EAR_THRESHOLD", "0.22"))

    # Aynı alarmın en az kaç saniyede bir tekrar tetiklenebileceği.
    ALARM_COOLDOWN_SECONDS = float(os.environ.get("ALARM_COOLDOWN_SECONDS", "5.0"))
    if ALARM_COOLDOWN_SECONDS <= 0:
        raise ValueError("ALARM_COOLDOWN_SECONDS sıfırdan büyük olmalıdır.")

    # Stream'deki JPEG'in üzerine göz noktaları/EAR çizilsin mi.
    DRAW_LANDMARKS = os.environ.get("DRAW_LANDMARKS", "true").lower() in {"1", "true", "yes", "on"}

    # 3000: prod build (nginx). 5173: `npm run dev` (vite).
    CORS_ORIGINS = os.environ.get(
        "CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
    )

    # true olmadan kamera/model hiç açılmaz.
    ENABLE_DETECTION_ENGINE = os.environ.get("ENABLE_DETECTION_ENGINE", "false").lower() == "true"
