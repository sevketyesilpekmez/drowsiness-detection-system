"""
Gerçek bir webcam ile YOLO+PyTorch tespit hattının uçtan uca smoke test'i.

    OpenCV webcam -> UltralyticsPersonDetector (YOLOv8n)
                  -> EyeRegionLocator (Haar cascade, üst-yüz kısıtlı)
                  -> TorchEyeClassifier (PyTorch CNN, weights/eye_classifier.pth)
                  -> EyeStateSmoother (zamansal yumuşatma)
                  -> YoloDrowsinessDetector (hepsini birleştiren detect_fn)
                  -> ekranda kutular + konsolda state/raw_closed/smoothed_closed/confidence

Bu script Flask/DetectionEngine'i kullanmaz — backend'in gerçek tespit
servislerini (app/__init__.py::_try_build_yolo_pytorch_detect_fn ile aynı
şekilde) doğrudan, izole bir pencerede test etmek için. Web entegrasyonu
zaten kurulu (Flask + DetectionEngine + React); bu script yalnızca
"kamera önünde gerçekten doğru çalışıyor mu" sorusuna hızlı ve görsel bir
cevap vermek amacıyla var.

Kullanım (backend/ dizininde, torch+ultralytics kurulu venv'inde):

    python scripts/live_camera_smoke_test.py

Çıkmak için pencerede 'q' tuşuna bas, ya da terminalde Ctrl+C — her iki
durumda da çıkışta "--- Özet ---" başlıklı bir istatistik bloğu basılır
(toplam kare, ortalama eyes_examined, raw/smoothed_closed oranları,
DROWSY/NO_FACE kare oranı). Bu, eye_search_height_ratio gibi parametreleri
farklı değerlerle kıyaslarken logu elle okumak yerine somut sayılara
bakmayı sağlar, örn.:

    for r in 0.55 0.60 0.65 0.70 0.75; do
        python scripts/live_camera_smoke_test.py --eye-search-height-ratio $r
        # ~30 saniye aynı pozisyonda bekle, 'q' ile çık, özeti not al
    done

Diğer argümanlar için: python scripts/live_camera_smoke_test.py --help
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# backend/ kökünü sys.path'e ekle ki `app...` import edilebilsin — bu
# script backend/scripts/ altında, backend/ altında değil.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera-index", type=int, default=0, help="cv2.VideoCapture kamera index'i")
    parser.add_argument("--yolo-weights", default="yolov8n.pt", help="Ultralytics ağırlık yolu/adı")
    parser.add_argument("--yolo-confidence", type=float, default=0.5, help="YOLO kişi tespiti güven eşiği")
    parser.add_argument(
        "--eye-weights",
        default="weights/eye_classifier.pth",
        help="train_eye_classifier.py'nin ürettiği checkpoint",
    )
    parser.add_argument(
        "--drowsiness-seconds",
        type=float,
        default=2.0,
        help="DROWSY için gereken kesintisiz kapalı süre",
    )
    parser.add_argument(
        "--closed-grace-period",
        type=float,
        default=0.3,
        help="Kapalı seri sürerken tolere edilecek tek karelik açık okuması (sn)",
    )
    parser.add_argument(
        "--eye-search-height-ratio", type=float, default=0.70, help="Kişi kutusunun üstten bu oranı taranır"
    )
    parser.add_argument(
        "--smoother-window-seconds",
        type=float,
        default=0.9,
        help="EyeStateSmoother zaman penceresi (saniye)",
    )
    parser.add_argument("--smoother-closed-ratio-threshold", type=float, default=0.6)
    parser.add_argument("--smoother-missing-tolerance-seconds", type=float, default=0.25)
    parser.add_argument("--smoother-minimum-valid-seconds", type=float, default=0.35)
    parser.add_argument("--single-eye-confidence-threshold", type=float, default=0.80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Lazy import: torch/ultralytics gerektiren servisler yalnızca burada
    # import edilir (app/__init__.py'deki aynı desen).
    from app.services.drowsiness_service import DrowsinessService
    from app.services.eye_classifier_service import EyeClassifierService
    from app.services.eye_region_locator import EyeRegionLocator
    from app.services.eye_state_smoother import EyeStateSmoother
    from app.services.torch_eye_classifier import TorchEyeClassifier
    from app.services.ultralytics_person_detector import UltralyticsPersonDetector
    from app.services.yolo_drowsiness_detector import YoloDrowsinessDetector

    print(f"YOLO ağırlığı yükleniyor: {args.yolo_weights}")
    person_detector = UltralyticsPersonDetector(
        weights_path=args.yolo_weights,
        confidence_threshold=args.yolo_confidence,
    )
    print("YOLO model yüklendi.")

    print(f"Göz sınıflandırıcı ağırlığı yükleniyor: {args.eye_weights}")
    eye_classifier_model = TorchEyeClassifier(weights_path=args.eye_weights)
    print("Göz sınıflandırıcı yüklendi.")

    detector = YoloDrowsinessDetector(
        person_detector=person_detector,
        eye_locator=EyeRegionLocator(eye_search_height_ratio=args.eye_search_height_ratio),
        eye_classifier=EyeClassifierService(model=eye_classifier_model),
        drowsiness_service=DrowsinessService(
            drowsiness_seconds=args.drowsiness_seconds,
            closed_grace_period=args.closed_grace_period,
        ),
        eye_state_smoother=EyeStateSmoother(
            window_seconds=args.smoother_window_seconds,
            closed_ratio_threshold=args.smoother_closed_ratio_threshold,
            missing_tolerance_seconds=args.smoother_missing_tolerance_seconds,
            minimum_valid_seconds=args.smoother_minimum_valid_seconds,
        ),
        single_eye_confidence_threshold=args.single_eye_confidence_threshold,
        draw_boxes=True,
    )

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"HATA: Kamera açılamadı (index={args.camera_index}).")
        return

    print("Kamera açıldı. Çıkmak için pencerede 'q' tuşuna bas.")

    # Farklı --eye-search-height-ratio (ve diğer) değerlerini kıyaslarken
    # logu elle okumak yerine somut sayılarla karşılaştırabilmek için her
    # karede sayaçlar tutulur, çıkışta bir özet basılır.
    stats = {
        "frames": 0,
        "eyes_examined_sum": 0,
        "eyes_examined_0": 0,
        "eyes_examined_1": 0,
        "eyes_examined_2": 0,
        "raw_closed_true": 0,
        "smoothed_closed_true": 0,
        "drowsy_frames": 0,
        "no_face_frames": 0,
    }

    last_print = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("HATA: Kare okunamadı.")
                break

            result = detector(frame)

            stats["frames"] += 1
            examined = result.get("eyes_examined", 0)
            stats["eyes_examined_sum"] += examined
            if examined == 0:
                stats["eyes_examined_0"] += 1
            elif examined == 1:
                stats["eyes_examined_1"] += 1
            else:
                stats["eyes_examined_2"] += 1
            if result.get("raw_eyes_closed", False):
                stats["raw_closed_true"] += 1
            if result.get("smoothed_eyes_closed", False):
                stats["smoothed_closed_true"] += 1
            if result["state"] == "DROWSY":
                stats["drowsy_frames"] += 1
            if result["state"] == "NO_FACE":
                stats["no_face_frames"] += 1

            # Üç farklı "kapalı" kavramı var:
            #   raw_closed:       o karenin ham sınıflandırma kararı (tek kareye dayanır).
            #   smoothed_closed:  EyeStateSmoother'ın son penceredeki
            #                     çoğunluğuna göre verdiği oy, nihai karar değil.
            #   effective_closed: DrowsinessService'in closed_grace_period
            #                     toleransını da uyguladıktan sonraki fiili kararı;
            #                     state/closed_duration buna göre hesaplanır. Bu
            #                     yüzden smoothed_closed=False iken effective_closed=
            #                     True görülebilir — grace_period'un kısa bir
            #                     "açık" okumasını tolere ettiği an (bkz. README).
            #
            # Konsola saniyede ~2 kez yaz, her karede basmak terminali okunmaz hale getirir.
            now = time.monotonic()
            if now - last_print > 0.5:
                print(
                    f"state={result['state']:<8} "
                    f"raw_closed={result.get('raw_eyes_closed', result['eyes_closed'])!s:<5} "
                    f"smoothed_closed={result.get('smoothed_eyes_closed', result['eyes_closed'])!s:<5} "
                    f"effective_closed={result['eyes_closed']!s:<5} "
                    f"eyes_examined={result.get('eyes_examined', 0)} "
                    f"confidence={result['confidence']:.2f} "
                    f"closed_duration={result['closed_duration']:.2f}s"
                )
                last_print = now

            raw_label = "ON" if result.get("raw_eyes_closed", result["eyes_closed"]) else "OFF"
            smoothed_label = "ON" if result.get("smoothed_eyes_closed", result["eyes_closed"]) else "OFF"
            effective_label = "ON" if result["eyes_closed"] else "OFF"
            overlay_lines = [
                f"State: {result['state']}",
                f"raw closed: {raw_label}",
                f"smoothed closed: {smoothed_label}",
                f"effective closed: {effective_label}",
                f"closed duration: {result['closed_duration']:.2f}s",
            ]
            for i, line in enumerate(overlay_lines):
                cv2.putText(
                    frame,
                    line,
                    (20, 40 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("YOLO+PyTorch Uyku Tespit - Smoke Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        _print_summary(stats, args)
        print("Kapatıldı.")


def _print_summary(stats: dict, args: argparse.Namespace) -> None:
    """Farklı --eye-search-height-ratio (vb.) değerlerini kıyaslarken log'u
    elle okumak yerine somut sayılarla karşılaştırabilmek için çalıştırma
    sonunda basılan özet (bkz. README)."""
    frames = stats["frames"]
    if frames == 0:
        print("\n(Özet için hiç kare işlenmedi.)")
        return

    avg_examined = stats["eyes_examined_sum"] / frames
    print("\n--- Özet ---")
    print(f"eye_search_height_ratio = {args.eye_search_height_ratio}")
    print(f"toplam kare: {frames}")
    print(
        f"ortalama eyes_examined: {avg_examined:.2f}  "
        f"(0 göz: %{100 * stats['eyes_examined_0'] / frames:.1f}, "
        f"1 göz: %{100 * stats['eyes_examined_1'] / frames:.1f}, "
        f"2 göz: %{100 * stats['eyes_examined_2'] / frames:.1f})"
    )
    print(
        f"raw_closed=True oranı: %{100 * stats['raw_closed_true'] / frames:.1f}  "
        f"smoothed_closed=True oranı: %{100 * stats['smoothed_closed_true'] / frames:.1f}"
    )
    print(
        f"DROWSY karesi: %{100 * stats['drowsy_frames'] / frames:.1f}  "
        f"NO_FACE karesi: %{100 * stats['no_face_frames'] / frames:.1f}"
    )


if __name__ == "__main__":
    main()
