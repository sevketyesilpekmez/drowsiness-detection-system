"""
Değerlendirme klipleri için etiketleme aracı.

Videoyu kare kare gezip göz kapanmalarını işaretler ve
`benchmark/labels/` altına değerlendirme koşusunun okuyabileceği JSON
üretir.

Bu dosya BİLEREK İNCE tutuldu: tüm karar mantığı
`evaluation/label_editor.py` içinde ve tam olarak test edilmiş
durumda. Buradaki kod yalnızca "kareyi göster, tuşu oku, editöre
söyle" yapar. Etiketleme hatası ölçüm sonucunu sessizce bozduğu için
mantığın GUI'ye gömülmesi kabul edilebilir değildi.

Kullanım:

    cd backend
    python scripts/label_clips.py ../benchmark/clips/glasses_01.mp4 \\
        --labels-dir ../benchmark/labels \\
        --subject S03 --conditions glasses daylight

Mevcut bir etiket dosyası varsa `--resume` ile üzerine devam edilebilir.

Tuşlar:

    boşluk    oynat / duraklat
    j / l     bir kare geri / ileri  (ok tuşları da denenebilir)
    , / .     bir saniye geri / ileri
    a / d     beş saniye geri / ileri
    c         olay BAŞLANGICI işaretle
    o         olay BİTİŞİ işaretle
    b         son olayı çeldirici (göz kırpma) yap / geri al
    u         son işlemi geri al
    s         kaydet
    q / ESC   çık (kaydedilmemiş değişiklik varsa sorar)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.ground_truth import load_clip_labels  # noqa: E402
from evaluation.label_editor import EditableEvent, LabelEditor  # noqa: E402
from evaluation.video_frames import probe_video  # noqa: E402

WINDOW_NAME = "Etiketleme"
MAX_DISPLAY_WIDTH = 1100

# Zaman çizelgesi şeridinin yüksekliği (piksel).
TIMELINE_HEIGHT = 46
HEADER_HEIGHT = 78

COLOR_BG = (28, 28, 32)
COLOR_TEXT = (235, 235, 235)
COLOR_DIM = (150, 150, 155)
COLOR_EVENT = (80, 200, 120)
COLOR_DISTRACTOR = (60, 165, 235)
COLOR_PENDING = (70, 200, 240)
COLOR_CURSOR = (240, 240, 240)
COLOR_WARN = (90, 180, 250)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("clip", help="Etiketlenecek video dosyası")
    parser.add_argument(
        "--labels-dir",
        default=None,
        help="JSON'un yazılacağı klasör (varsayılan: videonun yanındaki ../labels)",
    )
    parser.add_argument("--subject", default=None, help="Denek kimliği, ör. S03")
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=[],
        metavar="KOSUL",
        help="Koşul etiketleri, ör. glasses lowlight",
    )
    parser.add_argument("--notes", default=None, help="Serbest not")
    parser.add_argument(
        "--drowsiness-seconds",
        type=float,
        default=2.5,
        help="Uyarılar için alarm eşiği (kısa olayları işaretler).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Varsa mevcut etiket dosyasını yükleyip üzerine devam et.",
    )
    return parser.parse_args()


def default_labels_dir(clip_path: Path) -> Path:
    return clip_path.parent.parent / "labels"


def label_path_for(clip_path: Path, labels_dir: Path) -> Path:
    return labels_dir / f"{clip_path.stem}.json"


def build_editor(args: argparse.Namespace, clip_path: Path, labels_dir: Path) -> LabelEditor:
    info = probe_video(clip_path)
    editor = LabelEditor(
        clip=clip_path.name,
        fps=info.fps,
        duration_s=info.duration_s,
        subject=args.subject,
        conditions=tuple(args.conditions),
        notes=args.notes,
    )

    target = label_path_for(clip_path, labels_dir)
    if args.resume and target.exists():
        existing = load_clip_labels(target)
        editor.events = [
            EditableEvent(
                start_s=e.start_s, end_s=e.end_s, expect_alarm=e.expect_alarm, label=e.label
            )
            for e in existing.events
        ]
        editor.subject = editor.subject or existing.subject
        editor.conditions = editor.conditions or existing.conditions
        editor.notes = editor.notes or existing.notes
        print(f"{target} yüklendi: {len(editor.events)} olay.")

    return editor


def save(editor: LabelEditor, path: Path) -> str:
    """Kaydetmeden ÖNCE gerçek yükleyiciden doğrular — bozuk bir etiket
    dosyası diske hiç yazılmaz."""
    error = editor.validate()
    if error is not None:
        return f"KAYDEDİLMEDİ, doğrulama hatası: {error}"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(editor.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return f"Kaydedildi: {path}"


def format_time(seconds: float) -> str:
    minutes, secs = divmod(max(0.0, seconds), 60)
    return f"{int(minutes):02d}:{secs:05.2f}"


def draw_timeline(canvas, editor: LabelEditor, current_time_s: float, width: int, top: int) -> None:
    """Olayları ve imleci gösteren yatay şerit."""
    import cv2

    duration = max(editor.duration_s, 1e-6)
    baseline = top + TIMELINE_HEIGHT // 2

    cv2.rectangle(canvas, (0, top), (width, top + TIMELINE_HEIGHT), (44, 44, 50), -1)
    cv2.line(canvas, (0, baseline), (width, baseline), (70, 70, 78), 1)

    for event in editor.events:
        x1 = int(event.start_s / duration * width)
        x2 = max(x1 + 2, int(event.end_s / duration * width))
        color = COLOR_EVENT if event.expect_alarm else COLOR_DISTRACTOR
        cv2.rectangle(canvas, (x1, baseline - 11), (x2, baseline + 11), color, -1)

    if editor.has_pending:
        x1 = int(editor.pending_start_s / duration * width)
        x2 = int(current_time_s / duration * width)
        cv2.rectangle(canvas, (min(x1, x2), baseline - 7), (max(x1, x2), baseline + 7), COLOR_PENDING, 1)
        cv2.line(canvas, (x1, top + 4), (x1, top + TIMELINE_HEIGHT - 4), COLOR_PENDING, 2)

    cursor_x = int(current_time_s / duration * width)
    cv2.line(canvas, (cursor_x, top), (cursor_x, top + TIMELINE_HEIGHT), COLOR_CURSOR, 1)


def render(frame, editor: LabelEditor, *, frame_index: int, time_s: float, message: str, playing: bool):
    """Kare + üst bilgi + zaman çizelgesinden tek bir görüntü üretir."""
    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    if width > MAX_DISPLAY_WIDTH:
        scale = MAX_DISPLAY_WIDTH / width
        frame = cv2.resize(frame, (MAX_DISPLAY_WIDTH, int(height * scale)))
        height, width = frame.shape[:2]

    canvas = np.full((HEADER_HEIGHT + height + TIMELINE_HEIGHT, width, 3), COLOR_BG, dtype=np.uint8)
    canvas[HEADER_HEIGHT : HEADER_HEIGHT + height, 0:width] = frame

    font = cv2.FONT_HERSHEY_SIMPLEX
    state = "OYNUYOR" if playing else "DURDU"
    cv2.putText(
        canvas,
        f"{format_time(time_s)} / {format_time(editor.duration_s)}  kare {frame_index}  [{state}]",
        (12, 24),
        font,
        0.58,
        COLOR_TEXT,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(canvas, editor.status_line(), (12, 46), font, 0.5, COLOR_DIM, 1, cv2.LINE_AA)
    if message:
        cv2.putText(canvas, message, (12, 68), font, 0.5, COLOR_WARN, 1, cv2.LINE_AA)

    draw_timeline(canvas, editor, time_s, width, HEADER_HEIGHT + height)
    return canvas


def main() -> int:
    import cv2

    args = parse_args()
    clip_path = Path(args.clip)
    if not clip_path.exists():
        raise SystemExit(f"Video bulunamadı: {clip_path}")

    labels_dir = Path(args.labels_dir) if args.labels_dir else default_labels_dir(clip_path)
    editor = build_editor(args, clip_path, labels_dir)
    target = label_path_for(clip_path, labels_dir)

    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        raise SystemExit(f"Video açılamadı: {clip_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = editor.fps
    frame_index = 0
    playing = False
    message = "j/l kare · ,/. saniye · c basla · o bitir · b celdirici · u geri · s kaydet · q cik"
    dirty = False

    print(__doc__)

    def read_at(index: int):
        index = max(0, min(index, max(0, total_frames - 1)))
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        return (index, frame) if ok and frame is not None else (index, None)

    frame_index, frame = read_at(0)
    if frame is None:
        raise SystemExit("Videodan kare okunamadı.")

    while True:
        time_s = frame_index / fps
        cv2.imshow(WINDOW_NAME, render(
            frame, editor, frame_index=frame_index, time_s=time_s, message=message, playing=playing
        ))

        # Oynatma sırasında kare süresi kadar, duraklatıldığında tuşa
        # kadar bekle (boşuna CPU yakmamak için).
        key = cv2.waitKey(max(1, int(1000 / fps)) if playing else 30) & 0xFF

        if key == 255:  # tuş yok
            if playing:
                ok, next_frame = capture.read()
                if ok and next_frame is not None:
                    frame_index += 1
                    frame = next_frame
                else:
                    playing = False
                    message = "Klip sonu."
            continue

        if key in (ord("q"), 27):
            if dirty:
                message = "Kaydedilmemiş değişiklik var. Kaydetmek için 's', yine de çıkmak için tekrar 'q'."
                dirty = False
                continue
            break

        if key == ord(" "):
            playing = not playing
            if playing:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index + 1)
            continue

        playing = False
        step = None

        # Ok tuşlarının cv2.waitKey kodu platforma göre değişiyor
        # (Linux/X11: 81/83, macOS/Cocoa: 63234/63235 -> & 0xFF sonrası 2/3).
        # Harf tuşları HER PLATFORMDA aynı, bu yüzden j/l birincil.
        if key in (ord("j"), 81, 2):
            step = -1
        elif key in (ord("l"), 83, 3):
            step = 1
        elif key == ord(","):
            step = -int(fps)
        elif key == ord("."):
            step = int(fps)
        elif key == ord("a"):
            step = -int(fps * 5)
        elif key == ord("d"):
            step = int(fps * 5)

        if step is not None:
            frame_index, new_frame = read_at(frame_index + step)
            if new_frame is not None:
                frame = new_frame
            continue

        if key == ord("c"):
            message = editor.begin_event(time_s)
            dirty = True
        elif key == ord("o"):
            message = editor.end_event(time_s)
            dirty = True
        elif key == ord("b"):
            message = editor.toggle_last_distractor()
            dirty = True
        elif key == ord("u"):
            message = editor.undo()
            dirty = True
        elif key == ord("s"):
            message = save(editor, target)
            if message.startswith("Kaydedildi"):
                dirty = False
                for warning in editor.warnings(drowsiness_seconds=args.drowsiness_seconds):
                    print(f"  uyarı: {warning}")

    capture.release()
    cv2.destroyAllWindows()

    if dirty:
        print("UYARI: kaydedilmemiş değişiklikler vardı.")

    print(f"{editor.expected_event_count} olay, {editor.distractor_count} çeldirici.")
    for warning in editor.warnings(drowsiness_seconds=args.drowsiness_seconds):
        print(f"  uyarı: {warning}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
