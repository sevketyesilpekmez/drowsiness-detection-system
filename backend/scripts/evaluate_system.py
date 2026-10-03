"""
Uçtan uca sistem değerlendirmesi: etiketli videolar üzerinde tespit
oranı, alarm gecikmesi ve yanlış alarm sıklığı ölçer.

Bu script, `train_eye_classifier.py`'nin ölçtüğünden FARKLI bir şeyi
ölçer. Orası "model tek bir göz kırpıntısını doğru sınıflandırıyor mu"
sorusuna cevap verir; burası "denek 10. saniyede gözünü kapattığında
SİSTEM alarm verdi mi, ne kadar gecikmeyle, kaç yanlış alarmla" sorusuna.

Deterministiktir: zaman gerçek saatten değil kare indeksinden türetilir
(bkz. evaluation/clock.py), dolayısıyla aynı girdi her zaman aynı
sonucu verir ve makinenin hızından etkilenmez.

Kullanım:

    cd backend
    python scripts/evaluate_system.py \\
        --clips benchmark/clips \\
        --labels benchmark/labels \\
        --target-fps 8 \\
        --output benchmark/results/run.json

`--target-fps` KOZMETİK DEĞİL: gerçekte sistem 30 FPS'lik görüntüyü
CPU'da ~8 FPS işler, yani kare atlar. Kare atlama, zaman eşiklerinin
kaç ölçümle dolduğunu ve dolayısıyla yumuşatmanın ne kadar veriye
dayandığını değiştirir. Hedeflenen işleme hızını verin.

Parametre taraması:

    python scripts/evaluate_system.py ... --sweep DROWSINESS_SECONDS=1.5,2.0,2.5,3.0
    python scripts/evaluate_system.py ... --sweep target_fps=5,10,15,30

Her değer için tüm klipler yeniden koşulur ve tespit oranı ↔ yanlış
alarm dengesinin nasıl değiştiği tek tabloda raporlanır.

`target_fps` dışındaki tüm adlar Config ayarı olarak ele alınır ve
gerçekten var olup olmadıkları doğrulanır — yazım hatası sessizce
"parametrenin etkisi yok" sonucuna yol açmasın diye.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# backend/ kökünü sys.path'e ekle ki `app.*` ve `evaluation.*` import
# edilebilsin (bu script backend/scripts/ altında).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.clock import FrameClock, frame_step_for_target_fps  # noqa: E402
from evaluation.ground_truth import ClipLabels, load_label_directory  # noqa: E402
from evaluation.metrics import clip_metrics  # noqa: E402
from evaluation.report import (  # noqa: E402
    build_markdown_report,
    group_by_condition,
    group_by_subject,
)
from evaluation.runner import replay, timeline_to_rows  # noqa: E402
from evaluation.video_frames import probe_video, read_frames  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--clips", required=True, help="Video dosyalarının bulunduğu klasör")
    parser.add_argument("--labels", required=True, help="JSON etiket dosyalarının bulunduğu klasör")
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="Simüle edilecek işleme hızı. Verilmezse her kare işlenir (gerçekçi değil).",
    )
    parser.add_argument(
        "--drowsiness-seconds",
        type=float,
        default=None,
        help="Alarm eşiği. Verilmezse config'deki DROWSINESS_SECONDS kullanılır.",
    )
    parser.add_argument(
        "--match-tolerance",
        type=float,
        default=0.5,
        help="Olay bittikten sonra alarmın hâlâ isabet sayılacağı süre (sn).",
    )
    parser.add_argument("--output", default=None, help="Sonuçların yazılacağı JSON dosyası")
    parser.add_argument(
        "--timeline",
        action="store_true",
        help="Kare kare zaman çizelgesini de JSON çıktısına ekler (dosya büyür).",
    )
    parser.add_argument(
        "--sweep",
        default=None,
        metavar="AYAR=v1,v2,v3",
        help=(
            "Bir ayarı birden çok değerle koşturur, ör. "
            "DROWSINESS_SECONDS=1.5,2.0,2.5 veya EYE_SMOOTHER_WINDOW_SECONDS=0.5,0.9,1.4"
        ),
    )
    return parser.parse_args()


def find_clip_path(clips_dir: Path, labels: ClipLabels) -> Path:
    path = clips_dir / labels.clip
    if not path.exists():
        raise SystemExit(
            f"Etikette geçen video bulunamadı: {path}\n"
            f"(etiket dosyasındaki 'clip' alanı: {labels.clip!r})"
        )
    return path


def evaluate_clip(
    *,
    clip_path: Path,
    labels: ClipLabels,
    detect_fn: Any,
    clock: FrameClock,
    target_fps: float | None,
    drowsiness_seconds: float,
    match_tolerance_s: float,
    include_timeline: bool,
) -> dict[str, Any]:
    info = probe_video(clip_path)
    # Etiket dosyasındaki fps, dosyadan okunana göre önceliklidir:
    # bazı konteynerlerde CAP_PROP_FPS yanlış/yuvarlanmış gelir ve tüm
    # zaman hesabı buna dayandığı için elle düzeltilebilmeli.
    source_fps = labels.fps or info.fps
    frame_step = frame_step_for_target_fps(source_fps, target_fps)

    # Her klip için dedektörün iç durumu sıfırlanmalı; aksi halde bir
    # önceki klibin son karelerindeki kapalı-göz serisi bir sonrakine
    # sızar ve klipler birbirinin sonucunu etkiler.
    reset_detector(detect_fn)

    with read_frames(clip_path) as frames:
        # clock ENJEKTE EDİLİYOR: dedektörün içindeki DrowsinessService
        # ve AlarmService süreyi bu saatten okuyor. Ayrı bir saat
        # kullanılırsa onlarınki hiç ilerlemez ve hiçbir alarm üretilmez.
        result = replay(
            frames,
            detect_fn,
            source_fps=source_fps,
            frame_step=frame_step,
            clock=clock,
        )

    duration_s = labels.duration_s or result.duration_s or info.duration_s

    metrics = clip_metrics(
        events=labels.events,
        alarm_times_s=result.alarm_times_s,
        clip_duration_s=duration_s,
        drowsiness_seconds=drowsiness_seconds,
        states=result.states,
        match_tolerance_s=match_tolerance_s,
    )

    row: dict[str, Any] = {
        "clip": labels.clip,
        "subject": labels.subject,
        "conditions": list(labels.conditions),
        "source_fps": source_fps,
        "frame_step": frame_step,
        "effective_fps": source_fps / frame_step,
        "processed_frame_count": result.processed_frame_count,
        "detect_error_count": result.detect_error_count,
        "alarm_times_s": [round(t, 3) for t in result.alarm_times_s],
        "metrics": metrics,
    }

    if include_timeline:
        row["timeline"] = timeline_to_rows(result.entries)

    return row


def reset_detector(detect_fn: Any) -> None:
    """
    Dedektörün biriktirdiği KARAR durumunu temizler.

    `close()` DEĞİL `reset()` çağrılır. close() MediaPipe FaceMesh'i
    kapatır ve dedektör ikinci klipte tamamen çalışmaz hale gelir —
    bu ayrım tam olarak bu script yüzünden eklendi (bkz.
    YoloDrowsinessDetector.reset()).
    """
    reset_fn = getattr(detect_fn, "reset", None)
    if callable(reset_fn):
        reset_fn()


# detect_fn sınıf adından GERÇEKTE kurulan backend'i türetir.
# Config'in ne istediği değil, ne olduğu önemli.
BACKEND_BY_CLASS = {
    "YoloDrowsinessDetector": "yolo_pytorch",
    "DrowsinessDetector": "mediapipe",
    "_StrictBackendFailure": "BASARISIZ",
    "function": "dummy",
}


def active_backend_name(detect_fn: Any) -> str:
    return BACKEND_BY_CLASS.get(type(detect_fn).__name__, type(detect_fn).__name__)


def check_backend(detect_fn: Any) -> str:
    """
    Hangi backend'in GERÇEKTEN kurulduğunu doğrular ve döndürür.

    Bu kontrol olmadan sessiz bir düşüş mümkün: ALLOW_DETECTION_FALLBACK
    varsayılanı true, yani torch/ultralytics kurulu değilse ya da
    eğitilmiş ağırlık bulunamazsa sistem mediapipe/EAR'a düşer ve koşu
    sorunsuz görünür. Ölçüm aracının en tehlikeli hatası budur: yanlış
    sistemin sayıları doğru sistemin sayısı sanılır.
    """
    from app.config import Config

    actual = active_backend_name(detect_fn)
    requested = Config.DETECTION_BACKEND

    if actual == "BASARISIZ":
        raise SystemExit(
            "Dedektör kurulamadı (ALLOW_DETECTION_FALLBACK=false). "
            "Yukarıdaki ERROR satırında sebebi yazıyor."
        )

    if actual != requested:
        print()
        print("!" * 70)
        print(f"UYARI: {requested!r} istendi ama GERÇEKTE {actual!r} kuruldu.")
        print("ALLOW_DETECTION_FALLBACK=true olduğu için sessizce düşülmüş.")
        print("Bu sayılar istediğiniz sistemin sayıları DEĞİL.")
        print("Sebebi görmek için: ALLOW_DETECTION_FALLBACK=false ile tekrar koşun.")
        print("!" * 70)
        print()

    return actual


def run_once(
    *,
    clips_dir: Path,
    all_labels: tuple[ClipLabels, ...],
    target_fps: float | None,
    drowsiness_seconds: float,
    match_tolerance_s: float,
    include_timeline: bool,
) -> dict[str, Any]:
    # Import burada: torch/ultralytics yalnızca gerçekten koşarken
    # gerekli, --help gibi çağrılar bunlar olmadan da çalışmalı.
    from app import build_detect_fn_for_evaluation

    clock = FrameClock(fps=30.0)  # her klipte gerçek fps ile sıfırlanır
    detect_fn = build_detect_fn_for_evaluation(clock=clock)
    backend = check_backend(detect_fn)
    print(f"Aktif backend: {backend}\n")

    clip_rows = []
    for labels in all_labels:
        clip_path = find_clip_path(clips_dir, labels)
        print(f"  {labels.clip} ...", end="", flush=True)
        row = evaluate_clip(
            clip_path=clip_path,
            labels=labels,
            detect_fn=detect_fn,
            clock=clock,
            target_fps=target_fps,
            drowsiness_seconds=drowsiness_seconds,
            match_tolerance_s=match_tolerance_s,
            include_timeline=include_timeline,
        )
        metrics = row["metrics"]
        print(
            f" {metrics['hit_count']}/{metrics['eligible_event_count']} isabet, "
            f"{metrics['false_alarm_count']} yanlış alarm"
        )
        clip_rows.append(row)

    return {
        "clips": clip_rows,
        "conditions": group_by_condition(clip_rows),
        "subjects": group_by_subject(clip_rows),
        "backend": backend,
    }


def parse_sweep(raw: str) -> tuple[str, list[str]]:
    if "=" not in raw:
        raise SystemExit("--sweep biçimi: AYAR=deger1,deger2 (ör. DROWSINESS_SECONDS=2.0,2.5)")

    name, _, values = raw.partition("=")
    parsed = [v.strip() for v in values.split(",") if v.strip()]

    if not parsed:
        raise SystemExit(f"--sweep için değer verilmedi: {raw}")

    return name.strip(), parsed


# `--sweep target_fps=...` bir Config ayarı DEĞİL, runner parametresi:
# os.environ'a yazmak hiçbir işe yaramaz, sessizce yok sayılırdı.
TARGET_FPS_SWEEP = "target_fps"


def validate_sweep_name(name: str) -> None:
    """
    Bilinmeyen bir sweep adı SESSİZCE yok sayılmamalı.

    `--sweep DROWSINESS_SECOND=2.0` (sondaki S eksik) gibi bir yazım
    hatası, os.environ'a kimsenin okumadığı bir anahtar yazar; koşu
    normal görünür, N kez aynı sonucu üretir ve "parametrenin etkisi
    yok" diye yanlış bir sonuca varılır.
    """
    if name.lower() == TARGET_FPS_SWEEP:
        return

    from app.config import Config

    if not hasattr(Config, name):
        available = sorted(k for k in vars(Config) if k.isupper())
        raise SystemExit(
            f"--sweep: '{name}' diye bir ayar yok. "
            f"'{TARGET_FPS_SWEEP}' ya da şunlardan biri olmalı:\n  " + ", ".join(available)
        )


def main() -> int:
    # app/__init__.py'deki logger.info/warning çağrıları (hangi backend
    # seçildi, YOLO ağırlığı yüklendi mi, mediapipe'a düşüldü mü)
    # basicConfig olmadan tamamen sessiz kalıyordu — create_app() bunu
    # kendisi yapıyor ama bu script create_app kullanmıyor.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    args = parse_args()

    clips_dir = Path(args.clips)
    if not clips_dir.is_dir():
        raise SystemExit(f"Klip klasörü bulunamadı: {clips_dir}")

    all_labels = load_label_directory(args.labels)
    print(f"{len(all_labels)} etiketli klip bulundu.\n")

    sweep_name, sweep_values = (None, [None])
    if args.sweep:
        sweep_name, sweep_values = parse_sweep(args.sweep)
        validate_sweep_name(sweep_name)

    is_fps_sweep = sweep_name is not None and sweep_name.lower() == TARGET_FPS_SWEEP

    runs = []
    for value in sweep_values:
        target_fps = args.target_fps

        if is_fps_sweep:
            # Runner parametresi: doğrudan geçiliyor, Config'e dokunulmuyor.
            target_fps = float(value)
            print(f"--- {TARGET_FPS_SWEEP}={target_fps} ---")
        elif sweep_name is not None:
            # Config sınıfı import anında os.environ okuduğu için ayarı
            # ÖNCE yazıp modülleri sonra import etmek gerekir; bu yüzden
            # run_once içindeki `from app import ...` gecikmeli.
            os.environ[sweep_name] = str(value)
            _reload_config_modules()
            print(f"--- {sweep_name}={value} ---")

        drowsiness_seconds = args.drowsiness_seconds or _current_drowsiness_seconds()

        result = run_once(
            clips_dir=clips_dir,
            all_labels=all_labels,
            target_fps=target_fps,
            drowsiness_seconds=drowsiness_seconds,
            match_tolerance_s=args.match_tolerance,
            include_timeline=args.timeline,
        )
        result["settings"] = {
            # Backend rapora YAZILIR: hangi sistemin sayıları olduğu
            # sonradan tahmin edilecek bir şey olmamalı.
            "backend": result["backend"],
            "target_fps": target_fps,
            "drowsiness_seconds": drowsiness_seconds,
            "match_tolerance_s": args.match_tolerance,
            **({sweep_name: value} if sweep_name and not is_fps_sweep else {}),
        }
        runs.append(result)

        print()
        print(
            build_markdown_report(
                clip_rows=result["clips"],
                condition_rows=result["conditions"],
                subject_rows=result["subjects"],
                run_info=result["settings"],
            )
        )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"runs": runs}, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Sonuçlar yazıldı: {output_path}")

    return 0


def _reload_config_modules() -> None:
    """Sweep sırasında Config'in yeni ortam değişkenini görmesi için
    ilgili modülleri sys.modules'tan düşürür."""
    for name in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        del sys.modules[name]


def _current_drowsiness_seconds() -> float:
    from app.config import Config

    return float(Config.DROWSINESS_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
