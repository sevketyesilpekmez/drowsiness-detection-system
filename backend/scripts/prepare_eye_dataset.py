from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def parse_eye_state(path: Path) -> str:
    """
    MRL dosya biçimi:
    subject_index_gender_glasses_eye_state_reflections_lighting_sensor

    eye_state:
        0 = closed
        1 = open
    """
    parts = path.stem.split("_")

    if len(parts) < 8:
        raise ValueError(f"Geçersiz MRL dosya adı: {path.name}")

    eye_state = parts[4]

    if eye_state == "0":
        return "closed"

    if eye_state == "1":
        return "open"

    raise ValueError(f"Bilinmeyen göz etiketi: {path.name}")


def parse_subject_id(path: Path) -> str:
    """
    MRL dosya adının ilk alanı denek (kişi) kimliğidir: `s0001_00292_...`
    -> `s0001`. Kişi bazlı bölme bu kimliğe dayanır.
    """
    parts = path.stem.split("_")

    if len(parts) < 8:
        raise ValueError(f"Geçersiz MRL dosya adı: {path.name}")

    subject_id = parts[0]

    if not subject_id:
        raise ValueError(f"Dosya adından denek kimliği okunamadı: {path.name}")

    return subject_id


def split_subjects(
    images_by_subject: dict[str, list[Path]],
    *,
    val_ratio: float,
    rng: random.Random,
) -> tuple[list[str], list[str]]:
    """
    Denekleri (kişileri) train/val gruplarına ayırır — GÖRÜNTÜLERİ değil.

    Neden: MRL veri setinde aynı deneğin aynı oturumdan alınmış yüzlerce
    neredeyse aynı karesi var. Görüntü bazında rastgele bölünürse aynı
    kişinin komşu kareleri hem eğitim hem doğrulama setine düşer; model
    o kişiyi ezberleyip doğrulamada da yüksek skor alır. Ölçtüğümüz şey
    "yeni bir sürücüye genelleme" değil, "gördüğü kişiyi hatırlama"
    olur. Bu yüzden bir kişinin TÜM kareleri tek bir tarafta kalmalı.

    Deneklerin görüntü sayıları eşit olmadığı için hedef oran görüntü
    sayısı üzerinden takip edilir: karıştırılmış denek listesinden,
    toplam görüntünün `val_ratio` kadarı doldurulana dek val'e eklenir.

    Returns:
        (train_subjects, val_subjects) — ikisi de deterministik olması
        için sıralanmış döner.
    """
    if not images_by_subject:
        raise RuntimeError("Hiç denek bulunamadı.")

    if len(images_by_subject) < 2:
        raise RuntimeError(
            "Kişi bazlı bölme için en az 2 denek gerekir; "
            f"bulunan: {len(images_by_subject)}. Tek deneklik bir veri "
            "setinde train ve val'in ayrık olması mümkün değil."
        )

    total_images = sum(len(paths) for paths in images_by_subject.values())
    target_val_images = total_images * val_ratio

    subjects = sorted(images_by_subject)
    rng.shuffle(subjects)

    val_subjects: list[str] = []
    val_image_count = 0

    for subject in subjects:
        # Son deneği val'e alıp train'i boşaltmamak için koruma.
        if len(val_subjects) == len(subjects) - 1:
            break

        if val_image_count >= target_val_images:
            break

        val_subjects.append(subject)
        val_image_count += len(images_by_subject[subject])

    if not val_subjects:
        # val_ratio çok küçükse en az bir denek val'e ayrılır — aksi
        # halde doğrulama seti boş kalır.
        val_subjects.append(subjects[0])

    val_set = set(val_subjects)
    train_subjects = [subject for subject in subjects if subject not in val_set]

    return sorted(train_subjects), sorted(val_subjects)


def copy_files(
    files: list[Path],
    output_dir: Path,
    split_name: str,
    class_name: str,
) -> None:
    destination = output_dir / split_name / class_name
    destination.mkdir(parents=True, exist_ok=True)

    for index, source in enumerate(files):
        # Aynı isimli dosyaların çakışmasını önlemek için
        # klasör adını ve sıra numarasını ekliyoruz.
        target_name = (
            f"{source.parent.name}_{index:06d}_{source.name}"
        )
        shutil.copy2(source, destination / target_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MRL Eye Dataset'i train/val olarak hazırlar."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/raw/mrlEyes_2018_01"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eyes"),
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--split-by",
        choices=("subject", "image"),
        default="subject",
        help=(
            "subject (varsayılan): bir kişinin tüm kareleri tek bir "
            "tarafta kalır, doğrulama skoru 'yeni sürücüye genelleme'yi "
            "ölçer. image: eski (sızıntılı) davranış — aynı kişinin "
            "kareleri iki tarafa da dağılır, skor olduğundan yüksek "
            "çıkar. Yalnızca iki yöntemi karşılaştırmak için tutuluyor, "
            "raporlanan sonuç için kullanılmamalı."
        ),
    )
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(
            f"Kaynak klasör bulunamadı: {args.source}"
        )

    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio 0 ile 1 arasında olmalı.")

    rng = random.Random(args.seed)

    grouped: dict[str, list[Path]] = {
        "open": [],
        "closed": [],
    }
    images_by_subject: dict[str, list[Path]] = {}

    for path in args.source.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        class_name = parse_eye_state(path)
        grouped[class_name].append(path)
        images_by_subject.setdefault(parse_subject_id(path), []).append(path)

    if not grouped["open"] or not grouped["closed"]:
        raise RuntimeError(
            "Açık veya kapalı göz görüntüleri bulunamadı."
        )

    if args.output.exists():
        shutil.rmtree(args.output)

    if args.split_by == "subject":
        train_subjects, val_subjects = split_subjects(
            images_by_subject,
            val_ratio=args.val_ratio,
            rng=rng,
        )
        val_subject_set = set(val_subjects)

        counts = {
            ("train", "open"): 0,
            ("train", "closed"): 0,
            ("val", "open"): 0,
            ("val", "closed"): 0,
        }

        for class_name, files in grouped.items():
            training_files = [p for p in files if parse_subject_id(p) not in val_subject_set]
            validation_files = [p for p in files if parse_subject_id(p) in val_subject_set]

            copy_files(training_files, args.output, "train", class_name)
            copy_files(validation_files, args.output, "val", class_name)

            counts[("train", class_name)] = len(training_files)
            counts[("val", class_name)] = len(validation_files)

            print(f"{class_name}: train={len(training_files)}, val={len(validation_files)}")

        for split in ("train", "val"):
            if counts[(split, "open")] == 0 or counts[(split, "closed")] == 0:
                raise RuntimeError(
                    f"{split} setinde sınıflardan biri boş kaldı "
                    f"(open={counts[(split, 'open')]}, closed={counts[(split, 'closed')]}). "
                    "Farklı bir --seed veya --val-ratio deneyin."
                )

        print(
            f"\nBölme: kişi bazlı — train {len(train_subjects)} kişi, "
            f"val {len(val_subjects)} kişi, ortak 0 kişi."
        )
        print(f"Val denekleri: {', '.join(val_subjects)}")
    else:
        print(
            "UYARI: --split-by image seçildi. Aynı kişinin kareleri hem "
            "eğitim hem doğrulama setine düşecek; doğrulama skoru yeni "
            "bir sürücüye genelleme başarımını YANSITMAZ."
        )
        for class_name, files in grouped.items():
            shuffled = list(files)
            rng.shuffle(shuffled)

            validation_count = int(len(shuffled) * args.val_ratio)
            validation_files = shuffled[:validation_count]
            training_files = shuffled[validation_count:]

            copy_files(training_files, args.output, "train", class_name)
            copy_files(validation_files, args.output, "val", class_name)

            print(f"{class_name}: train={len(training_files)}, val={len(validation_files)}")

    print(f"\nVeri seti hazırlandı: {args.output}")


if __name__ == "__main__":
    main()
