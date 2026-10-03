"""
prepare_eye_dataset.py'nin kişi (denek) bazlı bölme davranışını doğrular.

Bu testlerin varlık sebebi somut bir bulgu: veri seti başlangıçta
görüntü bazında rastgele bölünüyordu ve üretilen sette 37 deneğin
37'si de HEM train HEM val tarafında yer alıyordu. MRL veri setinde
aynı deneğin aynı oturumdan alınmış yüzlerce neredeyse aynı karesi
olduğu için bu, doğrulama skorunu "yeni bir sürücüye genelleme" değil
"gördüğü kişiyi hatırlama" ölçen bir sayıya çeviriyordu. Aşağıdaki
testler, bölmenin bir daha sızıntı üretmemesini garanti eder.
"""

from __future__ import annotations

import random

import pytest

from scripts.prepare_eye_dataset import parse_subject_id, split_subjects


def make_path(subject: str, index: int, eye_state: str = "0"):
    """MRL adlandırmasına uyan sahte bir yol üretir:
    subject_index_gender_glasses_eye_state_reflections_lighting_sensor"""
    from pathlib import Path

    return Path(f"{subject}/{subject}_{index:05d}_0_0_{eye_state}_0_0_01.png")


def make_images_by_subject(subject_counts: dict[str, int]) -> dict[str, list]:
    return {
        subject: [make_path(subject, i) for i in range(count)]
        for subject, count in subject_counts.items()
    }


def test_parse_subject_id_reads_first_field():
    assert parse_subject_id(make_path("s0014", 3)) == "s0014"


def test_parse_subject_id_rejects_malformed_name():
    from pathlib import Path

    with pytest.raises(ValueError):
        parse_subject_id(Path("bozuk_isim.png"))


def test_split_has_no_subject_overlap():
    """Asıl regresyon testi: hiçbir denek iki tarafta birden olmamalı."""
    images = make_images_by_subject({f"s{i:04d}": 100 for i in range(1, 21)})

    train, val = split_subjects(images, val_ratio=0.2, rng=random.Random(42))

    assert set(train) & set(val) == set()
    assert set(train) | set(val) == set(images)
    assert train and val


def test_split_respects_val_ratio_approximately():
    """Denek başına görüntü sayısı eşit olmadığı için oran tam tutmaz;
    hedefin makul bir bandında kalması yeterli."""
    images = make_images_by_subject({f"s{i:04d}": 50 + i for i in range(1, 31)})
    total = sum(len(paths) for paths in images.values())

    _, val = split_subjects(images, val_ratio=0.2, rng=random.Random(7))
    val_images = sum(len(images[subject]) for subject in val)

    assert 0.1 <= val_images / total <= 0.35


def test_split_is_deterministic_for_same_seed():
    images = make_images_by_subject({f"s{i:04d}": 40 for i in range(1, 16)})

    first = split_subjects(images, val_ratio=0.2, rng=random.Random(1))
    second = split_subjects(images, val_ratio=0.2, rng=random.Random(1))

    assert first == second


def test_split_never_empties_training_set():
    """val_ratio uçta olsa bile train tarafı en az bir denek içermeli."""
    images = make_images_by_subject({"s0001": 10, "s0002": 10})

    train, val = split_subjects(images, val_ratio=0.99, rng=random.Random(3))

    assert train
    assert val


def test_split_assigns_at_least_one_subject_to_validation():
    images = make_images_by_subject({f"s{i:04d}": 100 for i in range(1, 11)})

    _, val = split_subjects(images, val_ratio=0.001, rng=random.Random(5))

    assert len(val) >= 1


def test_split_rejects_single_subject_dataset():
    """Tek deneklik bir sette ayrık bölme mümkün değil — sessizce
    sızıntılı bir set üretmek yerine hata vermeli."""
    images = make_images_by_subject({"s0001": 100})

    with pytest.raises(RuntimeError):
        split_subjects(images, val_ratio=0.2, rng=random.Random(1))


def test_split_rejects_empty_dataset():
    with pytest.raises(RuntimeError):
        split_subjects({}, val_ratio=0.2, rng=random.Random(1))
