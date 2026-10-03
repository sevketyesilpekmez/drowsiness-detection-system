"""
train_eye_classifier.py'nin metrik hesaplarını doğrular.

Bu fonksiyonlar bilinçli olarak torch'tan bağımsız tutuldu (yalnızca
confusion matrix alıyorlar), böylece eğitim ortamı kurulu olmadan da
test edilebiliyorlar.

Motivasyon: model seçimi başlangıçta yalnızca toplam accuracy'ye
bakıyordu. Bu sistemde iki hata türünün maliyeti eşit değil — kapalı
gözü "açık" sanmak uyuyan sürücünün uyarılmaması demek, tersi ise
yalnızca gereksiz alarm. Bu yüzden closed sınıfının recall/F1'i ve
false-awake oranı ayrıca hesaplanıp raporlanıyor.
"""

from __future__ import annotations

import pytest

from scripts.train_eye_classifier import classification_metrics, selection_score

# confusion_matrix[gerçek][tahmin], sıra: closed, open
PERFECT = [[100, 0], [0, 100]]
# Kapalı gözlerin 20'si "açık" sanılmış — güvenlik açısından kötü yön.
FALSE_AWAKE_HEAVY = [[80, 20], [0, 100]]
# Açık gözlerin 20'si "kapalı" sanılmış — yalnızca gereksiz alarm.
FALSE_ALARM_HEAVY = [[100, 0], [20, 80]]


def test_perfect_prediction_scores_one():
    metrics = classification_metrics(PERFECT)

    assert metrics["accuracy"] == 1.0
    assert metrics["closed"]["recall"] == 1.0
    assert metrics["closed"]["f1"] == 1.0
    assert metrics["false_awake_rate"] == 0.0
    assert metrics["false_alarm_rate"] == 0.0


def test_false_awake_rate_counts_missed_closed_eyes():
    metrics = classification_metrics(FALSE_AWAKE_HEAVY)

    assert metrics["closed"]["recall"] == pytest.approx(0.8)
    assert metrics["false_awake_rate"] == pytest.approx(0.2)
    # Kapalı diye tahmin edilenlerin hepsi gerçekten kapalıydı.
    assert metrics["closed"]["precision"] == 1.0


def test_false_alarm_rate_counts_unnecessary_alarms():
    metrics = classification_metrics(FALSE_ALARM_HEAVY)

    assert metrics["false_alarm_rate"] == pytest.approx(0.2)
    assert metrics["closed"]["recall"] == 1.0
    assert metrics["closed"]["precision"] == pytest.approx(100 / 120)


def test_accuracy_alone_hides_the_dangerous_direction():
    """İki senaryonun accuracy'si aynı (0.90) ama güvenlik açısından
    biri diğerinden belirgin kötü. Metrik genişletmesinin varlık sebebi
    tam olarak bu."""
    dangerous = classification_metrics(FALSE_AWAKE_HEAVY)
    benign = classification_metrics(FALSE_ALARM_HEAVY)

    assert dangerous["accuracy"] == benign["accuracy"] == pytest.approx(0.9)
    assert dangerous["false_awake_rate"] > benign["false_awake_rate"]
    assert dangerous["closed"]["recall"] < benign["closed"]["recall"]


def test_support_matches_ground_truth_counts():
    metrics = classification_metrics(FALSE_AWAKE_HEAVY)

    assert metrics["closed"]["support"] == 100
    assert metrics["open"]["support"] == 100


def test_degenerate_all_closed_model_is_penalised_by_f1():
    """Her kareye 'kapalı' diyen bozuk model: closed_recall mükemmel
    görünür ama F1 precision üzerinden cezalandırır. --select-by
    closed_recall uyarısının somut gerekçesi."""
    all_closed = [[100, 0], [100, 0]]

    metrics = classification_metrics(all_closed)

    assert metrics["closed"]["recall"] == 1.0
    assert metrics["closed"]["f1"] < 0.7
    assert selection_score(metrics, select_by="closed_recall") == 1.0
    assert selection_score(metrics, select_by="closed_f1") < 1.0


def test_selection_score_maps_to_expected_metric():
    metrics = classification_metrics(FALSE_AWAKE_HEAVY)

    assert selection_score(metrics, select_by="accuracy") == metrics["accuracy"]
    assert selection_score(metrics, select_by="closed_f1") == metrics["closed"]["f1"]
    assert selection_score(metrics, select_by="closed_recall") == metrics["closed"]["recall"]


def test_selection_score_rejects_unknown_criterion():
    with pytest.raises(ValueError):
        selection_score(classification_metrics(PERFECT), select_by="bilinmeyen")


def test_empty_confusion_matrix_raises():
    with pytest.raises(ValueError):
        classification_metrics([[0, 0], [0, 0]])


def test_zero_division_is_handled_for_absent_class():
    """Doğrulama setinde hiç 'open' örnek yoksa bölme hatası vermemeli."""
    metrics = classification_metrics([[50, 0], [0, 0]])

    assert metrics["open"]["recall"] == 0.0
    assert metrics["open"]["f1"] == 0.0
    assert metrics["closed"]["recall"] == 1.0
