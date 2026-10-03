"""
EyeStateCNN'i (backend/app/models/eye_state_cnn.py) etiketli bir göz
veri setiyle eğitip backend/weights/eye_classifier.pth'e kaydeder.

Beklenen veri seti klasör yapısı (torchvision.datasets.ImageFolder formatı):

    <data-dir>/
      train/
        closed/   *.jpg / *.png  (gözler kapalı örnekler)
        open/     *.jpg / *.png  (gözler açık örnekler)
      val/
        closed/
        open/

Klasör isimleri tam olarak küçük harfle "closed" ve "open" olmalı —
ImageFolder sınıfları alfabetik sıralar, bu da EyeStateCNN'in çıktı
indekslerinin (0->CLOSED_EYES, 1->OPEN_EYES) varsaydığı sırayla eşleşmesini
sağlar. Script, sınıfların gerçekten bu sırada olduğunu çalışma zamanında
doğrular (aksi halde eğitim/inference arasında sessiz bir etiket ters
çevrilmesi olurdu).

Önerilen halka açık veri setleri: MRL Eye Dataset, CEW (Closed Eyes In
The Wild) — indirdikten sonra yukarıdaki klasör yapısına göre yeniden
düzenlenmeleri gerekir.

Kullanım:

    cd backend
    python scripts/train_eye_classifier.py --data-dir /path/to/dataset --epochs 15
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# backend/ kökünü sys.path'e ekle ki `app.models...` import edilebilsin
# (bu script backend/scripts/ altında, backend/ altında değil).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EXPECTED_CLASSES = ["closed", "open"]
EYE_CROP_SIZE = 64  # torch_eye_classifier.py'deki TorchEyeClassifier(crop_size=...) ile AYNI olmalı


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, help="train/ ve val/ alt klasörlerini içeren veri seti kökü")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu", help="'cpu' ya da 'cuda:0'")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent.parent / "weights" / "eye_classifier.pth"),
        help="Eğitilen ağırlıkların kaydedileceği yol",
    )
    parser.add_argument(
        "--select-by",
        choices=("closed_f1", "accuracy", "closed_recall"),
        default="closed_f1",
        help=(
            "Hangi epoch'un 'en iyi' sayılıp kaydedileceği. closed_f1 "
            "(varsayılan): kapalı göz sınıfının F1'i — bu sistemde kapalı "
            "gözü 'açık' sanmanın maliyeti tersinden yüksek olduğu için "
            "toplam accuracy'den daha anlamlı. accuracy: eski davranış. "
            "closed_recall: yalnızca kaçırılan kapalı gözleri en aza "
            "indirir; dikkat — bu ölçüt tek başına 'her kareye kapalı de' "
            "diyen bozuk bir modeli de ödüllendirebilir, seçilirse "
            "false_alarm_rate mutlaka kontrol edilmeli."
        ),
    )
    return parser.parse_args()


def build_dataloaders(data_dir: Path, *, batch_size: int):
    # torch/torchvision importu bilerek fonksiyon içinde: modül üst
    # seviyesinde import edilseydi, --help gibi torch gerektirmeyen bir
    # çağrı bile torch kurulu olmayan bir ortamda patlardı.
    import torch
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((EYE_CROP_SIZE, EYE_CROP_SIZE)),
            transforms.ToTensor(),  # [0, 255] -> [0.0, 1.0], (H,W,C) -> (C,H,W)
        ]
    )

    train_dataset = datasets.ImageFolder(str(data_dir / "train"), transform=transform)
    val_dataset = datasets.ImageFolder(str(data_dir / "val"), transform=transform)

    for name, dataset in (("train", train_dataset), ("val", val_dataset)):
        if dataset.classes != EXPECTED_CLASSES:
            raise ValueError(
                f"{name}/ klasöründeki sınıflar {dataset.classes} bulundu, "
                f"{EXPECTED_CLASSES} bekleniyordu. Klasör isimleri TAM OLARAK "
                "küçük harfle 'closed' ve 'open' olmalı (bkz. bu dosyanın "
                "başındaki modül docstring'i) — aksi halde eğitilen model "
                "ile inference (torch_eye_classifier.py) arasında SESSİZ "
                "bir etiket ters çevrilmesi olur."
            )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device) -> tuple[float, float, list[list[int]]]:
    """Doğrulama loss'u, accuracy'si ve confusion matrix'i döner.

    confusion_matrix[gerçek_sınıf][tahmin_edilen_sınıf] biçiminde 2x2'lik
    bir liste, satır/sütun sırası EXPECTED_CLASSES ile aynı. sklearn'e
    bağımlı olmamak için elle sayılıyor."""
    import torch

    model.eval()
    total_loss = 0.0
    correct = 0
    confusion_matrix = [[0, 0], [0, 0]]

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()

            for true_label, predicted_label in zip(labels.tolist(), predictions.tolist()):
                confusion_matrix[true_label][predicted_label] += 1

    val_loss = total_loss / len(loader.dataset)
    val_accuracy = correct / len(loader.dataset)
    return val_loss, val_accuracy, confusion_matrix


def classification_metrics(confusion_matrix: list[list[int]]) -> dict:
    """
    Confusion matrix'ten sınıf bazlı precision/recall/F1 üretir.

    Neden accuracy yetmiyor: bu sistemde iki hata türünün maliyeti eşit
    değil. Kapalı bir gözü "açık" sanmak (false-awake) uyuyan sürücünün
    uyarılmaması demektir; açık bir gözü "kapalı" sanmak yalnızca gereksiz
    bir alarma yol açar. Toplam accuracy bu ikisini tek sayıda eritir ve
    sınıflar dengesizse yanıltıcı olur. Bu yüzden closed sınıfının
    recall'ü ve false-awake oranı ayrıca raporlanıyor.

    confusion_matrix[gerçek][tahmin], sıra EXPECTED_CLASSES (closed, open).

    Returns:
        accuracy, closed/open için precision-recall-f1, ayrıca
        false_awake_rate (gerçekte kapalıyken açık denen örneklerin oranı)
        ve false_alarm_rate (gerçekte açıkken kapalı denen oran).
    """
    (closed_closed, closed_open), (open_closed, open_open) = confusion_matrix

    total = closed_closed + closed_open + open_closed + open_open
    if total == 0:
        raise ValueError("Boş confusion matrix — metrik hesaplanamaz.")

    def ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    def f1(precision: float, recall: float) -> float:
        return ratio(2 * precision * recall, precision + recall) if (precision + recall) else 0.0

    closed_precision = ratio(closed_closed, closed_closed + open_closed)
    closed_recall = ratio(closed_closed, closed_closed + closed_open)
    open_precision = ratio(open_open, open_open + closed_open)
    open_recall = ratio(open_open, open_open + open_closed)

    return {
        "accuracy": ratio(closed_closed + open_open, total),
        "closed": {
            "precision": closed_precision,
            "recall": closed_recall,
            "f1": f1(closed_precision, closed_recall),
            "support": closed_closed + closed_open,
        },
        "open": {
            "precision": open_precision,
            "recall": open_recall,
            "f1": f1(open_precision, open_recall),
            "support": open_closed + open_open,
        },
        # Güvenlik açısından kritik metrik: uyuklayan sürücünün gözü
        # kapalıyken "açık" denen kare oranı.
        "false_awake_rate": ratio(closed_open, closed_closed + closed_open),
        # Rahatsız edici ama tehlikeli olmayan yön: boşuna alarm.
        "false_alarm_rate": ratio(open_closed, open_closed + open_open),
    }


def format_metrics(metrics: dict) -> str:
    """classification_metrics çıktısını okunabilir bir tabloya çevirir."""
    lines = [
        "                 precision     recall         f1     destek",
        (
            f"  closed         {metrics['closed']['precision']:>9.4f}  "
            f"{metrics['closed']['recall']:>9.4f}  {metrics['closed']['f1']:>9.4f}  "
            f"{metrics['closed']['support']:>9}"
        ),
        (
            f"  open           {metrics['open']['precision']:>9.4f}  "
            f"{metrics['open']['recall']:>9.4f}  {metrics['open']['f1']:>9.4f}  "
            f"{metrics['open']['support']:>9}"
        ),
        "",
        f"  false-awake oranı (kapalı gözü 'açık' sanma): {metrics['false_awake_rate']:.4f}",
        f"  false-alarm oranı (açık gözü 'kapalı' sanma): {metrics['false_alarm_rate']:.4f}",
    ]
    return "\n".join(lines)


def selection_score(metrics: dict, *, select_by: str) -> float:
    """Model seçiminde kullanılacak skoru döndürür (bkz. --select-by)."""
    if select_by == "accuracy":
        return metrics["accuracy"]
    if select_by == "closed_f1":
        return metrics["closed"]["f1"]
    if select_by == "closed_recall":
        return metrics["closed"]["recall"]
    raise ValueError(f"Bilinmeyen --select-by değeri: {select_by}")


def format_confusion_matrix(confusion_matrix: list[list[int]]) -> str:
    """['closed', 'open'] etiketleriyle okunabilir bir tablo üretir."""
    header = "                 tahmin=closed  tahmin=open"
    row_closed = f"  gerçek=closed        {confusion_matrix[0][0]:>5}         {confusion_matrix[0][1]:>5}"
    row_open = f"  gerçek=open          {confusion_matrix[1][0]:>5}         {confusion_matrix[1][1]:>5}"
    return "\n".join([header, row_closed, row_open])


def main() -> None:
    args = parse_args()

    import torch

    from app.models.eye_state_cnn import EyeStateCNN

    data_dir = Path(args.data_dir)
    device = torch.device(args.device)

    train_loader, val_loader = build_dataloaders(data_dir, batch_size=args.batch_size)
    print(f"Eğitim örnekleri: {len(train_loader.dataset)}, doğrulama örnekleri: {len(val_loader.dataset)}")
    print(f"Sınıf sırası: {train_loader.dataset.classes} (0=CLOSED_EYES, 1=OPEN_EYES)")

    model = EyeStateCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_score = -1.0
    best_epoch_metrics: dict | None = None
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = output_path.with_name(output_path.stem + "_metrics.json")

    epoch_history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy, confusion_matrix = evaluate(model, val_loader, criterion, device)
        metrics = classification_metrics(confusion_matrix)
        score = selection_score(metrics, select_by=args.select_by)

        print(
            f"Epoch {epoch}/{args.epochs} - train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4f} "
            f"closed_recall={metrics['closed']['recall']:.4f} "
            f"closed_f1={metrics['closed']['f1']:.4f} "
            f"false_awake={metrics['false_awake_rate']:.4f}"
        )
        epoch_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "metrics": metrics,
            }
        )

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), output_path)
            best_epoch_metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "confusion_matrix": confusion_matrix,
                "metrics": metrics,
                "selected_by": args.select_by,
                "selection_score": score,
            }
            print(
                f"  -> Yeni en iyi model kaydedildi: {output_path} "
                f"({args.select_by}={score:.4f}, val_accuracy={val_accuracy:.4f})"
            )
            print(format_confusion_matrix(confusion_matrix))
            print(format_metrics(metrics))

    print(f"Eğitim bitti. En iyi {args.select_by}: {best_score:.4f}. Ağırlıklar: {output_path}")

    # Eğitim/doğrulama loss'ları ve en iyi epoch metrikleri .pth ile
    # birlikte kalıcı bir dosyaya kaydedilir, yalnızca terminale basılıp
    # kaybolmasın diye.
    metrics_path.write_text(
        json.dumps(
            {
                "class_order": EXPECTED_CLASSES,
                "epochs": epoch_history,
                "best": best_epoch_metrics,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Metrikler kaydedildi: {metrics_path}")


if __name__ == "__main__":
    main()
