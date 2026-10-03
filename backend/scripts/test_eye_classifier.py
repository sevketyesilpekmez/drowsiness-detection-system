"""
weights/eye_classifier.pth'in gerçekten yüklenip inference yapabildiğini
doğrulayan hızlı smoke test.

Transform, train_eye_classifier.py'deki `build_dataloaders()` ile birebir
aynı olmalı: yalnızca ToTensor() ([0,1] aralığı), Normalize yok — model
[0,1] aralığında eğitildi. Production inference kodu
(torch_eye_classifier.py::_preprocess) de aynı şekilde /255.0 kullanıyor.
"""

import sys
from pathlib import Path

# backend/ kökünü sys.path'e ekle ki `app.models...` import edilebilsin —
# bu script backend/scripts/ altında, backend/ altında değil.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402

from app.models.eye_state_cnn import EyeStateCNN  # noqa: E402


CLASS_NAMES = ["closed", "open"]

# train_eye_classifier.py'deki transform ile BİREBİR aynı olmalı —
# Normalize YOK, yalnızca ToTensor() ([0,255] -> [0.0,1.0]).
transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ]
)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

model = EyeStateCNN()
state_dict = torch.load(
    "weights/eye_classifier.pth",
    map_location=device,
)
model.load_state_dict(state_dict)
model.to(device)
model.eval()


def predict(image_path: Path) -> None:
    image = Image.open(image_path).convert("L")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_index = probabilities.argmax(dim=1).item()

    print(f"Görüntü: {image_path}")
    print(f"Tahmin: {CLASS_NAMES[predicted_index]}")
    print(f"Güven: {probabilities[0][predicted_index].item():.4f}")
    print()


# Hem "closed" hem "open" örneği test ediliyor — modelin tek bir sınıfa
# hep aynı tahmini vermediğini (örn. her zaman "open" demediğini) görmek
# için tek görüntü yeterli değil.
predict(next(Path("data/eyes/val/closed").glob("*")))
predict(next(Path("data/eyes/val/open").glob("*")))
