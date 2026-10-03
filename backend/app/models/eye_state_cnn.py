from __future__ import annotations

import torch
import torch.nn as nn


class EyeStateCNN(nn.Module):
    """
    Göz kırpıntısını OPEN_EYES/CLOSED_EYES olarak sınıflandıran küçük bir CNN.

    Girdi: (N, 1, H, W) gri tonlamalı tensor. `AdaptiveAvgPool2d` sayesinde
    sabit bir giriş boyutu zorunlu değil, ama eğitim/inference'ta tutarlı
    bir boyut kullanılması önerilir (bkz. train_eye_classifier.py).

    Çıktı: (N, 2) ham logit, softmax uygulanmamıştır — eğitimde
    CrossEntropyLoss, inference'ta torch_eye_classifier.py softmax uygular.

    Sınıf sırası: 0 -> CLOSED_EYES, 1 -> OPEN_EYES (ImageFolder'ın klasör
    isimlerini alfabetik sıralamasından gelir, "closed" < "open").
    """

    def __init__(self, *, num_classes: int = 2) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
