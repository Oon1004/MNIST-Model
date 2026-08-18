import torch.nn as nn


class DigitCNN(nn.Module):
    """CNN for 28 x 28 grayscale handwritten digits."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # 28x28 입력에서 특징(선, 곡선)을 32개 추출한다.
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            # 더 복잡한 숫자 형태를 64개 특징으로 확장한다.
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            # 7x7 특징 맵을 펼쳐 0~9의 점수(logit) 10개를 만든다.
            nn.Flatten(), nn.Dropout(0.25), nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(0.25), nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.network(x)
