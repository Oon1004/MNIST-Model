"""Train the MNIST CNN and save model.pt."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from model import DigitCNN


def main():
    # 1. 실행할 때 전달한 옵션(에포크 수, 배치 크기)을 읽는다.
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    # 2. 매 실행마다 같은 학습/검증 분할이 만들어지도록 난수를 고정한다.
    torch.manual_seed(42)
    root = Path(__file__).parent

    # 3. CSV를 불러온다. 첫 번째 열은 정답, 다음 784개 열은 픽셀값이다.
    raw = np.loadtxt(root / "data/train.csv", delimiter=",", skiprows=1, dtype=np.float32)
    # 784개로 펼쳐진 픽셀을 1x28x28 이미지로 바꾸고, 값 범위를 0~1로 정규화한다.
    images = torch.from_numpy(raw[:, 1:].reshape(-1, 1, 28, 28) / 255.0)#     왜 255로 나눔????
    labels = torch.from_numpy(raw[:, 0].astype(np.int64))

    # 4. 데이터의 90%는 학습에, 나머지 10%는 검증에 사용한다. MNIST데이터가 학습과 테스트 용으로 나뉘어져 있지만 테스트는 label이 없음
    train_set, valid_set = random_split(
        TensorDataset(images, labels), [37_800, 4_200], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size)

#===========================================================================================================================

    # 5. CNN 모델, 손실 함수, 가중치와 바이어스를 갱신할 옵티마이저를 만든다.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DigitCNN().to(device)#cnn 모델을 생성 후 cpu나 gpu 장치로 이동

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    best_accuracy = 0.0
    history = {"train_loss": [], "train_accuracy": [], "valid_loss": [], "valid_accuracy": []}

    for epoch in range(args.epochs):
        # 6. 학습: 순전파 → 손실 계산 → 역전파 → 파라미터 갱신을 수행한다.
        model.train()
        train_loss = correct = total = 0
        for batch, (x, y) in enumerate(train_loader, start=1):
            optimizer.zero_grad()  # 이전 배치에서 계산된 기울기를 초기화한다.
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)  # 모델의 숫자별 점수와 실제 정답을 비교해 손실을 계산한다.
            loss.backward(); optimizer.step()  # 기울기를 계산하고 가중치와 바이어스를 갱신한다.
            train_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
            if batch % max(1, len(train_loader) // 4) == 0 or batch == len(train_loader):
                print(f"  epoch {epoch + 1} | {batch:>3}/{len(train_loader)} batches | loss {train_loss / total:.4f} | acc {correct / total:.2%}")
        train_accuracy = correct / total


        # 7. 검증: 파라미터를 바꾸지 않고, 처음 보는 검증 데이터의 성능을 측정한다.
        model.eval(); correct = total = 0
        valid_loss = 0.0
        with torch.no_grad():
            for x, y in valid_loader:
                logits = model(x.to(device))
                valid_loss += loss_fn(logits, y.to(device)).item() * len(y)
                correct += (logits.argmax(1).cpu() == y).sum().item()
                total += len(y)
        accuracy = correct / total
        history["train_loss"].append(train_loss / len(train_set))
        history["train_accuracy"].append(train_accuracy)
        history["valid_loss"].append(valid_loss / total)
        history["valid_accuracy"].append(accuracy)
        print(f"Epoch {epoch + 1}/{args.epochs} complete | validation loss: {valid_loss / total:.4f} | validation accuracy: {accuracy:.2%}")
        if accuracy > best_accuracy:
            # 검증 정확도가 가장 높은 모델의 가중치와 바이어스만 저장한다.
            best_accuracy = accuracy
            torch.save(model.state_dict(), root / "model.pt")
    print(f"Saved model.pt (best accuracy: {best_accuracy:.2%})")





    # 8. 기록한 손실과 정확도를 그래프로 저장한다.
    epochs = range(1, args.epochs + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["train_loss"], marker="o", label="train")
    axes[0].plot(epochs, history["valid_loss"], marker="o", label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross entropy"); axes[0].legend(); axes[0].grid()
    axes[1].plot(epochs, history["train_accuracy"], marker="o", label="train")
    axes[1].plot(epochs, history["valid_accuracy"], marker="o", label="validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1)); axes[1].legend(); axes[1].grid()
    figure.tight_layout()
    figure.savefig(root / "training_history.png", dpi=160)
    print("Saved training_history.png")


if __name__ == "__main__":
    main()
