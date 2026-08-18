"""Simple visual interface for the trained digit classifier."""
import tkinter as tk
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from model import DigitCNN


SIZE = 280


class App:
    def __init__(self, root):
        self.canvas = tk.Canvas(root, width=SIZE, height=SIZE, bg="black")
        self.canvas.pack(padx=12, pady=12)
        self.canvas.bind("<Button-1>", self.start)
        self.canvas.bind("<B1-Motion>", self.draw)
        buttons = tk.Frame(root); buttons.pack(pady=(0, 8))
        tk.Button(buttons, text="Predict", command=self.predict).pack(side="left", padx=4)
        tk.Button(buttons, text="Clear", command=self.clear).pack(side="left", padx=4)
        self.label = tk.Label(root, text="Draw a digit (0-9)", font=("Arial", 16))
        self.label.pack(pady=(0, 12))
        self.image = Image.new("L", (SIZE, SIZE), 0)
        self.pen = ImageDraw.Draw(self.image)
        self.previous = None
        # train.py가 저장한 학습 파라미터를 같은 모델 구조에 불러온다.
        self.model = DigitCNN()
        self.model.load_state_dict(torch.load(Path(__file__).parent / "model.pt", map_location="cpu", weights_only=True))
        self.model.eval()

    def start(self, event):
        self.previous = (event.x, event.y)
        self.draw(event)

    def draw(self, event):
        point = (event.x, event.y)
        if self.previous:
            self.canvas.create_line(*self.previous, *point, fill="white", width=18, capstyle="round", smooth=True)
            self.pen.line([self.previous, point], fill=255, width=18)
        self.previous = point

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (SIZE, SIZE), 0)
        self.pen = ImageDraw.Draw(self.image)
        self.previous = None
        self.label.config(text="Draw a digit (0-9)")

    def predict(self):
        box = self.image.getbbox()
        if box is None:
            self.label.config(text="Draw something first")
            return
        digit = self.image.crop(box)
        # 사용자가 그린 숫자를 MNIST 입력 형식(28x28, 중앙 배치)으로 바꾼다.
        digit.thumbnail((20, 20))
        canvas = Image.new("L", (28, 28), 0)
        canvas.paste(digit, ((28 - digit.width) // 2, (28 - digit.height) // 2))
        x = torch.from_numpy(np.asarray(canvas, dtype=np.float32)[None, None] / 255.0)
        # softmax 점수 중 가장 큰 숫자를 최종 예측으로 사용한다.
        with torch.no_grad(): probabilities = self.model(x).softmax(1)[0]
        number = probabilities.argmax().item()
        self.label.config(text=f"Prediction: {number}  ({probabilities[number]:.1%})")


if __name__ == "__main__":
    root = tk.Tk(); root.title("MNIST Digit Classifier")
    App(root); root.mainloop()
