# MNIST Deep Learning Framework

> Develop a production-quality handwritten digit recognition system using PyTorch while following modern AI engineering practices.
>
> This project is intended as a learning project based on the **Top-Down Learning Method**. The goal is not only to train an MNIST classifier but also to understand how a real-world AI project is structured, implemented, trained, evaluated, and deployed.

---

# Project Objectives

The project should resemble the architecture of a professional deep learning repository.

The implementation must emphasize:

- Clean Architecture
- Object-Oriented Design
- Modular Components
- Reusable Code
- Extensibility
- Maintainability
- High Readability
- Experiment Tracking
- Model Versioning
- Reproducible Training

The codebase should be organized so that replacing the model (e.g. CNN → ResNet → Vision Transformer) requires minimal changes.

---

# Technical Requirements

Language

- Python 3.11+

Framework

- PyTorch

Libraries

- torchvision
- torchmetrics
- numpy
- matplotlib
- pandas
- pillow
- tqdm
- pyyaml
- opencv-python
- scikit-learn
- tensorboard
- rich
- typer (CLI)
- pathlib

Additional libraries may be added if they improve the architecture.

---

# Target Performance

Dataset

MNIST

Target Accuracy

≥99%

Inference Time

Optimized for CPU inference.

---

# Software Architecture

The project should follow a layered architecture.

Presentation Layer

↓

Application Layer

↓

Training Engine

↓

Model Layer

↓

Dataset Layer

↓

Infrastructure Layer

Every layer should have a clear responsibility.

---

# Dataset

This project uses the **Digit Recognizer** dataset from Kaggle.

Dataset source:

https://www.kaggle.com/competitions/digit-recognizer

The dataset is expected to be stored in the following directory.

```
data/

├── train.csv
├── test.csv
└── sample_submission.csv
```

The project **must not** include the dataset in the repository.

If the dataset is not found, the program should notify the user with an informative error message.

The dataset loader should:

- Read the CSV files using Pandas.
- Normalize pixel values to the range [0, 1].
- Split the training data into training and validation sets.
- Create custom PyTorch Dataset classes.
- Return DataLoader objects for training, validation, and testing.

# Project Structure

```
mnist-framework/

│

├── configs/
│   ├── default.yaml
│   ├── train.yaml
│   └── model.yaml
│

├── data/
│

├── datasets/
│   ├── mnist_dataset.py
│   └── transforms.py
│

├── models/
│   ├── base_model.py
│   ├── cnn.py
│   ├── mlp.py
│   └── factory.py
│

├── engine/
│   ├── trainer.py
│   ├── evaluator.py
│   ├── predictor.py
│   └── callbacks.py
│

├── losses/
│

├── optimizers/
│

├── schedulers/
│

├── metrics/
│

├── utils/
│

├── checkpoints/
│

├── logs/
│

├── outputs/
│

├── scripts/
│

├── tests/
│

├── train.py
├── evaluate.py
├── predict.py
├── export.py
├── requirements.txt
└── README.md
```

---

# Model Design

The architecture should support multiple interchangeable models.

Examples

- Fully Connected Network
- CNN
- LeNet
- ResNet
- Vision Transformer

Changing the model should only require modifying the configuration file.

---

# Training Pipeline

The training process should be completely modular.

Required components

- Dataset
- DataLoader
- Transform Pipeline
- Model
- Loss Function
- Optimizer
- Scheduler
- Metrics
- Trainer
- Checkpoint Manager
- Logger
- Evaluator

---

# Configuration System

All hyperparameters must be managed through YAML configuration files.

Examples

- learning rate
- optimizer
- scheduler
- epochs
- batch size
- dropout
- activation
- model type
- dataset path

No magic numbers should exist inside the source code.

---

# Logging

Training logs should include

- Epoch
- Step
- Loss
- Accuracy
- Learning Rate
- Training Time
- Validation Time
- GPU Memory (if CUDA available)

Support TensorBoard logging.

---

# Checkpoint System

Automatically save

- best_model.pt
- last_model.pt

Include

- model state
- optimizer state
- scheduler state
- epoch
- metrics
- configuration

Support resume training.

---

# Evaluation

Provide

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

Visualize results.

---

# Prediction

The prediction module should

- load checkpoint
- preprocess image
- perform inference
- output probabilities
- visualize prediction

Support

PNG

JPEG

BMP

---

# Code Quality

The project should strictly follow

- PEP8
- SOLID Principles
- Type Hint
- Docstring
- Logging instead of print
- Exception Handling
- Dependency Injection where appropriate

Avoid duplicated code.

---

# Future Extensions

The architecture should allow future implementation of

- Transfer Learning
- Mixed Precision Training
- Distributed Training
- ONNX Export
- TorchScript
- Quantization
- Hyperparameter Search
- Experiment Management
- Docker Deployment
- REST API (FastAPI)
- Web Interface
- Mobile Deployment

without major refactoring.

---

# Final Goal

The final repository should resemble an open-source deep learning framework rather than a simple MNIST tutorial.

Priority should be given to software architecture, modularity, scalability, readability, and maintainability over minimizing code length.