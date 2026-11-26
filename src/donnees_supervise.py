"""Chargement de MNIST et création reproductible du pool d'apprentissage."""

from pathlib import Path

import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset


def load_mnist(data_dir="data", split_path="splits/pool_seed42.npz", download=True):
    """Retourne le pool de 40 000 images et le jeu de test officiel.

    Les indices sauvegardés correspondent aux positions dans le train officiel.
    La graine 42 permet de retrouver toujours le même tirage.
    """
    transform = transforms.ToTensor()
    train = datasets.MNIST(root=data_dir, train=True, transform=transform, download=download)
    test = datasets.MNIST(root=data_dir, train=False, transform=transform, download=download)

    split_path = Path(split_path)
    if split_path.exists():
        with np.load(split_path, allow_pickle=False) as saved:
            indices = saved["pool_indices"]
    else:
        indices = np.random.default_rng(42).choice(len(train), size=40_000, replace=False)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(split_path, pool_indices=indices, seed=42)

    if (indices.shape != (40_000,) or not np.issubdtype(indices.dtype, np.integer)
            or len(np.unique(indices)) != 40_000
            or np.any(indices < 0) or np.any(indices >= len(train))):
        raise ValueError("Le fichier du pool doit contenir 40 000 indices uniques valides.")
    return Subset(train, indices.tolist()), test


"""Architecture CNN partagée pour la classification et RotNet."""

from torch import nn


class SmallCNN(nn.Module):
    """CNN pour images MNIST, avec une sortie adaptée à la tâche."""

    def __init__(self, num_classes=10):
        super().__init__()
        # Les mêmes caractéristiques pourront être transférées entre tâches.
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, images):
        features = self.features(images)
        return self.classifier(features)


"""Fonctions communes pour entraîner et évaluer un modèle PyTorch."""

import random

import numpy as np
import torch
from torch import nn


def set_seed(seed):
    """Initialise les générateurs aléatoires utilisés par le projet."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, device):
    """Entraîne le modèle sur tous les lots et retourne perte et précision."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        scores = model(images)
        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (scores.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    if total == 0:
        raise ValueError("Le jeu de données est vide.")
    return {"loss": total_loss / total, "accuracy": correct / total}


@torch.no_grad()
def evaluate(model, loader, device, num_classes=10):
    """Évalue sans modifier les poids; lignes = vrais labels, colonnes = prédictions."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        scores = model(images)
        if scores.ndim != 2 or scores.shape[1] != num_classes:
            raise ValueError("Le nombre de sorties ne correspond pas à num_classes.")
        loss = criterion(scores, labels)
        predictions = scores.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        pairs = labels * num_classes + predictions
        confusion += torch.bincount(pairs, minlength=num_classes ** 2).reshape(num_classes, num_classes).cpu()
    if total == 0:
        raise ValueError("Le jeu de données est vide.")
    return {"loss": total_loss / total, "accuracy": correct / total,
            "correct": correct, "total": total, "confusion_matrix": confusion}


"""Entraînement de la référence supervisée sur le pool MNIST."""

import json
import time
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader



def train_reference(train_data, test_data, output_dir="results/reference_A",
                    seed=42, epochs=30, batch_size=128, learning_rate=0.001,
                    device=None):
    """Entraîne un modèle neuf, l'évalue une fois et enregistre ses résultats."""
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("Les époques, la taille du lot et le taux doivent être positifs.")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(seed)
    model = SmallCNN(num_classes=10).to(device)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    start = time.perf_counter()
    for epoch in range(epochs):
        metrics = train_one_epoch(model, train_loader, optimizer, device)
        history.append({"epoch": epoch + 1, **metrics})
        print(f"Époque {epoch + 1}/{epochs} : perte={metrics['loss']:.4f}, "
              f"précision entraînement={metrics['accuracy']:.2%}", flush=True)

    test_metrics = evaluate(model, test_loader, device)
    elapsed = time.perf_counter() - start
    config = {"method": "A", "seed": seed, "epochs": epochs,
              "batch_size": batch_size, "learning_rate": learning_rate,
              "optimizer": "Adam", "architecture": "SmallCNN",
              "pool_indices": "splits/pool_seed42.npz",
              "train_images": len(train_data), "num_digit_labels": len(train_data),
              "pretraining_images": 0, "source_checkpoint": None,
              "test_images": len(test_data), "device": str(device),
              "torch_version": str(torch.__version__),
              "torchvision_version": str(torchvision.__version__),
              "parameters": sum(p.numel() for p in model.parameters())}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": config},
               output / "model.pt")
    results = {"config": config, "training_history": history,
               "test_loss": test_metrics["loss"],
               "test_accuracy": test_metrics["accuracy"],
               "test_correct": test_metrics["correct"],
               "test_total": test_metrics["total"],
               "confusion_matrix": test_metrics["confusion_matrix"].tolist(),
               "duration_seconds": elapsed}
    with (output / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    return model, results
