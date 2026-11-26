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
