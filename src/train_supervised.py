"""Entraînement de la référence supervisée sur le pool MNIST."""

import json
import time
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader

from src.engine import evaluate, set_seed, train_one_epoch
from src.models import SmallCNN



def train_reference(train_data, test_data, output_dir="results/reference_A",
                    seed=42, epochs=30, batch_size=128, learning_rate=0.001,
                    device=None):
    """Entraîne un modèle neuf, l'évalue une fois et enregistre ses résultats."""
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("Les époques, la taille du lot et le taux doivent être positifs.")
    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(
            f"Le dossier {output} existe déjà. Choisissez un nouveau dossier avec output_dir."
        ) from error
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
              "pool_indices": getattr(train_data, "split_path", None),
              "train_images": len(train_data), "num_digit_labels": len(train_data),
              "pretraining_images": 0, "source_checkpoint": None,
              "test_images": len(test_data), "device": str(device),
              "torch_version": str(torch.__version__),
              "torchvision_version": str(torchvision.__version__),
              "parameters": sum(p.numel() for p in model.parameters())}
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
