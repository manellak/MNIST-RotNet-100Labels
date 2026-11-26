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
    pool = Subset(train, indices.tolist())
    # Conserver la provenance pour les configurations d'expériences.
    pool.split_path = str(split_path.resolve())
    return pool, test
