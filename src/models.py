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
