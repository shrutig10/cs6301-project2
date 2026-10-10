import torch
from torch import nn


class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.classifier = nn.Linear(128, 10)

    def forward(self, images):
        features = self.features(images)
        pooled = self.pool(features)
        flattened = self.flatten(pooled)
        logits = self.classifier(flattened)
        return logits


if __name__ == "__main__":
    model = BaselineCNN()
    images = torch.zeros(4, 3, 32, 32)

    with torch.no_grad():
        features = images

        for layer in model.features:
            features = layer(features)
            print(type(layer).__name__, "→", features.shape)

        pooled = model.pool(features)
        print("Global average pooling →", pooled.shape)

        flattened = model.flatten(pooled)
        print("Flatten →", flattened.shape)

        logits = model(images)
        print("Class scores →", logits.shape)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("Trainable parameters:", parameter_count)

    assert logits.shape == (4, 10)
    assert torch.isfinite(logits).all().item()

    print("Baseline checks passed.")