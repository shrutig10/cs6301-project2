import torch
from torch import nn


class TwoConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        residual=False,
    ):
        super().__init__()

        self.residual = residual

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        if residual:
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                )
            else:
                self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        if self.residual:
            out = out + self.shortcut(x)

        return self.relu(out)

class DeepCNN(nn.Module):
    def __init__(self, residual=False):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.stage1 = nn.Sequential(
            TwoConvBlock(32, 32, residual=residual),
            TwoConvBlock(32, 32, residual=residual),
        )

        self.stage2 = nn.Sequential(
            TwoConvBlock(
                32, 64, stride=2, residual=residual
            ),
            TwoConvBlock(64, 64, residual=residual),
        )

        self.stage3 = nn.Sequential(
            TwoConvBlock(
                64, 128, stride=2, residual=residual
            ),
            TwoConvBlock(128, 128, residual=residual),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.classifier = nn.Linear(128, 10)

    def forward(self, images):
        x = self.stem(images)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.classifier(x)


if __name__ == "__main__":
    images = torch.zeros(4, 3, 32, 32)

    for name, residual in [
        ("Plain", False),
        ("Residual", True),
    ]:
        model = DeepCNN(residual=residual)

        print(f"\n{name} model")

        with torch.no_grad():
            x = model.stem(images)
            print("Stem:", x.shape)

            x = model.stage1(x)
            print("Stage 1:", x.shape)

            x = model.stage2(x)
            print("Stage 2:", x.shape)

            x = model.stage3(x)
            print("Stage 3:", x.shape)

            logits = model(images)
            print("Logits:", logits.shape)

        parameter_count = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        print("Trainable parameters:", parameter_count)

        assert logits.shape == (4, 10)
        assert torch.isfinite(logits).all().item()

    print("\nDeep model shape checks passed.")