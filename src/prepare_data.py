from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader, Subset

from torchvision import datasets, transforms


project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data"

image_transform = transforms.ToTensor()

full_training_dataset = datasets.CIFAR10(
    root=str(data_dir),
    train=True,
    download=True,
    transform=image_transform,
)

print("Number of training-set images:", len(full_training_dataset))

image, label = full_training_dataset[0]

print("First image shape:", image.shape)
print("First image label:", label)
print("Class name:", full_training_dataset.classes[label])

split_path = data_dir / "split_seed42.json"

if split_path.exists():
    with split_path.open("r") as file:
        split_indices = json.load(file)
else:
    split_generator = torch.Generator().manual_seed(42)

    shuffled_indices = torch.randperm(
        len(full_training_dataset),
        generator=split_generator,
    ).tolist()

    split_indices = {
        "train": shuffled_indices[:45000],
        "validation": shuffled_indices[45000:],
    }

    with split_path.open("w") as file:
        json.dump(split_indices, file)

train_dataset = Subset(
    full_training_dataset,
    split_indices["train"],
)

validation_dataset = Subset(
    full_training_dataset,
    split_indices["validation"],
)

print("Training examples:", len(train_dataset))
print("Validation examples:", len(validation_dataset))

train_indices = set(split_indices["train"])
validation_indices = set(split_indices["validation"])

assert len(train_dataset) == 45000
assert len(validation_dataset) == 5000
assert len(train_indices) == 45000
assert len(validation_indices) == 5000
assert train_indices.isdisjoint(validation_indices)
assert train_indices | validation_indices == set(range(50000))

print("Split checks passed.")

statistics_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=0,
)

channel_sum = torch.zeros(3, dtype=torch.float64)
channel_squared_sum = torch.zeros(3, dtype=torch.float64)
pixels_per_channel = 0

for images, labels in statistics_loader:
    images = images.to(dtype=torch.float64)

    channel_sum += images.sum(dim=(0, 2, 3))
    channel_squared_sum += images.square().sum(dim=(0, 2, 3))

    pixels_per_channel += (
        images.shape[0] * images.shape[2] * images.shape[3]
    )

channel_mean = channel_sum / pixels_per_channel

channel_variance = (
    channel_squared_sum / pixels_per_channel
    - channel_mean.square()
)

channel_std = channel_variance.clamp(min=0).sqrt()

print("Training channel means:", channel_mean.tolist())
print("Training channel standard deviations:", channel_std.tolist())

statistics = {
    "source": "45000-image training split",
    "split_seed": 42,
    "mean": channel_mean.tolist(),
    "std": channel_std.tolist(),
}

statistics_path = data_dir / "normalization.json"

with statistics_path.open("w") as file:
    json.dump(statistics, file, indent=2)

print("Normalization statistics saved.")

normalized_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=statistics["mean"],
        std=statistics["std"],
    ),
])

normalized_full_dataset = datasets.CIFAR10(
    root=str(data_dir),
    train=True,
    download=False,
    transform=normalized_transform,
)

normalized_train_dataset = Subset(
    normalized_full_dataset,
    split_indices["train"],
)

normalized_validation_dataset = Subset(
    normalized_full_dataset,
    split_indices["validation"],
)

batch_size = 64

training_order_generator = torch.Generator().manual_seed(0)

train_loader = DataLoader(
    normalized_train_dataset,
    batch_size=batch_size,
    shuffle=True,
    generator=training_order_generator,
    num_workers=0,
    drop_last=False,
)

validation_loader = DataLoader(
    normalized_validation_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    drop_last=False,
)

images, labels = next(iter(train_loader))

print("Training batch image shape:", images.shape)
print("Training batch label shape:", labels.shape)
print("Image data type:", images.dtype)
print("Label data type:", labels.dtype)

assert images.shape == (64, 3, 32, 32)
assert labels.shape == (64,)
assert torch.isfinite(images).all().item()
assert ((labels >= 0) & (labels < 10)).all().item()

print("Batch checks passed.")