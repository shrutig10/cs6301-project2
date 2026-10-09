from pathlib import Path
import json

import torch
from torch.utils.data import Subset

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