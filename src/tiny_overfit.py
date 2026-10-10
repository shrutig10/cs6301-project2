import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from baseline import BaselineCNN


def main():
    torch.manual_seed(0)

    project_dir = Path(__file__).resolve().parent.parent
    data_dir = project_dir / "data"

    with (data_dir / "split_seed42.json").open("r") as file:
        split_indices = json.load(file)

    with (data_dir / "normalization.json").open("r") as file:
        statistics = json.load(file)

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=statistics["mean"],
            std=statistics["std"],
        ),
    ])

    full_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=False,
        transform=image_transform,
    )

    tiny_indices = split_indices["train"][:64]
    tiny_dataset = Subset(full_dataset, tiny_indices)

    tiny_loader = DataLoader(
        tiny_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print("Training device:", device)

    images, labels = next(iter(tiny_loader))
    images = images.to(device)
    labels = labels.to(device)

    model = BaselineCNN().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    history = []
    passed = False

    for step in range(1, 1001):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = loss_function(logits, labels)

        assert torch.isfinite(loss).item(), "Non-finite training loss."

        loss.backward()

        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None, (
                    f"Missing gradient: {name}"
                )
                assert torch.isfinite(parameter.grad).all().item(), (
                    f"Non-finite gradient: {name}"
                )

        optimizer.step()

        model.eval()

        with torch.no_grad():
            evaluated_logits = model(images)
            evaluated_loss = loss_function(evaluated_logits, labels)

            assert torch.isfinite(evaluated_loss).item()

            predictions = evaluated_logits.argmax(dim=1)
            accuracy = (predictions == labels).float().mean().item()

        history.append({
            "step": step,
            "loss": evaluated_loss.item(),
            "accuracy": accuracy,
        })

        if step == 1 or step % 25 == 0 or accuracy >= 0.95:
            print(
                f"Step {step}: "
                f"loss={evaluated_loss.item():.4f}, "
                f"accuracy={accuracy:.2%}"
            )

        if accuracy >= 0.95:
            passed = True
            break

    results_dir = project_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "seed": 0,
        "split_seed": 42,
        "device": str(device),
        "learning_rate": 0.001,
        "number_of_examples": 64,
        "indices": tiny_indices,
        "augmentation": False,
        "passed": passed,
        "history": history,
    }

    with (results_dir / "tiny_overfit.json").open("w") as file:
        json.dump(result, file, indent=2)

    if passed:
        print("Tiny-set overfit check passed.")
    else:
        print("Threshold not reached. Inspect the results before continuing.")


if __name__ == "__main__":
    main()