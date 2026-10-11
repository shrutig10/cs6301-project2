import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from deep_models import DeepCNN
from engine import evaluate


def main():
    project_dir = Path(__file__).resolve().parent.parent
    data_dir = project_dir / "data"
    checkpoints_dir = project_dir / "checkpoints"
    results_dir = project_dir / "results"

    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "final_test_results.json"

    if output_path.exists():
        raise FileExistsError(
            "Final test results already exist. Read the saved results."
        )

    with (data_dir / "normalization.json").open("r") as file:
        statistics = json.load(file)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=statistics["mean"],
            std=statistics["std"],
        ),
    ])

    test_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=False,
        transform=test_transform,
    )

    assert len(test_dataset) == 10000

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    loss_function = nn.CrossEntropyLoss()

    seed_results = []

    print("Evaluation device:", device)
    print("Test examples:", len(test_dataset))

    for seed in [0, 1, 2]:
        checkpoint_path = (
            checkpoints_dir / f"partC_residual_seed{seed}.pt"
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        settings = checkpoint["settings"]

        assert settings["model"] == "residual"
        assert settings["seed"] == seed
        assert settings["augmentation"] is True
        assert settings["normalization"] == statistics

        model = DeepCNN(residual=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)

        metrics = evaluate(
            model,
            test_loader,
            loss_function,
            device,
        )

        seed_results.append({
            "seed": seed,
            "selected_epoch": checkpoint["epoch"],
            "test_accuracy": metrics["accuracy"],
            "test_loss": metrics["loss"],
            "checkpoint": checkpoint_path.name,
        })

        print(
            f"Seed {seed}, checkpoint epoch {checkpoint['epoch']}: "
            f"test accuracy={metrics['accuracy']:.2%}, "
            f"test loss={metrics['loss']:.4f}"
        )

    accuracies = np.array([
        result["test_accuracy"] for result in seed_results
    ])

    summary = {
        "selected_configuration": "Residual CNN",
        "selection_metric": "Mean best-validation accuracy across seeds",
        "test_examples": len(test_dataset),
        "device": str(device),
        "seed_results": seed_results,
        "mean_test_accuracy": float(accuracies.mean()),
        "sample_std_test_accuracy": float(accuracies.std(ddof=1)),
    }

    with output_path.open("w") as file:
        json.dump(summary, file, indent=2)

    print(
        f"Final test accuracy: {accuracies.mean() * 100:.2f}% "
        f"± {accuracies.std(ddof=1) * 100:.2f} percentage points"
    )
    print("Final test results saved.")


if __name__ == "__main__":
    main()