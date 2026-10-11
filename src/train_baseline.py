import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from baseline import BaselineCNN
from engine import evaluate, train_one_epoch


def main():
    seed = 0
    epochs = 2
    batch_size = 64
    learning_rate = 0.001

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    project_dir = Path(__file__).resolve().parent.parent
    data_dir = project_dir / "data"
    results_dir = project_dir / "results"
    checkpoints_dir = project_dir / "checkpoints"

    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    run_name = "baseline_trial_seed0"
    log_path = results_dir / f"{run_name}.json"
    checkpoint_path = checkpoints_dir / f"{run_name}.pt"

    # Prevent accidentally replacing an existing run.
    if log_path.exists() or checkpoint_path.exists():
        raise FileExistsError(
            "This run already exists. Choose a new run_name."
        )

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

    train_dataset = Subset(full_dataset, split_indices["train"])
    validation_dataset = Subset(
        full_dataset,
        split_indices["validation"],
    )

    order_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=order_generator,
        num_workers=0,
        drop_last=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model = BaselineCNN().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    settings = {
        "model": "BaselineCNN",
        "seed": seed,
        "split_seed": 42,
        "epochs": epochs,
        "batch_size": batch_size,
        "optimizer": "Adam",
        "learning_rate": learning_rate,
        "weight_decay": 0.0,
        "augmentation": False,
        "normalization": statistics,
        "device": str(device),
        "torch_version": str(torch.__version__),
    }

    history = []
    best_accuracy = -1.0
    best_loss = float("inf")
    best_epoch = None

    print("Training device:", device)

    for epoch in range(1, epochs + 1):
        start_time = time.perf_counter()

        train_metrics = train_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            device,
        )

        validation_metrics = evaluate(
            model,
            validation_loader,
            loss_function,
            device,
        )

        seconds = time.perf_counter() - start_time

        accuracy = validation_metrics["accuracy"]
        validation_loss = validation_metrics["loss"]

        improved = (
            accuracy > best_accuracy
            or (
                accuracy == best_accuracy
                and validation_loss < best_loss
            )
        )

        if improved:
            best_accuracy = accuracy
            best_loss = validation_loss
            best_epoch = epoch

            # CPU copies make the saved weights portable.
            saved_weights = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

            torch.save({
                "model_state_dict": saved_weights,
                "settings": settings,
                "epoch": epoch,
                "validation_accuracy": accuracy,
                "validation_loss": validation_loss,
            }, checkpoint_path)

        history.append({
            "epoch": epoch,
            "avg_train_loss": train_metrics["loss"],
            "avg_val_loss": validation_loss,
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": accuracy,
            "seconds": seconds,
        })

        # Save after every epoch so completed results are preserved.
        with log_path.open("w") as file:
            json.dump({
                "settings": settings,
                "history": history,
                "best_epoch": best_epoch,
                "best_val_accuracy": best_accuracy,
                "best_val_loss": best_loss,
            }, file, indent=2)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train loss={train_metrics['loss']:.4f} | "
            f"train accuracy={train_metrics['accuracy']:.2%} | "
            f"val loss={validation_loss:.4f} | "
            f"val accuracy={accuracy:.2%} | "
            f"time={seconds:.1f}s"
        )

    print(
        f"Best validation accuracy: {best_accuracy:.2%} "
        f"at epoch {best_epoch}"
    )
    print("Trial complete. Logs and best checkpoint saved.")


if __name__ == "__main__":
    main()