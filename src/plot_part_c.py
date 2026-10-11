import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    project_dir = Path(__file__).resolve().parent.parent
    results_dir = project_dir / "results"

    configurations = {
        "plain": "Plain deep CNN",
        "residual": "Residual CNN",
    }

    metrics = [
        ("avg_train_loss", "Training loss", 1),
        ("avg_val_loss", "Validation loss", 1),
        ("train_accuracy", "Training accuracy (%)", 100),
        ("val_accuracy", "Validation accuracy (%)", 100),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    for configuration, label in configurations.items():
        runs = []

        for seed in [0, 1, 2]:
            path = results_dir / f"partC_{configuration}_seed{seed}.json"

            with path.open("r") as file:
                runs.append(json.load(file))

        epochs = [row["epoch"] for row in runs[0]["history"]]

        for run in runs:
            assert [row["epoch"] for row in run["history"]] == epochs

        best_accuracies = np.array([
            run["best_val_accuracy"] for run in runs
        ]) * 100

        best_losses = np.array([
            run["best_val_loss"] for run in runs
        ])

        print(
            f"{label}: best validation accuracy = "
            f"{best_accuracies.mean():.2f}% "
            f"± {best_accuracies.std(ddof=1):.2f} percentage points"
        )

        print(
            f"{label}: loss at selected checkpoints = "
            f"{best_losses.mean():.4f} "
            f"± {best_losses.std(ddof=1):.4f}"
        )

        for ax, (metric, title, scale) in zip(axes.flat, metrics):
            values = np.array([
                [row[metric] for row in run["history"]]
                for run in runs
            ]) * scale

            mean = values.mean(axis=0)
            standard_deviation = values.std(axis=0, ddof=1)

            line, = ax.plot(epochs, mean, label=label)

            ax.fill_between(
                epochs,
                mean - standard_deviation,
                mean + standard_deviation,
                color=line.get_color(),
                alpha=0.15,
            )

            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy (%)" if scale == 100 else "Loss")
            ax.grid(alpha=0.25)

    for ax in axes.flat:
        ax.legend()

    fig.suptitle("Part C: mean curves ± sample SD across three seeds")
    fig.tight_layout()

    fig.savefig(results_dir / "partC_learning_curves.png", dpi=200)
    fig.savefig(results_dir / "partC_learning_curves.pdf")

    plt.close(fig)
    print("Learning curves saved in results.")


if __name__ == "__main__":
    main()