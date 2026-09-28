from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


RESULTS_DIR = Path("results")


def load_scalars(log_dir):
    """Load all scalar metrics from a TensorBoard run."""
    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()

    scalars = {}

    for tag in accumulator.Tags()["scalars"]:
        events = accumulator.Scalars(tag)
        scalars[tag] = {
            "steps": [event.step for event in events],
            "values": [event.value for event in events],
        }

    return scalars


def plot_metric(runs, tag, title, ylabel, output_file):
    """Plot a metric for multiple model runs."""
    plt.figure(figsize=(8, 5))

    for model_name, run in runs.items():
        plt.plot(
            run[tag]["steps"],
            run[tag]["values"],
            marker="o",
            label=model_name,
        )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    runs = {
        "AlexNet-5": load_scalars("runs/AlexNet5"),
        "AlexNet-7": load_scalars("runs/AlexNet7"),
    }

    # Training and test accuracy
    plt.figure(figsize=(8, 5))

    for model_name, run in runs.items():
        plt.plot(
            run["Accuracy/Train"]["steps"],
            run["Accuracy/Train"]["values"],
            marker="o",
            label=f"{model_name} Train",
        )

        plt.plot(
            run["Accuracy/Test"]["steps"],
            run["Accuracy/Test"]["values"],
            marker="o",
            label=f"{model_name} Test",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Test Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "accuracy_curves.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Training and test loss
    plt.figure(figsize=(8, 5))

    for model_name, run in runs.items():
        plt.plot(
            run["Loss/Train"]["steps"],
            run["Loss/Train"]["values"],
            marker="o",
            label=f"{model_name} Train",
        )

        plt.plot(
            run["Loss/Test"]["steps"],
            run["Loss/Test"]["values"],
            marker="o",
            label=f"{model_name} Test",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training and Test Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "loss_curves.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("Figures saved successfully:")
    print(f"- {RESULTS_DIR / 'accuracy_curves.png'}")
    print(f"- {RESULTS_DIR / 'loss_curves.png'}")


if __name__ == "__main__":
    main()