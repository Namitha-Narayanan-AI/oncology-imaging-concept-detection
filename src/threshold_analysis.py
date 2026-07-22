from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
)

from dataset import get_dataloader, get_dataset
from model import SimpleCNN


def get_device():
    """
    Return the best available device for PyTorch inference.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def collect_probabilities(
    model,
    dataloader,
    device,
    positive_class_index,
):
    """
    Run the test images through the trained model and collect:

    1. The true class label for every image.
    2. The model's predicted probability of the positive class.

    In this project, PNEUMONIA is the positive class.
    """
    model.eval()

    all_labels = []
    all_positive_probabilities = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            logits = model(images)

            # Convert raw model outputs into class probabilities.
            probabilities = torch.softmax(logits, dim=1)

            # Keep only the probability assigned to PNEUMONIA.
            positive_probabilities = probabilities[:, positive_class_index]

            all_labels.extend(labels.tolist())

            all_positive_probabilities.extend(
                positive_probabilities.cpu().tolist()
            )

    return all_labels, all_positive_probabilities


def calculate_threshold_metrics(
    true_labels,
    positive_probabilities,
    threshold,
    negative_class_index,
    positive_class_index,
):
    """
    Apply one decision threshold to the pneumonia probabilities
    and calculate the resulting classification metrics.
    """

    predictions = []

    for probability in positive_probabilities:
        if probability >= threshold:
            predictions.append(positive_class_index)
        else:
            predictions.append(negative_class_index)

    matrix = confusion_matrix(
        true_labels,
        predictions,
        labels=[
            negative_class_index,
            positive_class_index,
        ],
    )

    tn, fp, fn, tp = matrix.ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    balanced_accuracy = (
        sensitivity + specificity
    ) / 2

    accuracy = accuracy_score(
        true_labels,
        predictions,
    )

    precision = precision_score(
        true_labels,
        predictions,
        pos_label=positive_class_index,
        zero_division=0,
    )

    f1 = f1_score(
        true_labels,
        predictions,
        pos_label=positive_class_index,
        zero_division=0,
    )

    return {
        "threshold": threshold,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "f1_score": f1,
    }


def analyse_thresholds(
    true_labels,
    positive_probabilities,
    negative_class_index,
    positive_class_index,
):
    """
    Evaluate the same model predictions across a range of
    pneumonia decision thresholds.
    """

    # thresholds = [
    #     value / 100
    #     for value in range(5, 100, 5)
    # ]

    thresholds = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        0.96,
        0.97,
        0.98,
        0.99,
        0.995,
        0.999,
    ]

    results = []

    for threshold in thresholds:
        metrics = calculate_threshold_metrics(
            true_labels=true_labels,
            positive_probabilities=positive_probabilities,
            threshold=threshold,
            negative_class_index=negative_class_index,
            positive_class_index=positive_class_index,
        )

        results.append(metrics)

    return pd.DataFrame(results)


def plot_threshold_results(
    results,
    output_path,
):
    """
    Plot sensitivity, specificity and balanced accuracy
    across decision thresholds.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        results["threshold"],
        results["sensitivity"],
        marker="o",
        label="Sensitivity",
    )

    plt.plot(
        results["threshold"],
        results["specificity"],
        marker="o",
        label="Specificity",
    )

    plt.plot(
        results["threshold"],
        results["balanced_accuracy"],
        marker="o",
        label="Balanced Accuracy",
    )

    plt.xlabel("Pneumonia Decision Threshold")
    plt.ylabel("Metric Score")

    plt.title(
        "Clinical Metric Trade-offs Across Decision Thresholds"
    )

    plt.ylim(0.0, 1.05)

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()


def main():
    data_dir = "data/chest_xray"
    model_path = "results/models/simple_cnn.pth"

    metrics_output_path = (
        "results/metrics/threshold_analysis.csv"
    )

    figure_output_path = (
        "results/figures/threshold_analysis.png"
    )

    batch_size = 32

    device = get_device()

    print("Device:", device)

    dataset = get_dataset(
        data_dir=data_dir,
        split="test",
    )

    dataloader = get_dataloader(
        data_dir=data_dir,
        split="test",
        batch_size=batch_size,
        shuffle=False,
    )

    print("Classes:", dataset.classes)
    print("Class mapping:", dataset.class_to_idx)
    print("Number of test images:", len(dataset))

    positive_class_index = (
        dataset.class_to_idx["PNEUMONIA"]
    )

    negative_class_index = (
        dataset.class_to_idx["NORMAL"]
    )

    model = SimpleCNN(
        num_classes=len(dataset.classes)
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
        )
    )

    model = model.to(device)

    true_labels, positive_probabilities = collect_probabilities(
        model=model,
        dataloader=dataloader,
        device=device,
        positive_class_index=positive_class_index,
    )

    results = analyse_thresholds(
        true_labels=true_labels,
        positive_probabilities=positive_probabilities,
        negative_class_index=negative_class_index,
        positive_class_index=positive_class_index,
    )

    metrics_output_path = Path(
        metrics_output_path
    )

    metrics_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        metrics_output_path,
        index=False,
    )

    plot_threshold_results(
        results=results,
        output_path=figure_output_path,
    )

    baseline_result = results.loc[
        (results["threshold"] - 0.50)
        .abs()
        .idxmin()
    ]

    best_balanced_result = results.loc[
        results["balanced_accuracy"].idxmax()
    ]

    print("\nResults at threshold 0.50:")
    print(baseline_result.to_string())

    print("\nHighest exploratory balanced accuracy:")
    print(best_balanced_result.to_string())

    print(
        f"\nMetrics saved to: "
        f"{metrics_output_path}"
    )

    print(
        f"Figure saved to: "
        f"{figure_output_path}"
    )


if __name__ == "__main__":
    main()