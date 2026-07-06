import json
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from dataset import get_dataloader, get_dataset
from model import SimpleCNN


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_clinical_metrics(conf_matrix):
    tn, fp, fn, tp = conf_matrix.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    false_negative_rate = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    false_positive_rate = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    positive_predictive_value = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    negative_predictive_value = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    balanced_accuracy = (sensitivity + specificity) / 2

    return {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "false_negative_rate": false_negative_rate,
        "false_positive_rate": false_positive_rate,
        "positive_predictive_value": positive_predictive_value,
        "negative_predictive_value": negative_predictive_value,
        "balanced_accuracy": balanced_accuracy,
    }


def evaluate_model(model, test_loader, device):
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.tolist())
            all_predictions.extend(predicted.cpu().tolist())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )
    recall = recall_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )
    f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    conf_matrix = confusion_matrix(all_labels, all_predictions)
    clinical_metrics = compute_clinical_metrics(conf_matrix)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix.tolist(),
        "clinical_metrics": clinical_metrics,
    }


def save_json(data, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


def main():
    data_dir = "data/chest_xray"
    model_path = "results/models/simple_cnn.pth"
    metrics_output_path = "results/metrics/test_metrics.json"
    clinical_output_path = "results/metrics/clinical_metrics.json"
    batch_size = 32

    device = get_device()
    print("Device:", device)

    test_dataset = get_dataset(
        data_dir=data_dir,
        split="test",
    )

    test_loader = get_dataloader(
        data_dir=data_dir,
        split="test",
        batch_size=batch_size,
        shuffle=False,
    )

    num_classes = len(test_dataset.classes)

    print("Classes:", test_dataset.classes)
    print("Number of test images:", len(test_dataset))

    model = SimpleCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
    )

    save_json(metrics, metrics_output_path)
    save_json(metrics["clinical_metrics"], clinical_output_path)

    print("Test Metrics:")
    print("Accuracy:", metrics["accuracy"])
    print("Precision:", metrics["precision"])
    print("Recall:", metrics["recall"])
    print("F1-score:", metrics["f1_score"])
    print("Confusion Matrix:", metrics["confusion_matrix"])

    print("\nClinical Metrics:")
    for key, value in metrics["clinical_metrics"].items():
        print(f"{key}: {value}")

    print(f"\nMetrics saved to {metrics_output_path}")
    print(f"Clinical metrics saved to {clinical_output_path}")


if __name__ == "__main__":
    main()