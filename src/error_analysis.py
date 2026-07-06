import csv
from pathlib import Path

import torch

from dataset import get_dataloader, get_dataset
from model import SimpleCNN


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def classify_error(true_label, predicted_label, positive_class_index):
    if true_label == predicted_label:
        return "correct"

    if true_label != positive_class_index and predicted_label == positive_class_index:
        return "false_positive"

    if true_label == positive_class_index and predicted_label != positive_class_index:
        return "false_negative"

    return "misclassification"


def main():
    data_dir = "data/chest_xray"
    model_path = "results/models/simple_cnn.pth"
    output_path = Path("results/metrics/error_analysis.csv")
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

    class_names = test_dataset.classes
    positive_class_index = class_names.index("PNEUMONIA")

    model = SimpleCNN(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    rows = []
    sample_index = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_labels = torch.argmax(probabilities, dim=1).cpu()

            for i in range(len(labels)):
                true_label = labels[i].item()
                predicted_label = predicted_labels[i].item()
                confidence = probabilities[i, predicted_label].item()
                image_path = test_dataset.samples[sample_index][0]

                error_type = classify_error(
                    true_label=true_label,
                    predicted_label=predicted_label,
                    positive_class_index=positive_class_index,
                )

                if error_type != "correct":
                    rows.append({
                        "sample_index": sample_index,
                        "image_path": image_path,
                        "true_label": class_names[true_label],
                        "predicted_label": class_names[predicted_label],
                        "confidence": confidence,
                        "error_type": error_type,
                    })

                sample_index += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_index",
                "image_path",
                "true_label",
                "predicted_label",
                "confidence",
                "error_type",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    false_positives = sum(row["error_type"] == "false_positive" for row in rows)
    false_negatives = sum(row["error_type"] == "false_negative" for row in rows)

    print(f"Saved {len(rows)} misclassified examples to {output_path}")
    print(f"False positives: {false_positives}")
    print(f"False negatives: {false_negatives}")


if __name__ == "__main__":
    main()