from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay

from dataset import get_dataloader, get_dataset
from model import SimpleCNN


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def unnormalize_image(image):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    image = image.cpu() * std + mean
    image = image.clamp(0, 1)

    return image.permute(1, 2, 0)


def collect_predictions(model, dataloader, device):
    model.eval()

    images_list = []
    labels_list = []
    predictions_list = []

    with torch.no_grad():
        for images, labels in dataloader:
            images_device = images.to(device)

            outputs = model(images_device)
            _, predictions = torch.max(outputs, 1)

            images_list.extend(images.cpu())
            labels_list.extend(labels.cpu().tolist())
            predictions_list.extend(predictions.cpu().tolist())

    return images_list, labels_list, predictions_list


def plot_confusion_matrix(labels, predictions, class_names, output_path):
    display = ConfusionMatrixDisplay.from_predictions(
        labels,
        predictions,
        display_labels=class_names,
        cmap="Blues",
        values_format="d"
    )

    display.ax_.set_title("Confusion Matrix - SimpleCNN")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_prediction_examples(images, labels, predictions, class_names, output_path, max_images=8):
    selected = []

    for image, label, prediction in zip(images, labels, predictions):
        selected.append((image, label, prediction))

        if len(selected) == max_images:
            break

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()

    for ax, (image, label, prediction) in zip(axes, selected):
        image = unnormalize_image(image)

        true_name = class_names[label]
        predicted_name = class_names[prediction]

        ax.imshow(image, cmap="gray")
        ax.axis("off")

        color = "green" if label == prediction else "red"
        ax.set_title(
            f"True: {true_name}\nPred: {predicted_name}",
            color=color,
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    data_dir = "data/chest_xray"
    model_path = "results/models/simple_cnn.pth"
    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print("Device:", device)

    test_dataset = get_dataset(
        data_dir=data_dir,
        split="test"
    )

    test_loader = get_dataloader(
        data_dir=data_dir,
        split="test",
        batch_size=32,
        shuffle=False
    )

    class_names = test_dataset.classes
    num_classes = len(class_names)

    model = SimpleCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    images, labels, predictions = collect_predictions(
        model=model,
        dataloader=test_loader,
        device=device
    )

    confusion_matrix_path = figures_dir / "confusion_matrix.png"
    prediction_examples_path = figures_dir / "prediction_examples.png"

    plot_confusion_matrix(
        labels=labels,
        predictions=predictions,
        class_names=class_names,
        output_path=confusion_matrix_path
    )

    plot_prediction_examples(
        images=images,
        labels=labels,
        predictions=predictions,
        class_names=class_names,
        output_path=prediction_examples_path
    )

    print(f"Saved confusion matrix to {confusion_matrix_path}")
    print(f"Saved prediction examples to {prediction_examples_path}")


if __name__ == "__main__":
    main()