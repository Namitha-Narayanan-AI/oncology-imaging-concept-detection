from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dataset import get_transform, get_dataset
from model import SimpleCNN


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.forward_hook = self.target_layer.register_forward_hook(
            self.save_activations
        )
        self.backward_hook = self.target_layer.register_full_backward_hook(
            self.save_gradients
        )

    def save_activations(self, module, input_tensor, output_tensor):
        self.activations = output_tensor.detach()

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class):
        self.model.zero_grad()

        output = self.model(input_tensor)
        target_score = output[:, target_class]
        target_score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()

        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def close(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_image_for_model(image_path, image_size=224):
    image = Image.open(image_path).convert("RGB")
    transform = get_transform(image_size=image_size)
    tensor = transform(image).unsqueeze(0)
    return image, tensor


def resize_for_display(image, image_size=224):
    image = image.resize((image_size, image_size))
    return np.array(image).astype(np.float32) / 255.0


def save_gradcam_figure(
    original_image,
    heatmap,
    true_label,
    predicted_label,
    confidence,
    error_type,
    output_path,
):
    original_array = resize_for_display(original_image)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original_array)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap="jet")
    plt.title("Grad-CAM")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(original_array)
    plt.imshow(heatmap, cmap="jet", alpha=0.45)
    plt.title("Overlay")
    plt.axis("off")

    plt.suptitle(
        f"{error_type} | True: {true_label} | Pred: {predicted_label} | Conf: {confidence:.3f}",
        fontsize=10,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    data_dir = "data/chest_xray"
    model_path = "results/models/simple_cnn.pth"
    output_dir = Path("results/figures/gradcam")
    image_size = 224
    max_examples_per_error_type = 4

    device = get_device()
    print("Device:", device)

    test_dataset = get_dataset(
        data_dir=data_dir,
        split="test",
        image_size=image_size,
    )

    class_names = test_dataset.classes
    positive_class_index = class_names.index("PNEUMONIA")

    model = SimpleCNN(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    target_layer = model.features[6]
    gradcam = GradCAM(model=model, target_layer=target_layer)

    selected_counts = {
        "false_positive": 0,
        "false_negative": 0,
    }

    saved_count = 0

    for sample_index, (image_path, true_label) in enumerate(test_dataset.samples):
        original_image, input_tensor = load_image_for_model(
            image_path=image_path,
            image_size=image_size,
        )

        input_tensor = input_tensor.to(device)

        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_label].item()

        if true_label == predicted_label:
            continue

        if true_label != positive_class_index and predicted_label == positive_class_index:
            error_type = "false_positive"
        elif true_label == positive_class_index and predicted_label != positive_class_index:
            error_type = "false_negative"
        else:
            error_type = "misclassification"

        if error_type not in selected_counts:
            continue

        if selected_counts[error_type] >= max_examples_per_error_type:
            continue

        heatmap = gradcam.generate(
            input_tensor=input_tensor,
            target_class=predicted_label,
        )

        true_label_name = class_names[true_label]
        predicted_label_name = class_names[predicted_label]

        output_path = (
            output_dir
            / error_type
            / f"{sample_index}_{true_label_name}_pred_{predicted_label_name}.png"
        )

        save_gradcam_figure(
            original_image=original_image,
            heatmap=heatmap,
            true_label=true_label_name,
            predicted_label=predicted_label_name,
            confidence=confidence,
            error_type=error_type,
            output_path=output_path,
        )

        selected_counts[error_type] += 1
        saved_count += 1

        print(f"Saved {error_type} Grad-CAM: {output_path}")

        if all(
            count >= max_examples_per_error_type
            for count in selected_counts.values()
        ):
            break

    gradcam.close()

    print(f"Saved {saved_count} Grad-CAM figures to {output_dir}")
    print("Selected counts:", selected_counts)


if __name__ == "__main__":
    main()