import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloader, get_dataset
from model import SimpleCNN


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    average_loss = total_loss / len(train_loader)
    accuracy = correct / total

    return average_loss, accuracy


def validate(model, val_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    average_loss = total_loss / len(val_loader)
    accuracy = correct / total

    return average_loss, accuracy


def main():
    data_dir = "data/chest_xray"
    batch_size = 32
    num_epochs = 3
    learning_rate = 0.001

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    train_loader = get_dataloader(
        data_dir=data_dir,
        split="train",
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = get_dataloader(
        data_dir=data_dir,
        split="val",
        batch_size=batch_size,
        shuffle=False
    )

    train_dataset = get_dataset(
        data_dir=data_dir,
        split="train"
    )

    num_classes = len(train_dataset.classes)

    print("Classes:", train_dataset.classes)
    print("Number of classes:", num_classes)

    model = SimpleCNN(num_classes=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        val_loss, val_accuracy = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device
        )

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        print("-" * 40)

    torch.save(model.state_dict(), "results/models/simple_cnn.pth")
    print("Model saved to results/models/simple_cnn.pth")


if __name__ == "__main__":
    main()