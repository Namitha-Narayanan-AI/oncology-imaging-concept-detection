from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

def get_dataset(data_dir, split, image_size=224):
    split_path = Path(data_dir) / split

    if not split_path.exists():
        raise FileNotFoundError(f'Dataset not found at {split_path}')
    transform = get_transform(image_size=image_size)

    dataset = datasets.ImageFolder(root=split_path, transform=transform)
    return dataset

def get_dataloader(data_dir, split, image_size=224, batch_size=32, shuffle=False):
    dataset = get_dataset(data_dir=data_dir, split=split, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

if __name__ == "__main__":
    data_dir = "data/chest_xray"

    dataset = get_dataset(
        data_dir=data_dir,
        split="train",
        image_size=224
    )

    print("Dataset loaded successfully")
    print("Number of images:", len(dataset))
    print("Classes:", dataset.classes)
    print("Class to index:", dataset.class_to_idx)

    image, label = dataset[0]

    print("First image tensor shape:", image.shape)
    print("First label:", label)
    print("First label name:", dataset.classes[label])