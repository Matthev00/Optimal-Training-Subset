import torch
from torchvision import datasets, transforms
from torchvision.models import MobileNet_V3_Small_Weights
from torch.utils.data import DataLoader, random_split, Dataset
from optimal_training_subset.config import DATASETS_DIR
from typing import Optional


def get_dataloaders(
    dataset_name: str = "FashionMNIST",
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    transform: Optional[transforms] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    
    torch.manual_seed(seed)

    if dataset_name == "FashionMNIST":
        train_dataset, test_dataset = get_dataset(transform=transform, dataset_name=dataset_name)
    elif dataset_name == "CIFAR100":
        train_dataset, test_dataset = get_dataset(transform=transform, dataset_name=dataset_name)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    val_size = int(val_split * len(train_dataset))
    train_size = len(train_dataset) - val_size

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_dataloader, val_dataloader, test_dataloader


def get_dataset(
    transform: Optional[transforms] = None, dataset_name: str = "FashionMNIST"
) -> tuple[Dataset, Dataset, Dataset]:
    if transform is None:
        if dataset_name == "FashionMNIST":
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            )
        else:
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            transform = weights.transforms()
    if dataset_name == "FashionMNIST":
        train_dataset = datasets.FashionMNIST(
            root=DATASETS_DIR, train=True, download=True, transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            root=DATASETS_DIR, train=False, download=True, transform=transform
        )
    elif dataset_name == "CIFAR100":
        train_dataset = datasets.CIFAR100(
            root=DATASETS_DIR, train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR100(
            root=DATASETS_DIR, train=False, download=True, transform=transform
        )
    return train_dataset, test_dataset


def main():
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders()
    print(f"Train dataloader: {len(train_dataloader)}")
    print(f"Validation dataloader: {len(val_dataloader)}")
    print(f"Test dataloader: {len(test_dataloader)}")


if __name__ == "__main__":
    main()
