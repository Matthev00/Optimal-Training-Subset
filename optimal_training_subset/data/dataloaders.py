import torch
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import numpy as np
from typing import Optional
import os
from optimal_training_subset.data.datasets import get_dataset, GPUDataset


def get_dataloaders(
    dataset_name: str = "FashionMNIST",
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    device: str = "cuda",
) -> tuple[Dataset, DataLoader, DataLoader, DataLoader]:

    torch.manual_seed(seed)

    if dataset_name == "FashionMNIST":
        train_dataset, test_dataset = get_dataset(dataset_name=dataset_name)
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
    elif dataset_name == "CIFAR10":
        (0.5, 0.5, 0.5)
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_dataset, test_dataset = get_dataset(dataset_name=dataset_name)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    val_size = int(val_split * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_dataset = GPUDataset(train_dataset, transform, device)
    val_dataset = GPUDataset(val_dataset, transform, device)
    test_dataset = GPUDataset(test_dataset, transform, device)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_dataset, train_dataloader, val_dataloader, test_dataloader


def get_subset_loader(
    dataset: Dataset, mask: np.ndarray, batch_size: int = 32, num_workers: int = 0
) -> DataLoader:
    assert mask.dtype == bool, "Mask must be a boolean array."
    assert len(mask) == len(dataset), "Mask and dataset must have the same length."
    indices = np.where(mask)[0]
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def main():
    num_workers = 0
    dataset_name = "CIFAR10"
    train_dataset, train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        dataset_name=dataset_name,
        num_workers=num_workers,
        device="cuda",
    )

    total_samples = len(train_dataset)
    mask = np.zeros(total_samples, dtype=bool)
    mask[:500] = True

    subset_loader = get_subset_loader(train_dataset, mask, num_workers=num_workers)

    print(f"Train dataloader: {len(train_dataloader)}")
    print(f"Validation dataloader: {len(val_dataloader)}")
    print(f"Test dataloader: {len(test_dataloader)}")
    print(f"Subset dataloader: {len(subset_loader)}")

    X, y = next(iter(subset_loader))
    print(X)


if __name__ == "__main__":
    main()
