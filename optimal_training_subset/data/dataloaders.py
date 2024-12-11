import torch
from torchvision import datasets, transforms
from torchvision.models import MobileNet_V3_Small_Weights
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import numpy as np
from optimal_training_subset.config import DATASETS_DIR
from typing import Optional
import os


def get_dataloaders(
    dataset_name: str = "FashionMNIST",
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    transform: Optional[transforms.Compose] = None,
) -> tuple[Dataset, DataLoader, DataLoader, DataLoader]:

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


def get_dataset(
    transform: Optional[transforms.Compose] = None, dataset_name: str = "FashionMNIST"
) -> tuple[Dataset, Dataset]:
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
    # train_dataset.targets = train_dataset.targets.to("cuda")
    # test_dataset.targets = test_dataset.targets.to("cuda")
    # train_dataset.data = train_dataset.data.to("cuda")
    # test_dataset.data = test_dataset.data.to("cuda")

    return train_dataset, test_dataset


def main():
    num_workers = os.cpu_count()
    dataset_name = "FashionMNIST"
    train_dataset, train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        dataset_name=dataset_name,
        num_workers=num_workers,
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
    print(X.device, y.device)


if __name__ == "__main__":
    main()
