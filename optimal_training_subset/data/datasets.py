import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from optimal_training_subset.config import DATASETS_DIR
from typing import Optional


class GPUDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: Optional[transforms.Compose], device: str):
        self.device = device
        self.transform = transform

        self.data = torch.empty((len(dataset), *transform(dataset[0][0]).shape), device=device)
        self.targets = torch.empty(len(dataset), dtype=torch.long, device=device)

        for i, (img, target) in enumerate(dataset):
            self.data[i] = transform(img).to(device)
            self.targets[i] = torch.tensor(target).to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def get_dataset(
    dataset_name: str = "FashionMNIST", transform: Optional[transforms.Compose] = None
) -> tuple[Dataset, Dataset]:
    if dataset_name == "FashionMNIST":
        train_dataset = datasets.FashionMNIST(root=DATASETS_DIR, train=True, download=True)
        test_dataset = datasets.FashionMNIST(root=DATASETS_DIR, train=False, download=True)
    elif dataset_name == "CIFAR100":
        train_dataset = datasets.CIFAR100(
            root=DATASETS_DIR, train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR100(
            root=DATASETS_DIR, train=False, download=True, transform=transform
        )

    return train_dataset, test_dataset
