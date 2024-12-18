import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from optimal_training_subset.config import DATASETS_DIR
from typing import Optional


class GPUDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: Optional[transforms.Compose], device: str):
        self.device = device
        self.transform = transform
        shape = transform(dataset[0][0]).shape

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
    dataset_name: str = "FashionMNIST") -> tuple[Dataset, Dataset]:
    if dataset_name == "FashionMNIST":
        train_dataset = datasets.FashionMNIST(root=DATASETS_DIR, train=True, download=True)
        test_dataset = datasets.FashionMNIST(root=DATASETS_DIR, train=False, download=True)
    elif dataset_name == "CIFAR10":
        train_dataset = datasets.CIFAR10(
            root=DATASETS_DIR, train=True, download=True
        )
        test_dataset = datasets.CIFAR10(
            root=DATASETS_DIR, train=False, download=True
        )

    return train_dataset, test_dataset

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


    train_dataset, _ = get_dataset(dataset_name="FashionMNIST", transform=transform)
    train_dataset.data = train_dataset.data.to(device)
    train_dataset.targets = train_dataset.targets.to(device)

    print(train_dataset[1])

    train_dataloader = DataLoader(train_dataset, batch_size=32)
    print(train_dataset.data.device)
    sample = next(iter(train_dataloader))[0]
    print(sample[0].shape, sample[1].shape)

if __name__ == "__main__":
    main()
