import torch
from torch.utils.data import DataLoader, Subset
import numpy as np


def evaluate_subset(
    subset_indices: np.ndarray,
    dataloader: DataLoader,
    model: torch.nn.Module,
    device: torch.device = torch.device("cpu")
) -> float:
    """
    Evaluates model accuracy on the validation dataset using a subset of training data.

    Args:
        subset_indices (np.ndarray): Boolean numpy array representing selected indices.
        dataloader (DataLoader): Validation dataloader.
        model (torch.nn.Module): Model to train and evaluate.
        device (torch.device): Device to use for training and evaluation (default: CPU).

    Returns:
        float: Accuracy of the model on the validation dataset.
    """
    train_dataset = dataloader.dataset.dataset

    selected_indices = np.where(subset_indices)[0]

    subset = Subset(train_dataset, selected_indices.tolist())
    train_loader = DataLoader(subset, batch_size=16, shuffle=True)

    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(1):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    return correct / total
