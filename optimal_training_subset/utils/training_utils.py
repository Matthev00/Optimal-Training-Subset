import torch
from torch.utils.data import DataLoader, Subset


def evaluate_model(
    subset_indices: list[int],
    dataloader: DataLoader,
    model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
) -> float:
    """
    Evaluates model accuracy on the validation dataset using a subset of training data.

    Args:
        subset_indices (List[int]): Indices representing the selected training subset.
        dataloader (DataLoader): Validation dataloader.

    Returns:
        float: Accuracy of the model on the validation dataset.
    """
    train_dataset = dataloader.dataset.dataset
    subset = Subset(train_dataset, subset_indices)
    train_loader = DataLoader(subset, batch_size=32, shuffle=True)

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
