from torch import optim
from tqdm import tqdm
import torch.nn as nn
import torch
from torch.utils.data import DataLoader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 2,
    learning_rate: float = 1e-3,
    device: torch.device = torch.device("cuda"),
) -> None:
    """
    Trains the given neural network model on the provided dataset.

    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader providing the training dataset.
        num_epochs (int, optional): Number of training epochs. Default is 2.
        learning_rate (float, optional): Learning rate for the optimizer. Default is 1e-3.

    Returns:
        None
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0
        for images, labels in train_loader:

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()


def calculate_balanced_accuracy(class_correct: list[int], class_total: list[int]) -> float:
    """
    Calculates the Balanced Accuracy metric based on class-level accuracies.

    Args:
        class_correct (list[int]): A list of correctly classified samples per class.
        class_total (list[int]): A list of total samples per class.

    Returns:
        float: The Balanced Accuracy, or 0 if no samples are available for any class.
    """
    assert len(class_correct) == len(class_total), "class_correct and class_total must have the same length."
    
    accuracies = [
        class_correct[i] / class_total[i] for i in range(len(class_total)) if class_total[i] > 0
    ]

    if len(accuracies) == 0:
        return 0.0

    return sum(accuracies) / len(accuracies)


def loss_fn(balanced_accuracy: float, alpha: float, beta: float, S: int, D: int) -> float:
    """
    Computes the loss function based on Balanced Accuracy and a subset penalty.

    Args:
        balanced_accuracy (float): The Balanced Accuracy of the model on the validation dataset.
        alpha (float): Weight for the Balanced Accuracy component of the loss.
        beta (float): Weight for the subset size penalty component of the loss.
        S (int): Size of the current subset.
        D (int): Size of the entire dataset.

    Returns:
        float: The computed loss value.
    """
    return alpha * balanced_accuracy - beta * (S / D)


def validate_model(
    model: nn.Module,
    val_loader: DataLoader,
    alpha: float = 1.0,
    beta: float = 0.5,
    S: int = 10,
    D: int = 10,
) -> tuple[float]:
    """
    Validates the model on the validation dataset and computes the loss based on Balanced Accuracy.

    Args:
        model (nn.Module): The neural network model to validate.
        val_loader (DataLoader): DataLoader for the validation dataset.
        alpha (float): Weight for Balanced Accuracy in the loss function.
        beta (float): Weight for subset size penalty in the loss function.
        S (int): Size of the current subset.
        D (int): Size of the full dataset.
        device (torch.device): Device for computation ('cuda' or 'cpu').

    Returns:
        Tuple[float, float]: Loss and Balanced Accuracy.
    """

    model.eval()

    num_classes = len(val_loader.dataset.targets.unique())

    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.inference_mode():
        for images, labels in val_loader:
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    balanced_accuracy = calculate_balanced_accuracy(class_correct, class_total)
    loss = loss_fn(balanced_accuracy, alpha, beta, S, D)

    return loss, balanced_accuracy
