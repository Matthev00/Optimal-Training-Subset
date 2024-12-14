from torch import optim
from tqdm import tqdm
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import balanced_accuracy_score
import numpy as np
from optimal_training_subset.data.dataloaders import get_subset_loader
import mlflow
from functools import wraps
from typing import Optional


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

    model.train()
    for epoch in tqdm(range(num_epochs)):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


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
    D: int = 1000,
    device: torch.device = torch.device("cpu"),
) -> tuple[float, float]:
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
    model.to(device)

    all_labels = []
    all_predictions = []

    with torch.inference_mode():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    balanced_accuracy = balanced_accuracy_score(all_labels, all_predictions)
    loss = loss_fn(balanced_accuracy, alpha, beta, S, D)

    return loss


def mlflow_logger(func):
    @wraps(func)
    def wrapper(*args, **kwargs) -> float:
        enable_mlflow = kwargs.get("enable_mlflow", False)
        if not enable_mlflow:
            return func(*args, **kwargs)

        experiment_name = kwargs.get("experiment_name", "default")
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            mlflow.log_param("D", kwargs.get("dataset_size"))
            subset_size = np.sum(args[0])
            mlflow.log_param("S", subset_size)

            loss = func(*args, **kwargs)
            mlflow.log_metric("loss", loss)
            log = kwargs.get("log")
            if log:
                mlflow.log_metric("Generation", log["generation"])
                mlflow.log_metric("Best Fitness", log["best_fitness"])
            return loss

    return wrapper


@mlflow_logger
def fitness_function(
    individual: np.ndarray,
    model: nn.Module,
    num_workers: int,
    train_dataset: Dataset,
    val_dataloader: DataLoader,
    dataset_size: int,
    log: Optional[dict] = None,
    enable_mlflow: bool = False,
    experiment_name: str = "default",
) -> float:
    """
    Fitness function for the evolutionary strategy.
    """
    subset_loader = get_subset_loader(train_dataset, individual, num_workers=num_workers)
    train_model(model, subset_loader)

    subset_size = np.sum(individual)
    loss = validate_model(model, val_dataloader, S=subset_size, D=dataset_size)
    return loss
