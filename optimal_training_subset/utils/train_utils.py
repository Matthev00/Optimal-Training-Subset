from torch import optim
from tqdm import tqdm
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import numpy as np
from optimal_training_subset.data.dataloaders import get_subset_loader
import mlflow
from functools import wraps
import seaborn as sns
import matplotlib.pyplot as plt
from optimal_training_subset.utils.config import BATCHES


def create_cf_heatmap(confusion_matrix: np.ndarray) -> None:
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    heatmap_path = "reports/figures/confusion_matrix.png"
    plt.savefig(heatmap_path)
    plt.close()


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    target_iterations: int = 2,
    learning_rate: float = 1e-3,
    device: torch.device = torch.device("cuda"),
) -> None:
    """
    Trains the given neural network model on the provided dataset, balancing
    the number of epochs and iterations to ensure fair training across datasets
    of different sizes.

    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader providing the training dataset.
        target_iterations (int, optional): Target number of training iterations.
            Default is 1000.
        learning_rate (float, optional): Learning rate for the optimizer. Default is 1e-3.
        device (torch.device, optional): Device for training (default is CUDA if available).

    Returns:
        None
    """
    NUM_BATCHES = BATCHES * target_iterations
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    model.train()
    model.to(device)

    train_iter = iter(train_loader)

    batch_count = 0
    with tqdm(total=NUM_BATCHES, desc="Training Progress") as pbar:
        while batch_count < NUM_BATCHES:
            try:
                images, labels = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                images, labels = next(train_iter)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_count += 1
            pbar.update(1)


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
    device: torch.device = torch.device("cuda"),
    compute_confusion: bool = True,
) -> tuple[float, float, None | np.ndarray]:
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
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)

    all_labels = []
    all_predictions = []

    with torch.inference_mode():
        for images, labels in val_loader:
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    balanced_accuracy = balanced_accuracy_score(all_labels, all_predictions)
    loss = loss_fn(balanced_accuracy, alpha, beta, S, D)
    if not compute_confusion:
        return loss, balanced_accuracy

    confusion = confusion_matrix(all_labels, all_predictions)
    return loss, balanced_accuracy, confusion


def mlflow_logger(func):
    @wraps(func)
    def wrapper(*args, **kwargs) -> float:
        enable_mlflow = kwargs.get("enable_mlflow", False)
        alghorithm_name = kwargs.get("algorithm").__class__.__name__
        if not enable_mlflow:
            return func(*args, **kwargs)

        experiment_name = kwargs.get("experiment_name", "default")
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            mlflow.log_param("algorithm", alghorithm_name)
            loss, b_accuracy, confusion = func(*args, **kwargs)
            mlflow.log_metric("TEST LOSS", loss)
            mlflow.log_metric("BALANCED ACCURACY", b_accuracy)
            if confusion is not None:
                create_cf_heatmap(confusion)
                mlflow.log_artifact("reports/figures/confusion_matrix.png")
            return loss

    return wrapper


def fitness_function(
    individual: np.ndarray,
    model_class: nn.Module,
    num_workers: int,
    train_dataset: Dataset,
    val_dataloader: DataLoader,
    dataset_size: int,
    device: torch.device = torch.device("cuda"),
) -> tuple[float]:
    """
    Fitness function for the evolutionary strategy.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    subset_loader = get_subset_loader(train_dataset, individual, num_workers=num_workers)
    model = model_class().to(device)
    train_model(model, subset_loader)
    subset_size = np.sum(individual)
    loss, _ = validate_model(
        model, val_dataloader, S=subset_size, D=dataset_size, compute_confusion=False
    )
    return (loss,)


@mlflow_logger
def evaluate_algorithm(
    algorithm,
    test_dataloader: DataLoader,
    train_dataset: Dataset,
    dataset_size: int,
    model_class,
    enable_mlflow: bool = True,
    experiment_name: str = "default",
    device: torch.device = torch.device("cpu"),
) -> tuple[np.ndarray, float]:

    best_solution, _ = algorithm.run()
    model = model_class()
    train_dataloader = get_subset_loader(train_dataset, best_solution)
    train_model(model, train_dataloader, device=device, target_iterations=5)
    S = np.sum(best_solution)
    loss, b_accuracy, confusion = validate_model(
        model, test_dataloader, S=S, D=dataset_size, device=device, compute_confusion=True
    )
    return loss, b_accuracy, confusion
