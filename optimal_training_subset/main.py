from optimal_training_subset.data.dataloaders import get_dataloaders
from optimal_training_subset.models.simple_cnn import SimpleCNN
import torch
from optimal_training_subset.evolutionary.one_plus_one import OnePlusOneStrategy
from optimal_training_subset.utils.train_utils import evaluate_algorithm, fitness_function
from functools import partial
from optimal_training_subset.optimizers.hill_climbing import HillClimbingOptimizer


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
num_workers = 0
train_dataset, train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
    dataset_name="FashionMNIST", num_workers=num_workers, device=device
)

fitness_function = partial(
    fitness_function,
    model_class=SimpleCNN,
    num_workers=num_workers,
    train_dataset=train_dataset,
    val_dataloader=val_dataloader,
    dataset_size=len(train_dataset),
)

strategy = OnePlusOneStrategy(
    dataset_size=len(train_dataset),
    fitness_function=fitness_function,
    max_generations=2,
    patience=10,
    initial_true_ratio=0.05,
)

optimizer = HillClimbingOptimizer(
    initial_solution=strategy.best_solution,
    fitness_function=fitness_function,
    max_iterations=2,
    enable_mlflow=True,
)

evaluate_algorithm(
    algorithm=optimizer,
    test_dataloader=test_dataloader,
    train_dataset=train_dataset,
    dataset_size=len(train_dataset),
    model_class=SimpleCNN,
    enable_mlflow=True,
    experiment_name="Hill Climbing",
    device=device,
)
