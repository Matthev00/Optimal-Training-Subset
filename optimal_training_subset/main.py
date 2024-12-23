from optimal_training_subset.data.dataloaders import get_dataloaders
from optimal_training_subset.models.simple_cnn import SimpleCNN
import torch
from optimal_training_subset.evolutionary.one_plus_one import OnePlusOneStrategy
from optimal_training_subset.utils.train_utils import evaluate_algorithm, fitness_function
from functools import partial
from optimal_training_subset.optimizers.hill_climbing import HillClimbingOptimizer
from optimal_training_subset.optimizers.tabu_hill_climbing import TabuHillClimbingOptimizer
from optimal_training_subset.models.cnn3channel import CNN3Channel
from optimal_training_subset.utils.train_utils import train_model, validate_model
import torch
import torchvision


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
num_workers = 0
train_dataset, train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
    dataset_name="CIFAR10", num_workers=num_workers, device=device
)

model = CNN3Channel().to(device)

train_model(model, train_dataloader, target_iterations=2)

fitness_function = partial(
    fitness_function,
    model_class=CNN3Channel,
    num_workers=num_workers,
    train_dataset=train_dataset,
    val_dataloader=val_dataloader,
    dataset_size=len(train_dataset),
    device=device,
)

optimizer = HillClimbingOptimizer(
    fitness_function=fitness_function,
    max_iterations=2,
    enable_mlflow=True,
    dataset_size=len(train_dataset),
    percentage_true=0.5,
)

evaluate_algorithm(
    algorithm=optimizer,
    test_dataloader=test_dataloader,
    train_dataset=train_dataset,
    dataset_size=len(train_dataset),
    model_class=CNN3Channel,
    enable_mlflow=True,
    experiment_name="Hill Climbing",
    device=device,
)
