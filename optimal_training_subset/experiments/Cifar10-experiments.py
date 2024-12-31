from functools import partial

import torch

from optimal_training_subset.config import EXPERIMENT_REPETITIONS
from optimal_training_subset.data.dataloaders import get_dataloaders
from optimal_training_subset.evolutionary.mu_plus_lambda import MuPlusLambdaStrategy
from optimal_training_subset.evolutionary.one_plus_one import OnePlusOneStrategy
from optimal_training_subset.models.cnn3channel import CNN3Channel
from optimal_training_subset.optimizers.hill_climbing import HillClimbingOptimizer
from optimal_training_subset.utils.train_utils import evaluate_algorithm, fitness_function


def main():

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    num_workers = 0
    train_dataset, _, val_dataloader, test_dataloader = get_dataloaders(
        dataset_name="CIFAR10", num_workers=num_workers, device=device
    )

    ff = partial(
        fitness_function,
        model_class=CNN3Channel,
        num_workers=num_workers,
        train_dataset=train_dataset,
        val_dataloader=val_dataloader,
        dataset_size=len(train_dataset),
    )

    for _ in range(EXPERIMENT_REPETITIONS):

        mpl = MuPlusLambdaStrategy(
            dataset_size=len(train_dataset),
            fitness_function=ff,
            max_generations=500,
            patience=50,
            initial_true_ratio=0.1,
            mu=100,
            lambda_=200,
        )

        hill_climbing = HillClimbingOptimizer(
            fitness_function=ff,
            neighbourhood_to_check=200,
            max_iterations=500,
            dataset_size=len(train_dataset),
            enable_mlflow=True,
            percentage_true=0.1,
        )

        opo = OnePlusOneStrategy(
            dataset_size=len(train_dataset),
            fitness_function=ff,
            max_generations=5000,
            patience=500,
            initial_true_ratio=0.1,
        )

        strategies = [hill_climbing, mpl, opo]
        for strategy in strategies:
            evaluate_algorithm(
                algorithm=strategy,
                test_dataloader=test_dataloader,
                train_dataset=train_dataset,
                dataset_size=len(train_dataset),
                model_class=CNN3Channel,
                enable_mlflow=True,
                experiment_name="CIFAR-10",
                device=device,
            )


if __name__ == "__main__":
    main()
