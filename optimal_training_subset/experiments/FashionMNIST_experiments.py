from optimal_training_subset.data.dataloaders import get_dataloaders
from optimal_training_subset.models.simple_cnn import SimpleCNN
from optimal_training_subset.evolutionary.one_plus_one import OnePlusOneStrategy
from optimal_training_subset.evolutionary.mu_plus_lambda import MuPlusLambdaStrategy
from optimal_training_subset.evolutionary.mu_lambda import MuLambdaStrategy
from optimal_training_subset.utils.train_utils import evaluate_algorithm, fitness_function
from optimal_training_subset.config import EXPERIMENT_REPETITIONS
from functools import partial
import torch


def main():

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    num_workers = 0
    train_dataset, train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        dataset_name="FashionMNIST", num_workers=num_workers, device=device
    )

    ff = partial(
        fitness_function,
        model_class=SimpleCNN,
        num_workers=num_workers,
        train_dataset=train_dataset,
        val_dataloader=val_dataloader,
        dataset_size=len(train_dataset),
    )

    opo = OnePlusOneStrategy(
        dataset_size=len(train_dataset),
        fitness_function=ff,
        max_generations=5000,
        patience=500,
        initial_true_ratio=0.1,
    )
    
    mpl = MuPlusLambdaStrategy(
        dataset_size=len(train_dataset),
        fitness_function=ff,
        max_generations=500,
        patience=50,
        initial_true_ratio=0.1,
        mu=100,
        lambda_=500,
    )

    ml = MuLambdaStrategy(
        dataset_size=len(train_dataset),
        fitness_function=ff,
        max_generations=500,
        patience=50,
        initial_true_ratio=0.1,
        mu=100,
        lambda_=500,
    )

    ## ADD HIKING ALG

    strategies = [opo, mpl, ml]

    for strategy in strategies:
        for _ in range(EXPERIMENT_REPETITIONS):
            evaluate_algorithm(
                algorithm=strategy,
                test_dataloader=test_dataloader,
                train_dataset=train_dataset,
                dataset_size=len(train_dataset),
                model_class=SimpleCNN,
                enable_mlflow=True,
                experiment_name="FashionMNIST",
                device=device,
            )


if __name__ == "__main__":
    main()
