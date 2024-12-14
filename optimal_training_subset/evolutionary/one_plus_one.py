import numpy as np
from typing import Callable
from optimal_training_subset.evolutionary.base_strategy import BaseEvolutionStrategy
from optimal_training_subset.utils.train_utils import fitness_function
from optimal_training_subset.models.simple_cnn import SimpleCNN
from functools import partial
import torch
from optimal_training_subset.data.dataloaders import get_dataloaders


class OnePlusOneStrategy(BaseEvolutionStrategy):
    def __init__(
        self,
        dataset_size: int,
        fitness_function: Callable,
        max_generations: int,
        patience: int,
        initial_true_ratio: float,
    ) -> None:
        super().__init__(
            dataset_size, fitness_function, max_generations, patience, initial_true_ratio
        )

    def _mutate_and_evaluate(self, individual: np.ndarray) -> tuple[np.ndarray, float]:
        offspring = self.toolbox.clone(individual)
        self.toolbox.mutate(offspring)
        del offspring.fitness.values
        offspring_fitness = self.toolbox.evaluate(offspring)
        return offspring, offspring_fitness

    def _select(
        self, individual: np.ndarray, offspring: np.ndarray, offspring_fitness: float
    ) -> None:
        if offspring_fitness >= self.best_fitness:
            individual[:] = offspring
            self.best_fitness = offspring_fitness
            self.best_solution = offspring.copy()
            self.generations_without_improvement = 0
        else:
            self.generations_without_improvement += 1

    def run(self) -> tuple[np.ndarray, float]:
        individual = self.toolbox.individual()
        self.best_fitness = self.toolbox.evaluate(individual)
        self.best_solution = individual.copy()

        while not self._should_stop():
            offspring, offspring_fitness = self._mutate_and_evaluate(individual)
            self._select(individual, offspring, offspring_fitness)
            self._log_progress()
            self.generation += 1

        return self.best_solution, self.best_fitness


if __name__ == "__main__":

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    num_workers = 0
    model = SimpleCNN()
    dataset_name = "FashionMNIST"
    train_dataset, train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        dataset_name=dataset_name, num_workers=num_workers, device=device
    )
    fitness_function = partial(
        fitness_function,
        model=model,
        num_workers=num_workers,
        train_dataset=train_dataset,
        val_dataloader=val_dataloader,
        D=len(train_dataset),
    )

    dataset_size = 48000
    max_generations = 2
    patience = 10
    initial_true_ratio = 0.05

    strategy = OnePlusOneStrategy(
        dataset_size, fitness_function, max_generations, patience, initial_true_ratio
    )
    best_solution, best_fitness = strategy.run()

    print("Best Solution:", best_solution)
    print("Best Fitness:", best_fitness)
