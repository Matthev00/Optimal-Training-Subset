import numpy as np
from typing import Callable
from optimal_training_subset.evolutionary.base_strategy import BaseEvolutionStrategy


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
        offspring_fitness = self.toolbox.evaluate(offspring)
        return offspring, offspring_fitness

    def _select(self, individual: np.ndarray, offspring: np.ndarray, offspring_fitness: float):
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

    def fitness_function(individual):
        return (sum(individual),)

    dataset_size = 100
    max_generations = 600
    patience = 10
    initial_true_ratio = 0.05

    strategy = OnePlusOneStrategy(
        dataset_size, fitness_function, max_generations, patience, initial_true_ratio
    )
    best_solution, best_fitness = strategy.run()

    print("Best Solution:", best_solution)
    print("Best Fitness:", best_fitness)
