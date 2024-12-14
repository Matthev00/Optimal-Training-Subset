from typing import Callable
from deap import tools
from optimal_training_subset.evolutionary.mu_plus_lambda import MuPlusLambdaStrategy
import numpy as np


class MuLambdaStrategy(MuPlusLambdaStrategy):
    def __init__(
        self,
        dataset_size: int,
        fitness_function: Callable,
        max_generations: int,
        patience: int,
        initial_true_ratio: float,
        mu: int,
        lambda_: int,
    ) -> None:
        super().__init__(
            dataset_size,
            fitness_function,
            max_generations,
            patience,
            initial_true_ratio,
            mu,
            lambda_,
        )

    def run(self) -> tuple[np.ndarray, float]:
        population = self.toolbox.population()
        self._evaluate(population)

        self._find_best(population)

        while not self._should_stop():
            offspring = self._mutate_and_cross(population=population)
            self._evaluate(offspring)
            population = tools.selBest(offspring, self.mu)
            self._find_best(population)
            self._log_progress()
            self.generation += 1

        return self.best_solution, self.best_fitness


if __name__ == "__main__":

    def fitness_function(individual):
        return (sum(individual),)

    dataset_size = 100
    max_generations = 50
    patience = 10
    initial_true_ratio = 0.05
    mu = 10
    lambda_ = 20

    strategy = MuLambdaStrategy(
        dataset_size, fitness_function, max_generations, patience, initial_true_ratio, mu, lambda_
    )
    best_solution, best_fitness = strategy.run()

    print("Best Solution:", best_solution)
    print("Best Fitness:", best_fitness)
