from typing import Callable

import numpy as np
from deap import tools

from optimal_training_subset.evolutionary.mu_plus_lambda import MuPlusLambdaStrategy


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
            population = tools.selRoulette(offspring, self.mu)
            self._find_best(population)
            self._log_progress()
            self.generation += 1

        return self.best_solution, self.best_fitness
