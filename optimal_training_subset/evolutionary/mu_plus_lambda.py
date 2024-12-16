from typing import Callable
from deap import tools
from optimal_training_subset.evolutionary.base_strategy import BaseEvolutionStrategy
import numpy as np


class MuPlusLambdaStrategy(BaseEvolutionStrategy):
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
        self.mu = mu
        self.lambda_ = lambda_
        super().__init__(
            dataset_size, fitness_function, max_generations, patience, initial_true_ratio
        )

    def _setup_deap(self) -> None:
        super()._setup_deap()
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual, n=self.mu
        )

    def _mutate_and_cross(self, population: list[np.ndarray]) -> list[np.ndarray]:
        offspring = []
        for _ in range(self.lambda_):
            parent1, parent2 = tools.selRandom(population, 2)
            child = self.toolbox.clone(parent1)
            self.toolbox.mate(child, parent2)
            self.toolbox.mutate(child)
            del child.fitness.values
            offspring.append(child)
        return offspring

    def _evaluate(self, population: list[np.ndarray]) -> None:
        for ind in population:
            ind.fitness.values = self.toolbox.evaluate(ind)

    def _find_best(self, population: list[np.ndarray]) -> None:
        self.best_solution = max(population, key=lambda ind: ind.fitness.values)
        self.best_fitness = self.best_solution.fitness.values

    def run(self) -> tuple[np.ndarray, float]:
        population = self.toolbox.population()
        self._evaluate(population)

        self._find_best(population)

        while not self._should_stop():
            offspring = self._mutate_and_cross(population=population)
            self._evaluate(offspring)
            population = tools.selRoulette(population + offspring, self.mu)
            self._find_best(population)
            self._log_progress()
            self.generation += 1

        return self.best_solution, self.best_fitness
