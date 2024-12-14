import numpy as np
from deap import base, creator, tools
from typing import Callable
from abc import ABC, abstractmethod
import mlflow


class BaseEvolutionStrategy(ABC):
    def __init__(
        self,
        dataset_size: int,
        fitness_function: Callable,
        max_generations: int,
        patience: int,
        initial_true_ratio: float,
        enable_mlflow: bool = True,
    ) -> None:
        """
        Base class for evolutionary strategies.
        params:
            dataset_size: int
                The size of the dataset
            fitness_function: callable
                The fitness function to be used
            max_generations: int
                The maximum number of generations to run
            patience: int
                The number of generations without improvement to wait before stopping
            initial_true_ratio: float
                The initial ratio of True values in the initial solution
        """
        self.dataset_size = dataset_size
        self.fitness_function = fitness_function
        self.max_generations = max_generations
        self.patience = patience
        self.initial_true_ratio = initial_true_ratio
        self.toolbox = base.Toolbox()
        self.best_solution = None
        self.best_fitness = None
        self.generation = 0
        self.generations_without_improvement = 0
        self.enable_mlflow = enable_mlflow

        self._setup_deap()

    def _setup_deap(self) -> None:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

        self.toolbox.register("attr_bool", self._initialize_with_ratio, self.initial_true_ratio)
        self.toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            self.toolbox.attr_bool,
            n=self.dataset_size,
        )
        self.toolbox.register("evaluate", self.fitness_function)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / self.dataset_size)

    def _initialize_with_ratio(self, true_ratio: float) -> bool:
        return np.random.rand() < true_ratio

    def _log_progress(self) -> None:
        if self.enable_mlflow:
            mlflow.log_metric("fitness", self.best_fitness.item(), step=self.generation)

    def _should_stop(self) -> bool:
        return (
            self.generation >= self.max_generations
            or self.generations_without_improvement >= self.patience
        )

    @abstractmethod
    def run(self):
        raise NotImplementedError("Subclasses must implement the run method")
