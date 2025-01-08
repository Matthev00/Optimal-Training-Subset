from collections.abc import Callable
import numpy as np
import mlflow


class HillClimbingOptimizer:
    def __init__(
        self,
        fitness_function: Callable,
        neighbourhood_to_check: int = 10,
        max_iterations: int = 100,
        dataset_size: int = 48000,
        percentage_true: float = 0.1,
        enable_mlflow: bool = False,
    ):
        """
        Hill Climbing Algorithm for dataset optimization.

        Args:
            initial_solution (np.ndarray): Initial binary vector representing the subset selection.
            fitness_function (Callable): Function to evaluate the quality of a solution.
            max_iterations (int): Maximum number of iterations for the optimization process.
        """
        self.dataset_size = dataset_size
        self.percentage_true = percentage_true
        self.current_solution = self.initialize_random_solution()
        self.fitness_function = fitness_function
        self.current_fitness = self.fitness_function(self.current_solution)
        self.max_iterations = max_iterations
        self.neighbourhood_to_check = neighbourhood_to_check
        self.best_solution = self.current_solution
        self.best_fitness = self.current_fitness
        self.enable_mlflow = enable_mlflow
        self.iteration = 0

    def initialize_random_solution(self) -> np.ndarray:
        """
        Initializes a random binary vector with a specified percentage of True values.

        Returns:
            np.ndarray: Random binary vector.
        """
        num_true = int(self.dataset_size * self.percentage_true)
        num_false = self.dataset_size - num_true
        solution = np.array([True] * num_true + [False] * num_false)
        np.random.shuffle(solution)
        return solution

    def _log_progress(self) -> None:
        if self.enable_mlflow:
            mlflow.log_metric("best fitness", self.best_fitness[0].item(), step=self.iteration)
            mlflow.log_metric("best_subset_size", np.sum(self.best_solution), step=self.iteration)
            mlflow.log_metric("current fitness", self.current_fitness[0].item(), step=self.iteration)
            mlflow.log_metric("current_subset_size", np.sum(self.current_solution), step=self.iteration)

    def generate_single_neighbor(self, weights) -> np.ndarray:
        """
        Generates a single neighbor differing by one bit from the current solution.

        Returns:
            np.ndarray: Neighboring solution.
        """
        i = np.random.choice(len(self.current_solution), 1, replace=False, p=weights)[0]
        neighbor = self.current_solution.copy()
        neighbor[i] = not neighbor[i]
        return neighbor

    def calculate_weights(self) -> np.ndarray:
        """
        Calculates the weights for the random choice of the bit to flip.

        Returns:
            np.ndarray: Weights for the random choice.
        """
        weights = np.zeros_like(self.current_solution, dtype=float)
        weights[self.current_solution == 0] = 1 / np.sum(self.current_solution == 0)
        weights[self.current_solution == 1] = 1 / np.sum(self.current_solution == 1)
        weights /= weights.sum()
        return weights

    def generate_neighbors(self) -> list[np.ndarray]:
        """
        Generates all neighbors differing by one bit from the current solution.

        Returns:
            list[np.ndarray]: list of neighboring solutions.
        """
        neighbors = []
        weights = self.calculate_weights()
        for _ in range(self.neighbourhood_to_check):
            neighbor = self.generate_single_neighbor(weights=weights)
            neighbors.append(neighbor)
        return neighbors

    def run(self) -> tuple[np.ndarray, float]:
        """
        Executes the Hill Climbing optimization process.

        Returns:
            np.ndarray: Best solution found during the optimization.
        """
        while self.iteration < self.max_iterations:
            self.iteration += 1
            neighbors = self.generate_neighbors()
            best_neighbor = None
            best_fitness = None

            for neighbor in neighbors:
                fitness = self.fitness_function(neighbor)
                if best_fitness is None or fitness > best_fitness:
                    best_neighbor = neighbor
                    best_fitness = fitness

            if best_neighbor is not None:
                self.current_solution = best_neighbor
                self.current_fitness = best_fitness

                if best_fitness > self.best_fitness:
                    self.best_solution = best_neighbor
                    self.best_fitness = best_fitness
            else:
                self._log_progress()
                break
            self._log_progress()

        return self.current_solution, self.best_fitness
