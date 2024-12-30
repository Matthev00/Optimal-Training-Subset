from collections.abc import Callable
import numpy as np
import mlflow
from collections import deque


class TabuHillClimbingOptimizer:
    def __init__(
        self,
        fitness_function: Callable,
        neighbourhood_to_check: int = 10,
        max_iterations: int = 100,
        tabu_size: int = 50,
        dataset_size: int = 48000,
        percentage_true: float = 0.1,
        enable_mlflow: bool = False,
    ):
        """
        Hill Climbing Algorithm with Tabu Search for dataset optimization.

        Args:
            initial_solution (np.ndarray): Initial binary vector representing the subset selection.
            fitness_function (Callable): Function to evaluate the quality of a solution.
            max_iterations (int): Maximum number of iterations for the optimization process.
            tabu_size (int): Maximum size of the tabu list.
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
        self.tabu_list = deque(maxlen=tabu_size)

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
            mlflow.log_metric("subset_size", np.sum(self.best_solution), step=self.iteration)

    def generate_single_neighbor(self) -> np.ndarray:
        """
        Generates a single neighbor differing by one bit from the current solution.

        Returns:
            np.ndarray: Neighboring solution.
        """
        i = np.random.randint(0, len(self.current_solution))
        neighbor = self.current_solution[:].copy()
        neighbor[i] = not neighbor[i]
        return neighbor

    def generate_neighbors(self) -> list[np.ndarray]:
        """
        Generates a set of neighbors differing by one bit from the current solution.

        Returns:
            list[np.ndarray]: List of neighboring solutions.
        """
        neighbors = []
        for _ in range(self.neighbourhood_to_check):
            neighbor = self.generate_single_neighbor()
            neighbors.append(neighbor)
        return neighbors

    def run(self) -> tuple[np.ndarray, float]:
        """
        Executes the Hill Climbing optimization process with Tabu Search.

        Returns:
            tuple[np.ndarray, float]: Best solution and its fitness found during the optimization.
        """
        while self.iteration < self.max_iterations:
            self.iteration += 1
            neighbors = self.generate_neighbors()
            best_neighbor = None
            best_fitness = None

            for neighbor in neighbors:
                neighbor_tuple = tuple(neighbor)
                if neighbor_tuple in self.tabu_list:
                    continue

                fitness = self.fitness_function(neighbor)
                self.tabu_list.append(tuple(neighbor))

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

        return self.best_solution, self.best_fitness
