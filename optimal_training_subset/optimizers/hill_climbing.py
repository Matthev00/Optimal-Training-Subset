from collections.abc import Callable
import numpy as np
import mlflow


class HillClimbingOptimizer:
    def __init__(
        self,
        initial_solution: np.ndarray,
        fitness_function: Callable,
        neighborhood_to_chcek: int = 10,
        max_iterations: int = 100,
        dataset_size: int = 48000,
        best_solution: np.ndarray = None,
        best_fitness: float = None,
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
        self.current_solution = initial_solution[:] if initial_solution else self.initialize_random_solution()
        self.fitness_function = fitness_function
        self.max_iterations = max_iterations
        self.neighbourhood_to_check = neighborhood_to_chcek
        self.best_solution = self.current_solution if best_solution is None else best_solution
        self.best_fitness = self.fitness_function(self.best_solution) if best_fitness is None else best_fitness
        self.current_fitness = self.fitness_function(self.current_solution)
        self.enable_mlflow = enable_mlflow
        self.iteration = 0

    def initialize_random_solution(self) -> np.ndarray:
        """
        Initializes a random binary vector of the same length as the dataset.

        Returns:
            np.ndarray: Random binary vector.
        """
        return np.random.choice([True, False], size=self.dataset_size)

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
        neighbor = self.current_solution[:]
        neighbor[i] = not neighbor[i]
        return neighbor

    def generate_neighbors(self) -> list[np.ndarray]:
        """
        Generates all neighbors differing by one bit from the current solution.

        Returns:
            list[np.ndarray]: list of neighboring solutions.
        """
        neighbors = []
        for _ in range(self.neighbourhood_to_check):
            neighbor = self.generate_single_neighbor()
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
            best_fitness = self.current_fitness

            for neighbor in neighbors:
                fitness = self.fitness_function(neighbor)
                if fitness > best_fitness:
                    best_neighbor = neighbor
                    best_fitness = fitness

            if best_neighbor is not None:
                self.current_solution = best_neighbor
                self.current_fitness = best_fitness
            else:
                break
            self._log_progress()

        return self.current_solution, self.best_fitness
