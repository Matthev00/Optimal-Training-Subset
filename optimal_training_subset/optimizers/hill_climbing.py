from collections.abc import Callable
import numpy as np
import mlflow


class HillClimbingOptimizer:
    def __init__(
        self,
        initial_solution: np.ndarray,
        fitness_function: Callable,
        neighborhood_to_chcek: int = 50,
        max_iterations: int = 100,
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
        self.current_solution = initial_solution[:]
        self.fitness_function = fitness_function
        self.max_iterations = max_iterations
        self.neighbourhood_to_check = neighborhood_to_chcek
        self.best_solution = best_solution
        self.best_fitness = best_fitness
        self.current_fitness = self.fitness_function(self.current_solution)
        self.enable_mlflow = enable_mlflow

    def _log_progress(self) -> None:
        if self.enable_mlflow:
            mlflow.log_metric("best fitness", self.best_fitness[0].item(), step=self.generation)
            mlflow.log_metric("subset_size", np.sum(self.best_solution), step=self.generation)

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
        for i in range(len(self.neighbourhood_to_check)):
            neighbor = self.generate_neighbors()
            neighbors.append(neighbor)
        return neighbors

    def optimize(self) -> np.ndarray:
        """
        Executes the Hill Climbing optimization process.

        Returns:
            np.ndarray: Best solution found during the optimization.
        """
        for _ in range(self.max_iterations):
            neighbors = self.generate_neighbors()
            best_neighbor = None
            best_fitness = self.current_fitness

            for neighbor in neighbors:
                fitness = self.fitness_function(neighbor)
                if fitness > best_fitness:
                    best_neighbor = neighbor
                    best_fitness = fitness

            if best_neighbor:
                self.current_solution = best_neighbor
                self.current_fitness = best_fitness
            else:
                break

        return self.current_solution
