from collections.abc import Callable


class HillClimbingOptimizer:
    def __init__(
        self, initial_solution: list[int], fitness_function: Callable, max_iterations: int = 100
    ):
        """
        Hill Climbing Algorithm for dataset optimization.

        Args:
            initial_solution (list[int]): Initial binary vector representing the subset selection.
            fitness_function (Callable): Function to evaluate the quality of a solution.
            max_iterations (int): Maximum number of iterations for the optimization process.
        """
        self.current_solution = initial_solution[:]
        self.fitness_function = fitness_function
        self.max_iterations = max_iterations
        self.current_fitness = self.fitness_function(self.current_solution)

    def generate_neighbors(self) -> list[list[int]]:
        """
        Generates all neighbors differing by one bit from the current solution.

        Returns:
            list[list[int]]: list of neighboring solutions.
        """
        neighbors = []
        for i in range(len(self.current_solution)):
            neighbor = self.current_solution[:]
            neighbor[i] = 1 - neighbor[i]
            neighbors.append(neighbor)
        return neighbors

    def optimize(self) -> list[int]:
        """
        Executes the Hill Climbing optimization process.

        Returns:
            list[int]: Best solution found during the optimization.
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
