from optimal_training_subset.optimizers.hill_climbing import HillClimbingOptimizer
from optimal_training_subset.data.dataloaders import get_dataloaders
from optimal_training_subset.utils.training_utils import evaluate_subset
import numpy as np
from torchvision.models import mobilenet_v3_small
from torchvision import transforms


if __name__ == "__main__":
    transform=transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    train_loader, val_loader, test_loader = get_dataloaders(transform=transform)
    model = mobilenet_v3_small(pretrained=True)

    initial_solution = np.random.choice([True, False], size=len(train_loader.dataset))

    def fitness_function(solution: np.ndarray) -> float:
        selected_indices = np.where(solution)[0].tolist()
        return evaluate_subset(selected_indices, val_loader, model)

    optimizer = HillClimbingOptimizer(
        initial_solution,
        fitness_function,
        max_iterations=100)
    best_solution = optimizer.optimize()

    print("Best solution:", best_solution)