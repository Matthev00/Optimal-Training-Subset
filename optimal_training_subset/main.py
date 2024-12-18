from optimal_training_subset.data.dataloaders import get_dataloaders
from optimal_training_subset.models.cnn3channel import CNN3Channel
from optimal_training_subset.utils.train_utils import train_model, validate_model
import torch
import torchvision


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
num_workers = 0
train_dataset, train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
    dataset_name="CIFAR10", num_workers=num_workers, device=device
)

model = CNN3Channel().to(device)

train_model(model, train_dataloader, num_epochs=2)

loss, balanced_accuracy = validate_model(
    model, val_dataloader, S=1000, D=50000, compute_confusion=False
)
print(balanced_accuracy)

train_model(model, train_dataloader, num_epochs=2)

loss, balanced_accuracy = validate_model(
    model, val_dataloader, S=1000, D=50000, compute_confusion=False
)
print(balanced_accuracy)
train_model(model, train_dataloader, num_epochs=2)

loss, balanced_accuracy = validate_model(
    model, val_dataloader, S=1000, D=50000, compute_confusion=False
)
print(balanced_accuracy)
