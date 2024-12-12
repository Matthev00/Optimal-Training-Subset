from optimal_training_subset.data.dataloaders import get_dataloaders
from optimal_training_subset.models.simple_cnn import SimpleCNN
from optimal_training_subset.models.mobilenet import get_mobilenet
import torch


model = SimpleCNN().to("cuda")
train_dataset, train_data_loader, val_data_loader, test_data_loader = get_dataloaders("FashionMNIST", batch_size=12)


X, y = next(iter(train_data_loader))
print(X.shape, y.shape)

y_pred = model(X)
loss = torch.nn.CrossEntropyLoss()(y_pred, y)
print(loss.item())