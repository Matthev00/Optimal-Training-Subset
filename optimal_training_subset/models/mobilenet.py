import torchvision
from torch import nn


def get_mobilenet(num_classes: int = 100) -> nn.Module:
    model_weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
    transforms = model_weights.transforms()
    model = torchvision.models.mobilenet_v3_small(weights=model_weights)

    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    return model, transforms
