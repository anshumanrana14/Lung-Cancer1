import torch
import torch.nn as nn
from torchvision import models


class LungCancerModel(nn.Module):
    def __init__(self, num_classes=4):
        super(LungCancerModel, self).__init__()

        # Load ResNet18 architecture
        self.model = models.resnet18(weights=None)

        # Replace final fully connected layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def load_trained_model(model_path='model.pth', device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LungCancerModel(num_classes=4)

    model.load_state_dict(torch.load(model_path, map_location=device))

    model.to(device)
    model.eval()

    return model
