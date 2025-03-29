import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class AlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        self.model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
