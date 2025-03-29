import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class EfficientNet_b0(nn.Module):
    def __init__(self, num_classes=4):
        super(EfficientNet_b0, self).__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
