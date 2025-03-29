import torch
import torch.nn as nn
from torchvision import models

class ResNeXt50(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNeXt50, self).__init__()
        self.model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)