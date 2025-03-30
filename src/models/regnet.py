import torch
import torch.nn as nn
from torchvision import models

class RegNetY800MF(nn.Module):
    def __init__(self, num_classes=4):
        super(RegNetY800MF, self).__init__()
        self.model = models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.DEFAULT)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)