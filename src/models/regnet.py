import torch
import torch.nn as nn
from torchvision import models

class RegNetY800MF(nn.Module):
    def __init__(self, num_classes=4, freeze=False, include_top=True):
        super(RegNetY800MF, self).__init__()
        self.model = models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.DEFAULT)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        if not include_top:
            self.model.fc = nn.Identity()
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc.parameters():
                param.requires_grad = True
    def forward(self, x):
        return self.model(x)