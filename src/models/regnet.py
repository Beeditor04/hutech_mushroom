import torch
import torch.nn as nn
from torchvision import models

class RegNetY800MF(nn.Module):
    def __init__(self, num_classes=4, freeze=False, include_top=True, pretrained=True):
        super(RegNetY800MF, self).__init__()
        if pretrained:
            self.model = models.regnet_y_800mf(weights=models.RegNet_Y_800MF_Weights.DEFAULT)
        else:
            self.model = models.regnet_y_800mf(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        if not include_top:
            self.model.fc = nn.Identity()
        if freeze and pretrained:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc.parameters():
                param.requires_grad = True
    def forward(self, x):
        return self.model(x)