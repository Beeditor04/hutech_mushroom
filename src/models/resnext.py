import torch
import torch.nn as nn
from torchvision import models

class ResNeXt50(nn.Module):
    def __init__(self, num_classes=4, freeze=False, include_top=True, pretrained=True):
        super(ResNeXt50, self).__init__()
        if pretrained:
            self.model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
        else:
            self.model = models.resnext50_32x4d(weights=None)
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