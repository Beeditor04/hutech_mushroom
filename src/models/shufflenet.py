import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=2, freeze=False, include_top=True, pretrained=True):
        super(ShuffleNetV2, self).__init__()
        if pretrained:
            self.model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
        else:
            self.model = models.shufflenet_v2_x1_0(weights=None)
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