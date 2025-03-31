import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class EfficientNet(nn.Module):
    def __init__(self, num_classes=4, freeze=False, include_top=True):
        super(EfficientNet, self).__init__()
        self.model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
        if not include_top:
            self.model.classifier[1] = nn.Identity()
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.classifier.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)
