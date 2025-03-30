import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class EfficientNet(nn.Module):
    def __init__(self, num_classes=4):
        super(EfficientNet, self).__init__()
        self.model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)


    def forward(self, x):
        return self.model(x)
