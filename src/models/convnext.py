import torch
import torch.nn as nn
from torchvision import models

class ConvNeXt(nn.Module):
    def __init__(self, num_classes=4):
        super(ConvNeXt, self).__init__()
        self.model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        in_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)