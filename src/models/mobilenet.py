import torch
import torch.nn as nn
from torchvision import models

class MobileNetV3(nn.Module):
    def __init__(self, num_classes=4):
        super(MobileNetV3, self).__init__()
        self.model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        in_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)