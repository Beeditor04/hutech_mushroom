import torch
import torch.nn as nn
from torchvision import models

class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super(ShuffleNetV2, self).__init__()
        self.model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)