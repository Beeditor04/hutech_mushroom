import torch
import torch.nn as nn
from torchvision import models

class DenseNet(nn.Module):
    def __init__(self, num_classes=4):
        super(DenseNet, self).__init__()
        self.model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)