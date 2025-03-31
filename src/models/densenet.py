import torch
import torch.nn as nn
from torchvision import models

class DenseNet(nn.Module):
    def __init__(self, num_classes=4, freeze=False, include_top=True):
        super(DenseNet, self).__init__()
        self.model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)
        if not include_top:
            self.model.classifier = nn.Identity()
    def forward(self, x):
        return self.model(x)