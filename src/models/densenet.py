import torch
import torch.nn as nn
from torchvision import models

class DenseNet(nn.Module):
    def __init__(self, num_classes=4, freeze=False, include_top=True, pretrained=True):
        super(DenseNet, self).__init__()
        if pretrained:
            self.model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        else:
            self.model = models.densenet121(weights=None)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)
        if not include_top:
            self.model.classifier = nn.Identity()
        if freeze and pretrained:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.classifier.parameters():
                param.requires_grad = True
    def forward(self, x):
        return self.model(x)