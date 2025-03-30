import torch
import torch.nn as nn
from torchvision import models

class ConvNeXt(nn.Module):
    def __init__(self, num_classes=4, freeze = False):
        super(ConvNeXt, self).__init__()
        self.model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        in_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(in_features, num_classes)
        if freeze:
            # only fine tune the last layer
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.classifier.parameters():
                param.requires_grad = True
    def forward(self, x):
        return self.model(x)