import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class AlexNet(nn.Module):
    def __init__(self, num_classes=2, freeze=False, include_top=True, pretrained=True):
        super(AlexNet, self).__init__()
        if pretrained:
            self.model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        else:
            self.model = models.alexnet(weights=None)
        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features, num_classes)
        if not include_top:
            self.model.classifier[6] = nn.Identity()
        if freeze and pretrained:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.classifier.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)
