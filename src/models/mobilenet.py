import torch
import torch.nn as nn
from torchvision import models

class MobileNetV3(nn.Module):
    def __init__(self, num_classes=4, freeze=False, pretrained=True, include_top=True):
        super(MobileNetV3, self).__init__()
        if pretrained:
            self.model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        else:
            self.model = models.mobilenet_v3_large(weights=None)
        in_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(in_features, num_classes)
        if not include_top:
            self.model.classifier[3] = nn.Identity()
        if freeze and pretrained:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.classifier.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)