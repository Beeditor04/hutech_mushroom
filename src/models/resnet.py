import torch
import torch.nn as nn
import timm


class ResNet18(nn.Module):
    def __init__(self, num_classes=4, freeze=False):
        super(ResNet18, self).__init__()
        self.model = timm.create_model("resnetv2_18", pretrained=True)
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)


    def forward(self, x):
        return self.model(x)