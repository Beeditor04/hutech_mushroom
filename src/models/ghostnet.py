import torch
import torch.nn as nn
import timm

class GhostNet(nn.Module):
    def __init__(self, num_classes=4):
        super(GhostNet, self).__init__()
        self.model = timm.create_model("ghostnetv2_100", pretrained=True)
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)