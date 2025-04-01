import torch
import torch.nn as nn
import timm


class ResNet18(nn.Module):
    def __init__(self, num_classes=4, freeze=False, include_top=True, pretrained=True):
        super(ResNet18, self).__init__()
        self.model = timm.create_model("resnetv2_18", pretrained=pretrained)
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)
        if not include_top:
            self.model.head = nn.Identity()
        if freeze and pretrained:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.head.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)