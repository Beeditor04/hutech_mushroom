import torch
import torch.nn as nn
import timm

class GhostNet(nn.Module):
    def __init__(self, num_classes=4, freeze=False, include_top=True, pretrained=True):
        super(GhostNet, self).__init__()
        if pretrained:
            self.model = timm.create_model("ghostnetv2_100", pretrained=True)
        else:
            self.model = timm.create_model("ghostnetv2_100", pretrained=False)
        in_features = self.model.head.in_features
        self.model.reset_classifier(num_classes=num_classes)
        if not include_top:
            self.model.head = nn.Identity()
        if freeze and pretrained:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.head.parameters():
                param.requires_grad = True
    def forward(self, x):
        return self.model(x)