import torch
import torch.nn as nn
from torchvision import models

class TinyViT(nn.Module):
    def __init__(self, num_classes=4, freeze=False, include_top=True, pretrained=True):
        super(TinyViT, self).__init__()
        self.model = models.vit_h_14(weights=models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1)
        in_features = self.model.heads.in_features
        self.model.heads = nn.Linear(in_features, num_classes)
        if not include_top:
            self.model.heads = nn.Identity()
        if freeze and pretrained:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        
    def forward(self, x):
        return self.model(x)