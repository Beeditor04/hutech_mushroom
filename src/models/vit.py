import torch
import torch.nn as nn
import timm

class TinyViT(nn.Module):
    def __init__(self, num_classes=4):
        super(TinyViT, self).__init__()
        self.model = timm.create_model("vit_tiny_patch16_224", pretrained=True)
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)