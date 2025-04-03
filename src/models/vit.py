import torch
import torch.nn as nn
import timm

class TinyViT(nn.Module):
    def __init__(self, num_classes=4, freeze=False, include_top=True, pretrained=True):
        super(TinyViT, self).__init__()
        self.model = timm.create_model("vit_base_patch32_224", pretrained=pretrained)
        in_features = self.model.head.in_features
        self.model.reset_classifier(num_classes=num_classes)
        if not include_top:
            self.model.reset_classifier(num_classes=0)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.head.parameters():
                param.requires_grad = True
    def forward(self, x):
        return self.model(x)