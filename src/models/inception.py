import torch
import torch.nn as nn
import timm

class InceptionV4(nn.Module):
    def __init__(self, num_classes=4, freeze=False, include_top=True, pretrained=True):
        super(InceptionV4, self).__init__()
        self.model = timm.create_model("inception_v4", pretrained=pretrained)
        # timm provides a helper to reset the classifier
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