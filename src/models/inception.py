import torch
import torch.nn as nn
import timm

class InceptionV4(nn.Module):
    def __init__(self, num_classes=4):
        super(InceptionV4, self).__init__()
        self.model = timm.create_model("inception_v4", pretrained=True)
        # timm provides a helper to reset the classifier
        self.model.reset_classifier(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)