import torch
import torch.nn as nn
import timm

class NegNetY800MF(nn.Module):
    def __init__(self, num_classes=4):
        super(NegNetY800MF, self).__init__()
        self.model = timm.create_model("negnet_y_800mf", pretrained=True)
        # Use timm's common API to replace the classifier
        self.model.reset_classifier(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)