import yaml 
import torch 
import torch.optim as optim

def load_config(path):
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

from models.mini_alexnet import MiniAlexNet
from models.alexnet import AlexNet
from models.efficientnet import EfficientNet_b0
from models.convnext import ConvNeXt
from models.densenet import DenseNet
from models.mobilenet import MobileNetV3
from models.resnet import ResNet18
from models.vit import TinyViT
from models.inception import InceptionV4
from models.negnet import NegNetY800MF
from models.resnext import ResNeXt50
from models.shufflenet import ShuffleNetV2

# List of supported model names:
# ["alexnet", "convnext", "densenet", "efficientnet", "mobilenet", "resnet", "vit", "inception", "negnet", "resnext", "shufflenet"]

def get_model(name, num_classes):
    name = name.lower()
    if name == "mini_alexnet":
        model = MiniAlexNet(num_classes=num_classes)
    elif name == "alexnet":
        model = AlexNet(num_classes=num_classes)
    elif name == "efficientnet":
        model = EfficientNet_b0(num_classes=num_classes)
    elif name == "convnext":
        model = ConvNeXt(num_classes=num_classes)
    elif name == "densenet":
        model = DenseNet(num_classes=num_classes)
    elif name == "mobilenet":
        model = MobileNetV3(num_classes=num_classes)
    elif name == "resnet":
        model = ResNet18(num_classes=num_classes)
    elif name == "vit":
        model = TinyViT(num_classes=num_classes)
    elif name == "inception":
        model = InceptionV4(num_classes=num_classes)
    elif name == "negnet":
        model = NegNetY800MF(num_classes=num_classes)
    elif name == "resnext":
        model = ResNeXt50(num_classes=num_classes)
    elif name == "shufflenet":
        model = ShuffleNetV2(num_classes=num_classes)
    else:
        raise ValueError(f"Model {name} not supported.")
    return model

def get_optimizer(model, config):
    LR = float(config['lr'])
    WEIGHT_DECAY = config['weight_decay']
    MOMENTUM = config['momentum']
    OPTIMIZER = config['optimizer']

    if OPTIMIZER == "adam":
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
    elif OPTIMIZER == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        raise ValueError(f"Invalid optimizer name: {OPTIMIZER}")
    return optimizer

def get_scheduler(optimizer, config):
    SCHEDULER = config['scheduler']
    if SCHEDULER == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer)
    elif SCHEDULER == "reduce_lr_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
    return scheduler

class EarlyStopping():
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
    