import yaml 
import torch 
import torch.optim as optim

def load_config(path):
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

from models.mini_alexnet import MiniAlexNet
from models.alexnet import AlexNet

def get_model(name, num_classes):
    if name == "mini_alexnet":
        model = MiniAlexNet(num_classes=num_classes)
    elif name == "alexnet":
        model = AlexNet(num_classes=num_classes)
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
    