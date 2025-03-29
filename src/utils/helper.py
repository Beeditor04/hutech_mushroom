import yaml 

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