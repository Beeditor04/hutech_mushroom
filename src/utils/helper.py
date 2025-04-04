import yaml 
import torch 
import torch.optim as optim

def load_config(path):
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
import matplotlib.pyplot as plt
from models.mini_alexnet import MiniAlexNet
from models.alexnet import AlexNet
from models.efficientnet import EfficientNet
from models.convnext import ConvNeXt
from models.densenet import DenseNet
from models.mobilenet import MobileNetV3
from models.resnet import ResNet18
from models.vit import TinyViT
from models.inception import InceptionV4
from models.regnet import RegNetY800MF
from models.resnext import ResNeXt50
from models.vggnet import VGGNet
from models.shufflenet import ShuffleNetV2

# List of supported model names:
# ["alexnet", "convnext", "densenet", "efficientnet", "mobilenet", "resnet", "vit", "inception", "negnet", "resnext", "shufflenet"]

def get_model(name, num_classes, freeze=False, include_top=True, pretrained=True):
    name = name.lower()
    if name == "mini_alexnet":
        model = MiniAlexNet(num_classes=num_classes, include_top=include_top)
    elif name == "alexnet":
        model = AlexNet(num_classes=num_classes, freeze=freeze, include_top=include_top, pretrained=pretrained)
    elif name == "efficientnet":
        model = EfficientNet(num_classes=num_classes, freeze=freeze, include_top=include_top, pretrained=pretrained)
    elif name == "convnext":
        model = ConvNeXt(num_classes=num_classes, freeze=freeze, include_top=include_top, pretrained=pretrained)
    elif name == "densenet":
        model = DenseNet(num_classes=num_classes, freeze=freeze, include_top=include_top, pretrained=pretrained)
    elif name == "mobilenet":
        model = MobileNetV3(num_classes=num_classes, freeze=freeze, include_top=include_top, pretrained=pretrained)
    elif name == "resnet":
        model = ResNet18(num_classes=num_classes, freeze=freeze, include_top=include_top, pretrained=pretrained)
    elif name == "vit":
        model = TinyViT(num_classes=num_classes, freeze=freeze, include_top=include_top, pretrained=pretrained)
    elif name == "inception":
        model = InceptionV4(num_classes=num_classes, freeze=freeze, include_top=include_top, pretrained=pretrained)
    elif name == "regnet":
        model = RegNetY800MF(num_classes=num_classes, freeze=freeze, include_top=include_top, pretrained=pretrained)
    elif name == "resnext":
        model = ResNeXt50(num_classes=num_classes, freeze=freeze, include_top=include_top, pretrained=pretrained)
    elif name == "shufflenet":
        model = ShuffleNetV2(num_classes=num_classes, freeze=freeze, include_top=include_top, pretrained=pretrained)
    elif name == "vggnet":
        model = VGGNet(num_classes=num_classes, freeze=freeze, include_top=include_top, pretrained=pretrained)
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

def get_scheduler(optimizer, config, num_train_steps):
    SCHEDULER = config['scheduler']
    warmup_ratio = config.get('scheduler_warmup', 0.05)
    warmup_steps = int(num_train_steps * warmup_ratio)
    is_warmup = config.get('is_warmup', True)

    if SCHEDULER == "step":
        step_size = config.get('scheduler_step', 30)
        gamma = config.get('scheduler_gamma', 0.1)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif SCHEDULER == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_train_steps)
    else:
        raise ValueError(f"Invalid scheduler name: {SCHEDULER}")

    if is_warmup:
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0

        warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return optim.lr_scheduler.SequentialLR(
            optimizer, 
            schedulers=[warmup_scheduler, scheduler], 
            milestones=[warmup_steps]
        )
    else:
        return scheduler

def plot_one_batch(loader, batch_size=4, class_names=None):
    images, labels = next(iter(loader))
    print(f"Batch size: {batch_size}")   

    rows = (batch_size + 3) // 4  
    fig, axes = plt.subplots(rows, 4, figsize=(10, 5))
    axes = axes.flatten()  
    
    for i in range(batch_size):
        img = images[i].permute(1, 2, 0)  # convert back to image
        label = labels[i]

        axes[i].imshow(img)
        axes[i].set_title(class_names[label], fontsize=10, pad=10)  
        axes[i].axis("off")

    for j in range(batch_size, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()  # Điều chỉnh layout để tránh chồng chữ lên ảnh
    plt.savefig("sample_batch.png", dpi=300, bbox_inches='tight')

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
    