import torchvision.transforms as transforms 
from torchvision import datasets
from torch.utils.data import DataLoader

import os 

def get_data_loader(dir, config, mode):
    transform_list = []

    transform_list.append(transforms.Resize(config['resize']))
    
    if config.get("horizontal_flip", 0):
        transform_list.append(transforms.RandomHorizontalFlip(p=config['horizontal_flip']))
    
    if config.get("vertical_flip", 0):
        transform_list.append(transforms.RandomVerticalFlip(p=config['vertical_flip']))
    
    if config.get("random_rotation", 0):
        transform_list.append(transforms.RandomRotation(config['random_rotation']))
        
    if any(config.get(key, 0) for key in ['brightness', 'contrast', 'saturation', 'hue']):
        transform_list.append(transforms.ColorJitter(
            brightness=config.get('brightness', 0),
            contrast=config.get('contrast', 0),
            saturation=config.get('saturation', 0),
            hue=config.get('hue', 0)
        ))
    
    if config.get("sigma", 0) or config.get("kernel_size", 0):
        transform_list.append(transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2)))
    
    transform_list.append(transforms.ToTensor())
    # convert from list to transforms
    base_transform = transforms.Compose(transform_list)
    if mode in ("test", "val"):
        base_transform = transforms.Compose([
            transforms.Resize(config['resize']),
            transforms.ToTensor()
        ])

    # dataset
    df_path = os.path.join(dir, mode)
    dataset = datasets.ImageFolder(df_path, transform=base_transform)  
    
    # dataloader
    BATCH_SIZE = config["batch_size"] if mode == "train" else 1
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    return loader
