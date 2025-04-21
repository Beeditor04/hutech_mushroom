import torchvision.transforms.v2 as transforms 
from torchvision import datasets, tv_tensors
from torch.utils.data import DataLoader
from PIL import ImageFilter, Image
import numpy as np
import cv2

import os 
class EdgeEnhance(object):
    def __call__(self, image):
        return image.filter(ImageFilter.EDGE_ENHANCE)

def get_dataset(dir, config, mode):
    transform_list = []
    base_transform = None
    if mode in ("test", "val"):
        transform_list = [
            transforms.Resize(config['resize']),
            transforms.ToTensor(),
        ]
        if config.get("edge_enhance", 0):
            transform_list.insert(0, transforms.RandomApply([EdgeEnhance()], p=1))
        if config.get("normalize", 0):
            transform_list.append(transforms.Normalize(mean=config['mean'], std=config['std']))
        base_transform = transforms.Compose(transform_list)
    else:
        if config.get("edge_enhance", 0):
            transform_list.append(transforms.RandomApply([EdgeEnhance()], p=1))
        transform_list.append(transforms.Resize(config['resize']))
        if config.get("random_crop", 0):
            transform_list.append(transforms.RandomCrop(config['random_crop']))
        if config.get("horizontal_flip", 0):
            transform_list.append(transforms.RandomHorizontalFlip(p=config['horizontal_flip']))
        if config.get("vertical_flip", 0):
            transform_list.append(transforms.RandomVerticalFlip(p=config['vertical_flip']))
        if config.get("zoom_out", 0):
            transform_list.append(transforms.RandomZoomOut(fill={tv_tensors.Image: (123, 117, 104)}))
        if config.get("random_rotation", 0):
            transform_list.append(transforms.RandomRotation(config['random_rotation']))  
        if any(config.get(key, 0) for key in ['brightness', 'contrast', 'saturation', 'hue']):
            transform_list.append(transforms.ColorJitter(
                brightness=config.get('brightness', 0),
                contrast=config.get('contrast', 0),
                saturation=config.get('saturation', 0),
                hue=config.get('hue', 0)
            ))
        if config.get("random_perspective", 0):
            transform_list.append(transforms.RandomPerspective(distortion_scale=config['random_perspective'], p=0.5))
        if config.get("random_affine", 0):
            transform_list.append(transforms.RandomAffine(degrees=config['random_affine']))
        if config.get("gray", 0):
            transform_list.append(transforms.RandomGrayscale(p=config['gray']))
        if config.get("blur", 0):
            transform_list.append(transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2)))
        if config.get("auto_aug", 0):
            transform_list.append(transforms.AutoAugment())
        transform_list.append(transforms.Resize(config['resize']))
        transform_list.append(transforms.ToTensor())
        if config.get("random_erasing", 0):
            transform_list.append(transforms.RandomErasing(
                p=config['random_erasing'],
                scale=(0.02, 0.1),
                ratio=(0.3, 3.3),
                value='random'
            ))
        base_transform = transforms.Compose(transform_list)

    # df_path = os.path.join(dir, mode)
    df_path = dir
    dataset = datasets.ImageFolder(df_path, transform=base_transform)
    return dataset

def get_loader(dataset, batch_size=1, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)