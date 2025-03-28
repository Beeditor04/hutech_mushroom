import torchvision.transforms as transforms 
from torchvision import datasets
from torch.utils.data import DataLoader

import os 

def get_data_loader(dir, batch_size, type):
    base_transform = transforms.Compose([
        transforms.Resize(227),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
    ])
    df_path = os.path.join(dir, type)
    df = datasets.ImageFolder(df_path, transform=base_transform)  

    loader = DataLoader(df, batch_size=batch_size, shuffle=True)

    return loader
