import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim

from tqdm import tqdm

import os
import wandb

# src function
from models.mini_alexnet import MiniAlexNet
from loader.loader import get_data_loader
from utils.metrics import compute_metrics
from utils.helper import load_config

import time
wandb.login()
wandb.require("core")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    dataset_size = len(loader.dataset)
    running_loss = 0.0
    running_corrects = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # stats
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects.double() / dataset_size
    return epoch_loss, epoch_acc

def validate(model, loader, criterion):
    model.eval()
    dataset_size = len(loader.dataset)
    val_running_loss = 0.0
    val_running_corrects = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_running_corrects += torch.sum(preds == labels.data)

    val_loss = val_running_loss / dataset_size
    val_acc = val_running_corrects.double() / dataset_size
    return val_loss, val_acc

def test(model, loader):
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy()) 
        return all_preds, all_labels

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

def trainer(config=None):
    #setup wandb
    if config is None:  # no sweep config, load from file
        config_path = "../config/exp.yaml"
        config = load_config(config_path)
        
    run = wandb.init(project="minialexnet", config=config)
    config = wandb.config

    ## versioning dataset
    artifact_data = run.use_artifact("beehappy2554-bosch-global/minialexnet/minialexnet-dataset:latest", type='dataset')
    artifact_data_dir = artifact_data.download()
    print(config)
    # build dataset
    train_loader = get_data_loader(artifact_data_dir, config['batch_size'], type="train")
    val_loader = get_data_loader(artifact_data_dir, config['batch_size'], type="val")

    # build model
    model = MiniAlexNet()
    model.to(device)
    optimizer = get_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()

    # setup log
    EPOCHS = config['num_epochs']

    timestamp = time.strftime("%Y-%m-%d-%H:%M:%S")
    print("Training model...")
    print(f"Device: {device}")
    print(f"Number of epochs: {EPOCHS}")
    print(f"Optimizer: {optimizer}")
    print(f"Criterion: {criterion}")
    print(f"Dataset: {len(train_loader.dataset)} training samples, {len(val_loader.dataset)} validation samples")

    # train
    for epoch in tqdm(range(EPOCHS)):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)

    # eval
        val_loss, val_acc = validate(model, val_loader, criterion)
        print(f"[Epoch {epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} Train Acc: {train_acc:.4f} Val Acc: {val_acc:.4f}")
        run.log({"train_loss": train_loss, "val_loss": val_loss, "train_acc": train_acc, "val_acc": val_acc})

    # final verdict: test
    test_loader = get_data_loader(artifact_data_dir, config['batch_size'], type="test")
    preds, labels = test(model, test_loader)
    class_names = test_loader.dataset.classes   
    accuracy = compute_metrics(preds, labels, class_names)
    run.log({"test_accuracy": accuracy,
             "confusion_matrix": wandb.plot.confusion_matrix(y_true=labels, preds=preds, class_names=class_names)})
   
    # Save model
    model_dir = "../models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"minialexnet-{timestamp}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

    # Save model as artifact
    artifact_model = wandb.Artifact("minialexnet-model", type="model")
    artifact_model.add_file(model_path)
    run.log_artifact(artifact_model)

    run.finish()

if __name__ == "__main__":
    trainer()

