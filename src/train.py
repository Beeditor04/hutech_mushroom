import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import time
from tqdm import tqdm

import os
import wandb

# src function

## helper
from loader.loader import get_data_loader
from utils.metrics import compute_metrics
from utils.helper import load_config, get_model, get_optimizer, get_scheduler, EarlyStopping, plot_one_batch
from parsers.parser_train import parse_args

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    dataset_size = len(loader.dataset)
    running_loss = 0.0
    running_corrects = 0
    for images, labels in tqdm(loader):
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
        for images, labels in tqdm(loader):
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
            inputs = inputs.to("cpu")
            labels = labels.to("cpu")

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy()) 
        return all_preds, all_labels


def trainer(config=None):
    args = parse_args()
    print("args", args)
    if args.config is not None:  # no sweep config, load from file
        config_path = args.config 
        config = load_config(config_path)
        print("HERE config!", config)


    PROJECT = "hutech_mushroom"
    run = None
    #setup wandb
    if config is None:
        run = wandb.init(project=PROJECT)
    else: 
        run = wandb.init(project=PROJECT, config=config)
        run.config.update(config)

    print("here config!!!", config)
    config = run.config
    print("HERE config!", config)

    DATASET = config['dataset']

    ## versioning datasets
    artifact_data = run.use_artifact(f"beehappy2554-bosch-global/{PROJECT}/{DATASET}", type='dataset')
    artifact_data_dir = artifact_data.download()
    print(config)
    # build dataset
    train_loader = get_data_loader(artifact_data_dir, config, mode="train")
    val_loader = get_data_loader(artifact_data_dir, config, mode="val")
    class_names = ["nấm mỡ", "nấm bào ngư", "nấm đùi gà", "nấm linh chi trắng"] 
    plot_one_batch(train_loader, config['batch_size'], class_names)
    # build model
    model = get_model(
        name=config['model'], 
        num_classes=len(train_loader.dataset.classes),
        freeze=config['freeze'],
        pretrained=config['pretrained']
        )
    model.to(device)

    # build optimizer
    optimizer = get_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()
    scheduler = get_scheduler(optimizer, config)
    early_stopping = EarlyStopping(patience=config['es_patience'])
    # setup log
    EPOCHS = config['num_epochs']
    TOTAL_PARAMS = sum(p.numel() for p in model.parameters())
    TOTAL_TRAINABLE_PARAMS = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    timestamp = time.strftime("%Y-%m-%d-%H:%M:%S")
    print("Training model...")
    print(f"Device: {device}")
    print(f"Number of epochs: {EPOCHS}")
    print(f"Optimizer: {optimizer}")
    print(f"Criterion: {criterion}")
    print(f"Scheduler: {scheduler}")
    print(f"Total parameters: {TOTAL_PARAMS}")
    print(f"Trainable parameters: {TOTAL_TRAINABLE_PARAMS}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Early stopping: {config['es_patience']}")
    print(f"Dataset: {len(train_loader.dataset)} training samples, {len(val_loader.dataset)} validation samples")

    # Save model
    model_dir = "../models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"best-{config['model']}-{timestamp}.pt")
    print(f"Model will be saved at {model_path}")

    # train
    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)

    # eval
        val_loss, val_acc = validate(model, val_loader, criterion)
        if scheduler is not None:
            scheduler.step(val_loss)

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Saving model at epoch {epoch+1} with val loss {val_loss:.4f}")
            torch.save(model.state_dict(), model_path)
        print(f"[Epoch {epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} Train Acc: {train_acc:.4f} Val Acc: {val_acc:.4f}")
        run.log({"train_loss": train_loss, "val_loss": val_loss, "train_acc": train_acc, "val_acc": val_acc})
        if early_stopping(val_loss):
            print("Early stopping triggered")
            break

    # final verdict: test
    print("Testing model...")
    test_loader = get_data_loader(artifact_data_dir, config, mode="test")
    
    model = get_model(
        config['model'], 
        num_classes=len(test_loader.dataset.classes),
        freeze=config['freeze'],
        pretrained=config['pretrained']
        )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    model.to("cpu")
    
    start_time = time.time()
    preds, labels = test(model, test_loader)
    end_time = time.time()

    inference_time = end_time - start_time
    class_names = test_loader.dataset.classes   
    accuracy = compute_metrics(preds, labels, class_names)
    
    print(f"Test time: {inference_time:.2f} seconds")
    print(f"Test Accuracy: {accuracy:.4f}")

    table = wandb.Table(columns=["inference_time", "test_accuracy"])
    table.add_data(inference_time, accuracy)
    run.log({"test_acc vs time": wandb.plot.scatter(
        table, 
        "inference_time", 
        "test_accuracy", 
        title="Inference Time vs Test Accuracy"
    )})
    run.log({"test_accuracy": accuracy})

    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

    # Save model as artifact
    artifact_model = wandb.Artifact(f"{config['model']}-model", type="model")
    artifact_model.add_file(model_path)
    run.log_artifact(artifact_model)
    
    run.finish()

if __name__ == "__main__":
    trainer()

