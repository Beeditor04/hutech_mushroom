import os
import torch
import numpy as np
import random
from PIL import Image
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from loader.loader import get_dataset, get_loader
from utils.helper import load_config, get_model, get_optimizer, get_scheduler, EarlyStopping, get_loss
from utils.metrics import compute_metrics
from utils.setup import set_seed

# --- SEEDING ---
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# --- CONFIG ---
CONFIG_PATH = "../config/exp.yaml"  # <-- set your config path
config = load_config(CONFIG_PATH)

# --- DATASET ---
TRAIN_DATA_DIR_PATH = '../data/train'
TEST_DATA_DIR_PATH = '../final_test'

full_train_dataset = get_dataset(TRAIN_DATA_DIR_PATH, config, mode="train")
val_size = int(0.1 * len(full_train_dataset))
train_size = len(full_train_dataset) - val_size
train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size])

train_loader = get_loader(train_subset, batch_size=config["batch_size"], shuffle=True)
val_loader = get_loader(val_subset, batch_size=config["batch_size"], shuffle=False)

# --- MODEL ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(
    name=config['model'],
    num_classes=len(full_train_dataset.classes),
    freeze=config['freeze'],
    pretrained=config['pretrained']
)
model.to(device)

optimizer = get_optimizer(model, config)
criterion = get_loss(config, device)
scheduler = get_scheduler(optimizer, config, len(train_loader)*config['num_epochs'])
early_stopping = EarlyStopping(patience=config['es_patience'])

# --- TRAIN LOOP ---
EPOCHS = config['num_epochs']
best_val_loss = float('inf')
model_path = f"best-{config['model']}.pt"

for epoch in range(EPOCHS):
    model.train()
    running_loss, running_corrects = 0.0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
    train_loss = running_loss / len(train_loader.dataset)
    train_acc = running_corrects.double() / len(train_loader.dataset)

    # Validation
    model.eval()
    val_running_loss, val_running_corrects = 0.0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_running_corrects += torch.sum(preds == labels.data)
    val_loss = val_running_loss / len(val_loader.dataset)
    val_acc = val_running_corrects.double() / len(val_loader.dataset)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_path)
    if early_stopping(val_loss):
        print("Early stopping triggered")
        break
    if scheduler is not None:
        scheduler.step()

# --- LOAD BEST MODEL ---
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- INFERENCE ON TEST SET & EXPORT CSV ---
test_transform = transforms.Compose([
    transforms.Resize(config['resize']),
    transforms.ToTensor(),
    # transforms.Normalize(mean=config['mean'], std=config['std'])
])

test_images = sorted([f for f in os.listdir(TEST_DATA_DIR_PATH) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
results = []

for img_name in test_images:
    img_path = os.path.join(TEST_DATA_DIR_PATH, img_name)
    image = Image.open(img_path).convert("RGB")
    img_tensor = test_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)
        label = pred.item()
    results.append({"image_name": img_name, "label": label})

df = pd.DataFrame(results)
os.makedirs("output", exist_ok=True)
df.to_csv("output/results.csv", index=False)
print("Saved predictions to output/results.csv")