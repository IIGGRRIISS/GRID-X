import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import sys
from pathlib import Path
import numpy as np

# Get the directory where this script is located
script_dir = Path(__file__).parent
# Project root is three levels up: scripts/models/cnn/ -> scripts/models/ -> scripts/ -> GRID-X
project_root = script_dir.parent.parent.parent

# Add script directory to path so we can import dataset and model
sys.path.append(str(script_dir))

from dataset import CircuitDataset
from model import get_model, extract_features

# ---------- Paths ----------
IMG_DIR = project_root / 'data' / 'cnn' / 'circuit_images'
CSV_PATH = project_root / 'data' / 'cnn' / 'circuit_metadata.csv'
MODEL_SAVE_PATH = project_root / 'models' / 'circuit_classifier.pth'
CLASS_NAMES_PATH = project_root / 'models' / 'class_names.txt'
EMBEDDINGS_SAVE_PATH = project_root / 'models' / 'circuit_embeddings.npy'
LABELS_SAVE_PATH = project_root / 'models' / 'circuit_labels.npy'

# ---------- Hyperparameters ----------
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- Data Augmentation ----------
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------- Load Dataset ----------
full_dataset = CircuitDataset(IMG_DIR, CSV_PATH, transform=train_transforms)
num_classes = len(full_dataset.classes)

# Split into train/val (80/20)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Assign the correct transforms (a bit tricky with random_split, but we'll set the dataset's transform)
train_dataset.dataset.transform = train_transforms
val_dataset.dataset.transform = val_transforms

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------- Initialize Model ----------
model = get_model(num_classes, pretrained=True).to(DEVICE)

# Only train the final layer
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# ---------- Training Loop ----------
print(f"Training on {DEVICE}")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = 100 * correct / total

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

# ---------- Save Model and Class Names ----------
torch.save(model.state_dict(), MODEL_SAVE_PATH)
with open(CLASS_NAMES_PATH, 'w') as f:
    for name in full_dataset.classes:
        f.write(name + '\n')
print(f"Model saved to {MODEL_SAVE_PATH}")
print(f"Class names saved to {CLASS_NAMES_PATH}")

# ---------- Pre‑compute Embeddings for Similarity ----------
print("Extracting feature embeddings for all images...")
full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False)
features, labels = extract_features(model, full_loader, DEVICE)
np.save(EMBEDDINGS_SAVE_PATH, features)
np.save(LABELS_SAVE_PATH, labels)
print(f"Embeddings saved to {EMBEDDINGS_SAVE_PATH} ({features.shape})")