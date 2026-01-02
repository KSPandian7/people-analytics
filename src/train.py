import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
from sklearn.model_selection import train_test_split

from config import *

# -----------------------------------
# Dataset
# -----------------------------------

class PedestrianDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img_path, labels = self.records[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        labels = torch.tensor(labels, dtype=torch.float32)
        return image, labels


# -----------------------------------
# Main (REQUIRED ON WINDOWS)
# -----------------------------------

def main():
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -----------------------------------
    # Parse PETA labels (ONCE)
    # -----------------------------------

    records = []

    with open(LABEL_FILE, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            img_name = parts[0]
            attrs = parts[1:]
            img_path = os.path.join(IMAGE_DIR, img_name)

            if not os.path.exists(img_path):
                continue

            labels = [1 if a in attrs else 0 for a in SELECTED_ATTRS]
            records.append((img_path, labels))

    print("Total samples:", len(records))

    # -----------------------------------
    # Train / Validation split
    # -----------------------------------

    train_records, val_records = train_test_split(
        records, test_size=0.2, random_state=42
    )

    print("Train samples:", len(train_records))
    print("Val samples:", len(val_records))

    # -----------------------------------
    # Transforms
    # -----------------------------------

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # -----------------------------------
    # DataLoaders (WINDOWS OPTIMIZED)
    # -----------------------------------

    train_loader = DataLoader(
        PedestrianDataset(train_records, transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        PedestrianDataset(val_records, transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))

    # -----------------------------------
    # Model
    # -----------------------------------

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(SELECTED_ATTRS))
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # -----------------------------------
    # Training loop
    # -----------------------------------

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()

        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

    # -----------------------------------
    # Save model
    # -----------------------------------

    torch.save(
        {
            "model_state": model.state_dict(),
            "attributes": SELECTED_ATTRS
        },
        "peta_resnet18_attributes.pth"
    )

    print("Model saved: peta_resnet18_attributes.pth")


# -----------------------------------
# Entry point (MANDATORY ON WINDOWS)
# -----------------------------------

if __name__ == "__main__":
    main()
