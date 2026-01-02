import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from config import *

# ---------------------------
# Dataset (same as training)
# ---------------------------

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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------------------------
    # Load records again
    # ---------------------------

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

    from sklearn.model_selection import train_test_split
    _, val_records = train_test_split(
        records, test_size=0.2, random_state=42
    )

    # ---------------------------
    # Transforms & Loader
    # ---------------------------

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_loader = DataLoader(
        PedestrianDataset(val_records, transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # ---------------------------
    # Load model
    # ---------------------------

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(SELECTED_ATTRS))

    checkpoint = torch.load("peta_resnet18_attributes.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    # ---------------------------
    # Inference
    # ---------------------------

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).int()

            y_true.append(labels.cpu().numpy())
            y_pred.append(preds.cpu().numpy())

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    # ---------------------------
    # Classification report
    # ---------------------------

    print("\n=== Classification Report ===\n")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=SELECTED_ATTRS,
            zero_division=0
        )
    )

    # ---------------------------
    # Confusion matrices
    # ---------------------------

    for i, attr in enumerate(SELECTED_ATTRS):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])

        plt.figure(figsize=(3, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(attr)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()


if __name__ == "__main__":
    main()
