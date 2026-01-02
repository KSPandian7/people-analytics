import torch
import torch.nn as nn
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from torchvision import models, transforms
from ultralytics import YOLO

from config import *
from screening import ATTRIBUTE_THRESHOLDS

# -----------------------
# Setup
# -----------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load attribute model
attr_model = models.resnet18(weights=None)
attr_model.fc = nn.Linear(attr_model.fc.in_features, len(SELECTED_ATTRS))

checkpoint = torch.load(
    "peta_resnet18_attributes.pth",
    map_location=device,
    weights_only=True
)
attr_model.load_state_dict(checkpoint["model_state"])
attr_model.to(device)
attr_model.eval()

# Load YOLO model
yolo_model = YOLO("yolov8n.pt")

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------
# Inference function
# -----------------------

def analyze_image(image: Image.Image):
    image = image.convert("RGB")

    # YOLO detection
    results = yolo_model(image)

    boxes = []
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:  # person
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                boxes.append((int(x1), int(y1), int(x2), int(y2)))

    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.imshow(image)
    ax.axis("off")

    # For each person
    for (x1, y1, x2, y2) in boxes:
        crop = image.crop((x1, y1, x2, y2))
        input_tensor = transform(crop).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = attr_model(input_tensor)
            probs = torch.sigmoid(logits).squeeze(0)

        labels = []
        for i, attr in enumerate(SELECTED_ATTRS):
            if probs[i].item() >= ATTRIBUTE_THRESHOLDS[attr]:
                labels.append(f"{attr} ({probs[i].item():.2f})")

        # Draw box
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )
        ax.add_patch(rect)

        ax.text(
            x1,
            y1 - 5,
            "\n".join(labels),
            color="yellow",
            fontsize=9,
            bbox=dict(facecolor="black", alpha=0.6)
        )

    plt.tight_layout()
    return fig


# -----------------------
# Gradio Interface
# -----------------------

demo = gr.Interface(
    fn=analyze_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Plot(),
    title="Pedestrian Attribute Analytics",
    description="Upload an image to detect people and analyze their attributes"
)

if __name__ == "__main__":
    demo.launch()
