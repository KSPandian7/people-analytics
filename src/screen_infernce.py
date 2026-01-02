import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from config import *
from screening import ATTRIBUTE_THRESHOLDS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(SELECTED_ATTRS))

    checkpoint = torch.load(
        "peta_resnet18_attributes.pth",
        map_location=device,
        weights_only=True
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def screen_output(probs):
    results = {}
    for i, attr in enumerate(SELECTED_ATTRS):
        threshold = ATTRIBUTE_THRESHOLDS[attr]
        confidence = probs[i].item()
        results[attr] = {
            "present": confidence >= threshold,
            "confidence": round(confidence, 3)
        }
    return results


def main():
    model = load_model()

    # Pick any image from dataset
    test_image = r"E:\people-analytics\data\PETA\PETA dataset\CUHK\archive\0005.png"

    input_tensor = preprocess(test_image).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).squeeze(0)

    screened = screen_output(probs)

    print("\n=== SCREENED OUTPUT ===\n")
    for attr, info in screened.items():
        if info["present"]:
            print(f"{attr:25s}  ✔  ({info['confidence']})")


if __name__ == "__main__":
    main()
