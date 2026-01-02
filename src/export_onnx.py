import torch
import torch.nn as nn
from torchvision import models

from config import *

def main():
    device = torch.device("cpu")  # export on CPU (best practice)

    # Recreate model architecture
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(SELECTED_ATTRS))

    # Load trained weights (SAFE MODE)
    checkpoint = torch.load(
        "peta_resnet18_attributes.pth",
        map_location=device,
        weights_only=True
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Dummy input (batch=1 for deployment)
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        "peta_attributes.onnx",
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"}
        },
        opset_version=17
    )

    print("ONNX model exported: peta_attributes.onnx")

if __name__ == "__main__":
    main()
