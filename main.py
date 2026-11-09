from pathlib import Path

import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

VAL_DIR = Path("data/imagenette2-320/val")

def main():
    print(torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(torch.cuda.get_device_name(0))
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights).to(device).eval()
    preprocess = weights.transforms()
    categories = weights.meta["categories"]
    print(preprocess)

    # One image
    image_path = sorted((VAL_DIR / "n01440764").iterdir())[0]
    print("image", image_path.name)
    image = Image.open(image_path).convert("RGB")
    x = preprocess(image).unsqueeze(0).to(device)
    print("x", x.shape)
    with torch.no_grad():
        logits = model(x)
    probs = logits.softmax(dim=1)
    top = probs.topk(5)
    for prob, index in zip(top.values[0], top.indices[0]):
        name = categories[index.item()]
        print(name, prob.item())

if __name__ == "__main__":
    main()
