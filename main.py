from pathlib import Path

import torch
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights, resnet18

VAL_DIR = Path("data/imagenette2-320/val")
IMAGENETTE_INDICES = [0, 217, 482, 491, 497, 566, 571, 574, 701]

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

    dataset = ImageFolder(VAL_DIR, transform=preprocess)
    print("images", len(dataset))
    for wind, index in zip(dataset.classes, IMAGENETTE_INDICES):
        print(wind, index, categories[index])


if __name__ == "__main__":
    main()
