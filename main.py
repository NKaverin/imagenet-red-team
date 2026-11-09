from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights, resnet18

VAL_DIR = Path("data/imagenette2-320/val")
IMAGENETTE_INDICES = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

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

    # Dataset
    dataset = ImageFolder(VAL_DIR, transform=preprocess)
    print("images", len(dataset))
    to_imagenet = torch.tensor(IMAGENETTE_INDICES, device=device)
    loader = DataLoader(dataset, batch_size=64, num_workers=4)
    correct_1000 = 0
    correct_10 = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            # Predict as on full ImageNet
            preds_1000 = logits.argmax(dim=1)
            correct_1000 += (preds_1000 == to_imagenet[labels]).sum().item()
            # Only Imagenette classes
            preds_10 = logits[:, to_imagenet].argmax(dim=1)
            correct_10 += (preds_10 == labels).sum().item()
    print("top for 1000 classes", correct_1000 / len(dataset))
    print("top for 10 classes", correct_10 / len(dataset))

if __name__ == "__main__":
    main()
