# imagenet-red-team

Adversarial attacks on ImageNet classifiers.

## Results

Accuracy on Imagenette val (imagenette2-320), torchvision ResNet-18 IMAGENET1K_V1 (argmax over all classes vs 10):

| Model | 1000 classes | 10 classes |
|---|---|---|
| ResNet-18 | 0.781 | 0.990 |

## Run

```
mkdir data
curl.exe -L -o data\imagenette2-320.tgz https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar -xzf data\imagenette2-320.tgz -C data
uv run main.py
```
 