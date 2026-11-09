import torch

def main():
    print(torch.__version__)
    print(torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(torch.cuda.get_device_name(0))

if __name__ == "__main__":
    main()
