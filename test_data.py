import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# --------------------------------------------------
# 1. Device
# --------------------------------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)


# --------------------------------------------------
# 2. Data preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# --------------------------------------------------
# 3. Load FashionMNIST
# --------------------------------------------------

train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


# --------------------------------------------------
# 4. Create DataLoaders
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# 5. Check the dataset
# --------------------------------------------------

images, labels = next(iter(train_loader))

print("\nDataset information:")
print("Training samples:", len(train_dataset))
print("Test samples:", len(test_dataset))

print("\nBatch information:")
print("Image shape:", images.shape)
print("Label shape:", labels.shape)

print("\nPixel information:")
print("Minimum:", images.min().item())
print("Maximum:", images.max().item())

print("\nFirst 10 labels:")
print(labels[:10])