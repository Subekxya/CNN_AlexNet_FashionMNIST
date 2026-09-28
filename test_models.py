import torch

from models import AlexNet5, AlexNet7


# Create a fake batch of FashionMNIST images
x = torch.randn(128, 1, 28, 28)


# Test AlexNet-5
model5 = AlexNet5()
output5 = model5(x)

print("AlexNet-5")
print("Input shape :", x.shape)
print("Output shape:", output5.shape)


# Test AlexNet-7
model7 = AlexNet7()
output7 = model7(x)

print("\nAlexNet-7")
print("Input shape :", x.shape)
print("Output shape:", output7.shape)


# Count parameters
params5 = sum(p.numel() for p in model5.parameters())
params7 = sum(p.numel() for p in model7.parameters())

print("\nParameter count:")
print("AlexNet-5:", f"{params5:,}")
print("AlexNet-7:", f"{params7:,}")