import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import AlexNet5, AlexNet7


# ============================================================
# 1. Configuration
# ============================================================

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

SEED = 42

torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)


# ============================================================
# 2. FashionMNIST preprocessing
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# ============================================================
# 3. Load dataset
# ============================================================

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


# ============================================================
# 4. DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 5. Training function
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_accuracy = 100.0 * correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# 6. Evaluation function
# ============================================================

def evaluate(model, loader, criterion):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_accuracy = 100.0 * correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# 7. Run experiment
# ============================================================

def run_experiment(model, model_name):

    print("\n" + "=" * 60)
    print(f"Training {model_name}")
    print("=" * 60)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    writer = SummaryWriter(
        log_dir=f"runs/{model_name}"
    )

    for epoch in range(EPOCHS):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            criterion
        )

        # TensorBoard logging
        writer.add_scalar(
            "Loss/Train",
            train_loss,
            epoch + 1
        )

        writer.add_scalar(
            "Loss/Test",
            test_loss,
            epoch + 1
        )

        writer.add_scalar(
            "Accuracy/Train",
            train_accuracy,
            epoch + 1
        )

        writer.add_scalar(
            "Accuracy/Test",
            test_accuracy,
            epoch + 1
        )

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_accuracy:.2f}%"
        )

    writer.close()

    print(f"\nFinal test accuracy: {test_accuracy:.2f}%")

    return test_accuracy


# ============================================================
# 8. Train AlexNet-5
# ============================================================

torch.manual_seed(SEED)

model5 = AlexNet5()

accuracy5 = run_experiment(
    model5,
    "AlexNet5"
)


# ============================================================
# 9. Train AlexNet-7
# ============================================================

torch.manual_seed(SEED)

model7 = AlexNet7()

accuracy7 = run_experiment(
    model7,
    "AlexNet7"
)


# ============================================================
# 10. Final comparison
# ============================================================

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(f"AlexNet-5 Test Accuracy: {accuracy5:.2f}%")
print(f"AlexNet-7 Test Accuracy: {accuracy7:.2f}%")