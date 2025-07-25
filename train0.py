import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

from multiprocessing import freeze_support


# === Paths ===
training_data_dir = 'brain_tumor_dataset/Training'   # Your train folder path
testing_data_dir = 'brain_tumor_dataset/Testing'     # Your test folder path
model_save_dir = 'brain_tumor_model'
os.makedirs(model_save_dir, exist_ok=True)


# === Augmentation (Mild for MRI) ===
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# === Fine-tuned ResNet18 Model ===
class ResNetFineTuned(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNetFineTuned, self).__init__()
        self.model = models.resnet18(pretrained=True)

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze deeper layers for fine-tuning
        for name, child in self.model.named_children():
            if name in ['layer3', 'layer4', 'fc']:
                for param in child.parameters():
                    param.requires_grad = True

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# === Training Function ===
def train_model():
    freeze_support()

    train_set = ImageFolder(training_data_dir, transform=transform)
    val_set = ImageFolder(testing_data_dir, transform=transform)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    print("Classes:", train_set.classes)
    print(f"Training samples: {len(train_set)}, Validation samples: {len(val_set)}")
    print(f"Sample batch shape: {next(iter(train_loader))[0].shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetFineTuned(num_classes=4).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # ↓ Lower LR for fine-tuning
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    num_epochs = 25
    train_loss_list, val_loss_list, accuracy_list = [], [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total

        scheduler.step(avg_val_loss)

        train_loss_list.append(avg_train_loss)
        val_loss_list.append(avg_val_loss)
        accuracy_list.append(accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.2f}%")

    # === Save Model ===
    model_path = os.path.join(model_save_dir, 'resnet18_brain_tumor.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at: {model_path}")

    # === Plot Metrics ===
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label='Train Loss', color='blue')
    plt.plot(val_loss_list, label='Val Loss', color='orange')
    plt.title('Loss Curve')
    plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_list, label='Val Accuracy', color='green')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy (%)')
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(model_save_dir, 'training_metrics.png'))
    plt.show()

    # === Evaluation Report ===
    true_labels, pred_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    target_names = val_set.classes
    report = classification_report(true_labels, pred_labels, target_names=target_names, output_dict=True)
    print("\nClassification Report:\n", classification_report(true_labels, pred_labels, target_names=target_names))
    print(f"Overall Accuracy: {report['accuracy'] * 100:.2f}%")

    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_dir, 'confusion_matrix.png'))
    plt.show()

    # === Per-Class Bar Chart ===
    precision = [report[label]["precision"] for label in target_names]
    recall = [report[label]["recall"] for label in target_names]
    f1_score = [report[label]["f1-score"] for label in target_names]

    x = np.arange(len(target_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, precision, width, label='Precision')
    ax.bar(x, recall, width, label='Recall')
    ax.bar(x + width, f1_score, width, label='F1 Score')
    ax.set_xticks(x)
    ax.set_xticklabels(target_names, rotation=45)
    ax.set_ylim(0, 1.05)
    ax.set_title('Per-Class Metrics')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_dir, 'per_class_metrics.png'))
    plt.show()


if __name__ == '__main__':
    train_model()