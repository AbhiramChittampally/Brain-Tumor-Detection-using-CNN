import os
import shutil
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

original_train_dir = 'brain_tumor_dataset/Training'
original_test_dir = 'brain_tumor_dataset/Testing'
web_data_dir = 'brain_tumor_extra_data'
combined_dir = 'combined_dataset'
combined_train_dir = os.path.join(combined_dir, 'Training')
combined_test_dir = os.path.join(combined_dir, 'Testing')
model_save_dir = 'brain_tumor_model'
os.makedirs(model_save_dir, exist_ok=True)

def combine_datasets():
    if os.path.exists(combined_train_dir):
        shutil.rmtree(combined_train_dir)
    shutil.copytree(original_train_dir, combined_train_dir)

    for tumor_class in os.listdir(web_data_dir):
        src_folder = os.path.join(web_data_dir, tumor_class)
        dst_folder = os.path.join(combined_train_dir, tumor_class)
        os.makedirs(dst_folder, exist_ok=True)

        for i, file in enumerate(os.listdir(src_folder)):
            src_file = os.path.join(src_folder, file)
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                shutil.copy2(src_file, os.path.join(dst_folder, f'web_{i}_{file}'))

    if os.path.exists(combined_test_dir):
        shutil.rmtree(combined_test_dir)
    shutil.copytree(original_test_dir, combined_test_dir)

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

class ResNetFineTuned(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNetFineTuned, self).__init__()
        self.model = models.resnet18(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
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

def train_model():
    freeze_support()
    combine_datasets()

    train_set = ImageFolder(combined_train_dir, transform=transform)
    val_set = ImageFolder(combined_test_dir, transform=transform)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    print("Classes:", train_set.classes)
    print(f"Training samples: {len(train_set)}, Validation samples: {len(val_set)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetFineTuned(num_classes=4).to(device)

    pretrained_path = 'brain_tumor_model/resnet18_brain_tumor.pth'
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        print("Loaded pretrained model.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    num_epochs = 10
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

    torch.save(model.state_dict(), os.path.join(model_save_dir, 'resnet18_brain_tumor_finetuned.pth'))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label='Train Loss', color='blue')
    plt.plot(val_loss_list, label='Val Loss', color='orange')
    plt.title('Loss Curve')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.grid(True); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_list, label='Val Accuracy', color='green')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy (%)'); plt.grid(True); plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(model_save_dir, 'training_metrics_finetune.png'))
    plt.show()

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

    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_dir, 'confusion_matrix_finetune.png'))
    plt.show()

    # === Per-Class Metrics Bar Plot ===
    metrics = ['precision', 'recall', 'f1-score']
    class_metrics = {m: [] for m in metrics}
    classes = target_names

    for cls in classes:
        for m in metrics:
            class_metrics[m].append(report[cls][m])

    x = np.arange(len(classes))
    width = 0.25

    plt.figure(figsize=(10, 5))
    plt.bar(x - width, class_metrics['precision'], width=width, label='Precision')
    plt.bar(x, class_metrics['recall'], width=width, label='Recall')
    plt.bar(x + width, class_metrics['f1-score'], width=width, label='F1 Score')

    plt.xticks(x, classes, rotation=45)
    plt.ylim(0, 1.05)
    plt.title('Per-Class Metrics')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_dir, 'per_class_metrics_finetune.png'))
    plt.show()

    # === Print Overall Accuracy ===
    overall_accuracy = np.mean(np.array(true_labels) == np.array(pred_labels)) * 100
    print(f"\n✅ Overall Accuracy on Validation Set: {overall_accuracy:.2f}%")

if __name__ == '__main__':
    train_model()
