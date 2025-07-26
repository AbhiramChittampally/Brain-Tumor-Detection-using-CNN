import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

# === Paths ===
training_data_dir = 'brain_tumor_dataset/Training'
testing_data_dir = 'brain_tumor_dataset/Testing'
model_save_dir = 'brain_tumor_model'
os.makedirs(model_save_dir, exist_ok=True)

# Enhanced Augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Increased resolution for ResNet
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Simple transform for validation/test
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Custom dataset for quality control
class CleanDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root_dir, transform=transform)
        self.clean_indices = self._filter_corrupted()
        self.classes = self.dataset.classes  # Expose classes attribute
        self.class_to_idx = self.dataset.class_to_idx  # Expose class mapping
        
    def _filter_corrupted(self):
        valid_indices = []
        for idx in range(len(self.dataset)):
            try:
                img, _ = self.dataset[idx]
                # Check for valid image tensor
                if not torch.isnan(img).any() and img.max() > 0:
                    valid_indices.append(idx)
            except:
                continue
        return valid_indices
    
    def __len__(self):
        return len(self.clean_indices)
    
    def __getitem__(self, idx):
        real_idx = self.clean_indices[idx]
        return self.dataset[real_idx]

# Calculate class weights
def get_class_weights(dataset):
    class_counts = [0] * len(dataset.classes)
    for _, label in dataset:
        class_counts[label] += 1
    total_samples = sum(class_counts)
    return torch.tensor([total_samples / count for count in class_counts], dtype=torch.float)

# Enhanced Model Architecture
class ResNetFineTuned(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNetFineTuned, self).__init__()
        # Use updated model loading method
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.model = models.resnet18(weights=weights)
        
        # Gradual unfreezing strategy
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Unfreeze deeper layers progressively
        for name, child in self.model.named_children():
            if name in ['layer3', 'layer4', 'fc']:
                for param in child.parameters():
                    param.requires_grad = True
                    
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased dropout
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Training Function with enhancements
def train_model():
    # Load data with quality control
    train_set = CleanDataset(training_data_dir, transform=train_transform)
    test_set = CleanDataset(testing_data_dir, transform=val_transform)
    
    # Split training into train/validation
    train_indices, val_indices = train_test_split(
        range(len(train_set)),
        test_size=0.2,
        stratify=[label for _, label in train_set],
        random_state=42
    )
    
    # Create subsets
    train_subset = torch.utils.data.Subset(train_set, train_indices)
    val_subset = torch.utils.data.Subset(train_set, val_indices)
    
    # Handle class imbalance
    class_weights = get_class_weights(train_set)
    sampler_weights = [class_weights[label] for _, label in train_subset]
    sampler = WeightedRandomSampler(sampler_weights, len(sampler_weights))
    
    # DataLoaders
    train_loader = DataLoader(train_subset, batch_size=32, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    print("Classes:", train_set.classes)
    print(f"Training samples: {len(train_subset)}, Validation samples: {len(val_subset)}, Test samples: {len(test_set)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetFineTuned(num_classes=4).to(device)
    
    # Loss function with class weighting
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # Remove verbose parameter for compatibility
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    num_epochs = 30
    best_val_acc = 0.0
    current_lr = optimizer.param_groups[0]['lr']

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_subset)
        avg_val_loss = val_loss / len(val_subset)
        val_acc = 100 * correct / total
        
        scheduler.step(avg_val_loss)
        
        # Check if learning rate changed
        new_lr = optimizer.param_groups[0]['lr']
        lr_changed = " (LR reduced!)" if new_lr < current_lr else ""
        current_lr = new_lr
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}% | LR: {current_lr:.7f}{lr_changed}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(model_save_dir, 'resnet18_brain_tumor.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model with val acc: {val_acc:.2f}%")

    # Final evaluation on test set
    test_model(model, test_loader, device, test_set.classes)  # Pass classes directly

def test_model(model, test_loader, device, target_names):
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    test_acc = 100 * correct / total
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_dir, 'confusion_matrix.png'))
    plt.show()
    
    # Per-class metrics
    report = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True)
    precision = [report[cls]['precision'] for cls in target_names]
    recall = [report[cls]['recall'] for cls in target_names]
    f1 = [report[cls]['f1-score'] for cls in target_names]
    
    x = np.arange(len(target_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision')
    ax.bar(x, recall, width, label='Recall')
    ax.bar(x + width, f1, width, label='F1-Score')
    
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