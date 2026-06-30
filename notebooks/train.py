"""
train.py
Builds and trains a ResNet18-based classifier on the Brain Tumor MRI dataset
using transfer learning. Saves the trained model + prints evaluation metrics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import get_dataloaders

# ---- Config ----
DEVICE = torch.device("cpu")
EPOCHS = 10
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = "../models/mri_classifier.pt"

print(f"Using device: {DEVICE}")


def build_model(num_classes):
    """
    Loads a pretrained ResNet18 and replaces its final layer.
    This is transfer learning: we keep all the learned image features
    from ImageNet, and only retrain the last layer for our 4 classes.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze all earlier layers — they already know how to detect edges,
    # textures, shapes. We don't want to destroy that knowledge.
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final classification layer to output 4 classes instead of 1000
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model.to(DEVICE)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, all_preds, all_labels


def main():
    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    num_classes = len(class_names)

    model = build_model(num_classes)
    criterion = nn.CrossEntropyLoss()
    # Only the new final layer has requires_grad=True, so only it gets trained
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    print("\nStarting training...\n")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    # ---- Final test evaluation ----
    print("\nEvaluating on held-out test set...\n")
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion)
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=class_names))

    # ---- Save model ----
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
    }, MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")

    # ---- Plot accuracy curve ----
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.savefig("../docs/accuracy_curve.png")
    print("Accuracy curve saved to docs/accuracy_curve.png")

    # ---- Plot confusion matrix ----
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("../docs/confusion_matrix.png")
    print("Confusion matrix saved to docs/confusion_matrix.png")


if __name__ == "__main__":
    main()
