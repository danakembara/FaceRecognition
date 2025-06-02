import os
import pandas as pd
import random
import time

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image


class FaceRecognitionDataset(Dataset):
    def __init__(self, root, split='train', transforms=None):

        assert split in ['train', 'val', 'test']    # split must be 'train', 'val', or 'test'
        self.transforms = transforms

        image_dir = root + 'dataset/images'
        label_dir = root + 'dataset/labels'

        # Load and clean data
        df = pd.read_csv(
            os.path.join(label_dir, 'list_attribute.txt'),
            sep='\s+',
            skiprows=1
        )
        df = df.reset_index().rename(columns={'index': 'Filepath'})                     # Use the index as filepath
        df = df[['Filepath', 'Male']]                                                   # Select relevant columns ('Filepath' as dir to images, 'Male' as target)
        df['Filepath'] = df['Filepath'].apply(lambda x: os.path.join(image_dir, x))     # Join Filepath with image_dir to form a complete dir
        df = df[df['Filepath'].apply(os.path.exists)].reset_index(drop=True)            # Only select Filepath that exists with its corresponding image (filter process) 
        df['Male'] = df['Male'].apply(lambda x: 1 if x == 1 else 0)                     # Convert -1 into 0 (Female) for binary classification

        # # (OPTIONAL) Save cleaned dataset to CSV for checking
        # df.to_csv(os.path.join(label_dir, 'cleaned_list_attribute.csv'), index=False)

        # Fill images and labels into lists
        all_images = df['Filepath'].tolist()
        all_labels = df['Male'].tolist()

        # Split train, val and test (70%, 15%, 15%)
        X_train, X_temp, y_train, y_temp = train_test_split(
            all_images, all_labels, test_size=0.30, random_state=42, stratify=all_labels    # Stratify is used to ensures class balance in all sets
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
        )

        if split == 'train':
            self.images = X_train
            self.labels = y_train
        elif split == 'val':
            self.images = X_val
            self.labels = y_val
        else:
            self.images = X_test
            self.labels = y_test

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        label = torch.tensor(self.labels[idx])

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label
    

class CustomTransform:
    def __init__(self, p=0.5, seed=None):   # Applies data augmentation transformations to an image with probability `p`
        self.p = p
        self.seed = seed

        self.image_transform = transforms.Compose([    # for train set, custom transform if augmented
            transforms.Resize((224, 224)),             # Resize to consistent size (224x224)
            transforms.RandomHorizontalFlip(),         # Flip the images horizontally
            transforms.RandomRotation(5),              # Rotate randomly 15 degrees
            transforms.ColorJitter(                    # Slight brightness, contrast, saturation, hue
                brightness=0.05,
                contrast=0.05,
                saturation=0.05,
                hue=0.05
            ),
            transforms.ToTensor(),                     # Convert to tensor
            transforms.Normalize(                      # Normalize (ImageNet stats)
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.basic_transform = transforms.Compose([    # for train set, use basic transform if not augmented
            transforms.Resize((224, 224)),
            transforms.ToTensor(),                     # Convert to tensor
            transforms.Normalize(                      # Normalize (ImageNet stats)
                mean=[0.485, 0.456, 0.406],         
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __call__(self, img):
        if self.seed is not None:
            random.seed(self.seed)

        if random.random() < self.p:              # Use augmentation
            return self.image_transform(img)
        else:
            return self.basic_transform(img)      # No augmentation


class NoTransform:
    def __init__(self):                           # For validation and test set (no augmentation, only resizes, converts to tensor, and normalizes the image)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __call__(self, img):
        return self.transform(img)


def visualize_dataset(dataset, num_samples=5, randomize=True):
    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (tensor * std + mean).clamp(0, 1)

    indices = random.sample(range(len(dataset)), num_samples) if randomize else list(range(num_samples))

    images = []
    labels = []

    for i in indices:
        image, label = dataset[i]
        image = denormalize(image)
        images.append(image)
        labels.append(int(label))

    plt.figure(figsize=(num_samples * 2.5, 3))

    for i in range(num_samples):
        gender_str = 'female' if labels[i] == 0 else 'male'
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i].permute(1, 2, 0).cpu().numpy())
        plt.title(gender_str)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def visualize_augmented_dataset(dataset, num_samples=5, randomize=True):
    no_aug = NoTransform()
    with_aug = CustomTransform(p=1.0)  # always apply augmentation because we want to see the visualization

    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)        
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (tensor * std + mean).clamp(0, 1)

    num_samples = min(num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples) if randomize else list(range(num_samples))

    original_images = []
    augmented_images = []

    for i in indices:
        img_path = dataset.images[i]
        img = Image.open(img_path).convert('RGB')

        original = denormalize(no_aug(img))
        augmented = denormalize(with_aug(img))

        original_images.append(original)
        augmented_images.append(augmented)

    # Stack images horizontally in each row
    top_row = torch.cat(original_images, dim=2)
    bottom_row = torch.cat(augmented_images, dim=2)

    # Combine rows vertically
    grid_image = torch.cat([top_row, bottom_row], dim=1)

    plt.figure(figsize=(num_samples * 2.5, 5))
    plt.imshow(grid_image.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.title("Top: Original images  |  Bottom: Augmented images")
    plt.show()


def count_parameters(model, model_name="Model"):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{model_name}:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")


def load_train_data(root, batch_size, transforms):
    dataset = FaceRecognitionDataset(root=root, split='train', transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=1)
    return dataloader


def load_val_data(root, batch_size, transforms):
    dataset = FaceRecognitionDataset(root=root, split='val', transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=1)
    return dataloader


def load_test_data(root, batch_size, transforms):
    dataset = FaceRecognitionDataset(root=root, split='test', transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=1)
    return dataloader


def train(model, train_loader, val_loader, num_epochs, learning_rate, model_name='model', save_dir='weights'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()                                                  # binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)     # learning rate scheduler

    save_path = os.path.join(save_dir, f'{model_name}_last.pth')
    start_time = time.time()

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        # Train
        model.train()
        total_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().to(device)  # float since we implement BCEWithLogitsLoss

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)

            # backward and optimizer
            loss.backward()
            optimizer.step()

            # calculate loss, pred, correct
            total_loss += loss.item() * images.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels.bool()).sum().item()
            total += labels.size(0)

        # Avg loss and accuracy
        avg_train_loss = total_loss / total
        train_acc = correct / total * 100

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().to(device)

                outputs = model(images).squeeze(1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = torch.sigmoid(outputs) > 0.5
                val_correct += (preds == labels.bool()).sum().item()
                val_total += labels.size(0)

        # Avg loss and accuracy
        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total * 100

        # Append to the lists
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"[{model_name}] Epoch {epoch + 1}/{num_epochs} — "
              f"Train Loss: {avg_train_loss:.4f} — Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} — Val Acc: {val_acc:.2f}%")

        scheduler.step()

    # Save the model based on its last epoch
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to: {save_path}")
    print(f"Training time for {model_name}: {time.time() - start_time:.2f} seconds")

    # Plotting losses and accuracies
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(val_accuracies, label='Val Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()


def evaluate(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    start_time = time.time()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.float().to(device)

            outputs = model(images).squeeze(1)
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    end_time = time.time()
    inference_time = end_time - start_time

    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Female", "Male"], yticklabels=["Female", "Male"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # Report
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")
    print(f"Inference Time: {inference_time:.2f} seconds")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "inference_time": inference_time
    }
