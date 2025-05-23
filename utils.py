import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import torch.nn as nn
import random
from matplotlib import pyplot as plt
import time


class FaceRecognitionDataset(Dataset):
    def __init__(self, root, is_train=True, transforms=None):
        self.images = []
        self.labels = []
        self.transforms = transforms

        image_dir = root + 'dataset/images'
        label_dir = root + 'dataset/labels'

        # Load and clean data
        df = pd.read_csv(
            os.path.join(label_dir, 'list_attribute.txt'),
            sep='\s+',
            skiprows=1
        )
        df = df.reset_index().rename(columns={'index': 'Filepath'})
        df = df[['Filepath', 'Male']]
        df['Filepath'] = df['Filepath'].apply(lambda x: os.path.join(image_dir, x))
        df = df[df['Filepath'].apply(os.path.exists)].reset_index(drop=True)
        df['Male'] = df['Male'].apply(lambda x: 1 if x == 1 else 0)

        # Save cleaned dataset to CSV 
        df.to_csv(os.path.join(label_dir, 'cleaned_list_attribute.csv'), index=False)

        # Fill images and labels lists
        for _, row in df.iterrows():
            self.images.append(row['Filepath'])
            self.labels.append(row['Male'])

        # Split train and test
        X_train, X_test, y_train, y_test = train_test_split(
            self.images, self.labels, test_size=0.20, random_state=42
        )

        if is_train:
            self.images = X_train
            self.labels = y_train
        else:
            self.images = X_test
            self.labels = y_test

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label
    

class CustomTransform:
    """
    Applies data augmentation transformations to an image with probability `p`.
    """
    def __init__(self, p=0.5, seed=None):
        self.p = p
        self.seed = seed

        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),             # Resize to consistent size
            transforms.RandomHorizontalFlip(p=0.5),    # Flip half of the images horizontally
            transforms.RandomRotation(15),             # Rotate randomly ±15 degrees
            transforms.ColorJitter(                    # Slight brightness, contrast, etc.
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05
            ),
            transforms.ToTensor(),                     # Convert to tensor
            transforms.Normalize(                      # Normalize (ImageNet stats)
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.basic_transform = transforms.Compose([
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

        if random.random() < self.p:
            return self.image_transform(img)
        else:
            return self.basic_transform(img)


class NoTransform:
    """
    Only resizes, converts to tensor, and normalizes the image (no augmentation).
    """
    def __init__(self):
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
    """
    Displays a row of denormalized images from a dataset with gender labels.

    Args:
        dataset (FaceRecognitionDataset): Dataset instance.
        num_samples (int): Number of images to display.
        randomize (bool): Whether to randomly pick the samples.
    """
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
    """
    Shows original images on the top row and augmented images on the bottom row.

    Args:
        dataset (FaceRecognitionDataset): Dataset with raw image paths.
        num_samples (int): Number of samples to visualize.
        randomize (bool): Whether to randomly pick the samples.
    """
    no_aug = NoTransform()
    with_aug = CustomTransform(p=1.0)  # always apply augmentation

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


def load_train_data(root, batch_size, transforms):
    dataset = FaceRecognitionDataset(root=root, is_train=True, transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=1)
    return dataloader


def load_test_data(root, batch_size, transforms):
    dataset = FaceRecognitionDataset(root=root, is_train=False, transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=1)
    return dataloader


def train(model, train_loader, num_epochs, learning_rate, model_name='model', save_dir='weights'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_accuracy = 0.0
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{model_name}_best.pth')

    start_time = time.time()  # start timer here

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            predictions = torch.sigmoid(outputs) > 0.5
            correct += (predictions == labels.bool()).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total * 100
        print(f"[{model_name}] Epoch {epoch + 1}/{num_epochs} — Loss: {avg_loss:.4f} — Accuracy: {accuracy:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved to: {save_path} (Accuracy: {best_accuracy:.2f}%)")

    end_time = time.time()  # end timer here
    elapsed = end_time - start_time
    print(f"Training time for {model_name}: {elapsed:.2f} seconds")


def evaluate(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.float().to(device)

            outputs = model(images).squeeze(1)
            predictions = torch.sigmoid(outputs) > 0.5

            correct += (predictions == labels.bool()).sum().item()
            total += labels.size(0)

    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy