import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class SteganalysisDataset(Dataset):
    """
    Steganalysis dataset for binary classification
    Label 0: Cover images (clean images without hidden data)
    Label 1: Stego images (images with hidden data)
    """
    def __init__(self, cover_dir, stego_dir, transform=None):
        self.cover_dir = os.path.expanduser(cover_dir)
        self.stego_dir = os.path.expanduser(stego_dir)
        self.transform = transform
        self.images = []
        self.labels = []

        # Load Cover images (label 0)
        if os.path.exists(self.cover_dir):
            cover_files = sorted([f for f in os.listdir(self.cover_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pgm'))])
            for img_name in cover_files:
                self.images.append(os.path.join(self.cover_dir, img_name))
                self.labels.append(0)
            print(f"Loaded {len(cover_files)} cover images from {self.cover_dir}")
        else:
            print(f"Warning: Cover directory not found: {self.cover_dir}")

        # Load Stego images (label 1)
        if os.path.exists(self.stego_dir):
            stego_files = sorted([f for f in os.listdir(self.stego_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pgm'))])
            for img_name in stego_files:
                self.images.append(os.path.join(self.stego_dir, img_name))
                self.labels.append(1)
            print(f"Loaded {len(stego_files)} stego images from {self.stego_dir}")
        else:
            print(f"Warning: Stego directory not found: {self.stego_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        try:
            # Load image WITHOUT converting to RGB first
            image = Image.open(img_path)
            
            # CRITICAL: Keep as grayscale if it's grayscale
            # This preserves the subtle steganographic changes
            if image.mode != 'RGB':
                image = image.convert('L')  # Ensure it's grayscale
                # Convert to RGB by replicating channel AFTER transforms
            
            label = self.labels[idx]

            if self.transform:
                image = self.transform(image)
                
                # If image is still 1 channel after transform, replicate to 3
                if image.shape[0] == 1:
                    image = image.repeat(3, 1, 1)

            return image, torch.tensor(label, dtype=torch.float32)

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            if self.transform:
                blank_img = Image.new('L', (256, 256), color=128)
                image = self.transform(blank_img)
                if image.shape[0] == 1:
                    image = image.repeat(3, 1, 1)
            else:
                image = torch.zeros(3, 256, 256)
            return image, torch.tensor(0.0, dtype=torch.float32)


def get_transforms(img_size=256, augment=True):
    """
    Get data transforms for training and validation
    CRITICAL: No aggressive augmentation for steganalysis!
    Augmentation can destroy the steganographic signal.
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),  # OK for steganalysis
            # NO rotation, color jitter, or other transforms!
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])

    return train_transform, val_transform


def get_data_loaders(cover_train_dir, stego_train_dir, cover_val_dir, stego_val_dir,
                     batch_size=32, img_size=256, augment=False, num_workers=4):
    """
    Create data loaders for training and validation
    
    Note: augment=False by default for steganalysis!
    """
    train_transform, val_transform = get_transforms(img_size, augment)

    # Training dataset
    train_dataset = SteganalysisDataset(cover_train_dir, stego_train_dir,
                                        transform=train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # Validation dataset
    val_loader = None
    if (cover_val_dir and os.path.exists(os.path.expanduser(cover_val_dir)) and
        stego_val_dir and os.path.exists(os.path.expanduser(stego_val_dir))):
        val_dataset = SteganalysisDataset(cover_val_dir, stego_val_dir,
                                          transform=val_transform)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    return train_loader, val_loader


def get_test_loader(cover_test_dir, stego_test_dir, batch_size=32,
                    img_size=256, num_workers=4):
    """Create data loader for testing"""
    _, test_transform = get_transforms(img_size, augment=False)

    test_dataset = SteganalysisDataset(cover_test_dir, stego_test_dir,
                                       transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return test_loader


def calculate_metrics(y_true, y_pred):
    """Calculate classification metrics"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath, device):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {filepath} (Epoch {epoch}, Loss: {loss:.4f})")
    return epoch, loss