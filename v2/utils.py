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
        """
        Args:
            cover_dir: Path to directory containing cover images
            stego_dir: Path to directory containing stego images
            transform: Optional transform to be applied on images
        """
        self.cover_dir = os.path.expanduser(cover_dir)
        self.stego_dir = os.path.expanduser(stego_dir)
        self.transform = transform
        self.images = []
        self.labels = []

        # Load Cover images (label 0)
        if os.path.exists(self.cover_dir):
            cover_files = [f for f in os.listdir(self.cover_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pgm'))]
            for img_name in cover_files:
                self.images.append(os.path.join(self.cover_dir, img_name))
                self.labels.append(0)
            print(f"Loaded {len(cover_files)} cover images from {self.cover_dir}")
        else:
            print(f"Warning: Cover directory not found: {self.cover_dir}")

        # Load Stego images (label 1)
        if os.path.exists(self.stego_dir):
            stego_files = [f for f in os.listdir(self.stego_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pgm'))]
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
            # Load image
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]

            if self.transform:
                image = self.transform(image)

            return image, torch.tensor(label, dtype=torch.float32)

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if there's an error
            if self.transform:
                blank_img = Image.new('RGB', (256, 256), color=(0, 0, 0))
                image = self.transform(blank_img)
            else:
                image = torch.zeros(3, 256, 256)
            return image, torch.tensor(0.0, dtype=torch.float32)


def get_transforms(img_size=256, augment=True):
    """
    Get data transforms for training and validation

    Args:
        img_size: Target image size (default: 256)
        augment: Whether to apply data augmentation (default: True)

    Returns:
        train_transform, val_transform: Transform pipelines
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor()
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    return train_transform, val_transform


def get_data_loaders(cover_train_dir, stego_train_dir, cover_val_dir, stego_val_dir,
                     batch_size=32, img_size=256, augment=True, num_workers=4):
    """
    Create data loaders for training and validation

    Args:
        cover_train_dir: Path to training cover images
        stego_train_dir: Path to training stego images
        cover_val_dir: Path to validation cover images
        stego_val_dir: Path to validation stego images
        batch_size: Batch size (default: 32)
        img_size: Image size (default: 256)
        augment: Whether to apply augmentation (default: True)
        num_workers: Number of data loading workers (default: 4)

    Returns:
        train_loader, val_loader: DataLoader objects
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
    """
    Create data loader for testing

    Args:
        cover_test_dir: Path to test cover images
        stego_test_dir: Path to test stego images
        batch_size: Batch size (default: 32)
        img_size: Image size (default: 256)
        num_workers: Number of data loading workers (default: 4)

    Returns:
        test_loader: DataLoader object
    """
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
    """
    Calculate classification metrics

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        dict: Dictionary containing accuracy, precision, recall, and F1 score
    """
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
    """
    Save model checkpoint

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath, device):
    """
    Load model checkpoint

    Args:
        model: PyTorch model
        optimizer: Optimizer
        filepath: Path to checkpoint file
        device: Device to load model to

    Returns:
        epoch, loss: Loaded epoch and loss
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {filepath} (Epoch {epoch}, Loss: {loss:.4f})")
    return epoch, loss


if __name__ == '__main__':
    # Test dataset loading
    print("Testing SteganalysisDataset...")
    print("="*70)

    cover_dir = '/Users/dmitryhoma/Projects/datasets/ready_to_use/GBRASNET/BOWS2/cover/train'
    stego_dir = '/Users/dmitryhoma/Projects/datasets/ready_to_use/GBRASNET/BOWS2/stego/WOW/0.2bpp/train'

    if os.path.exists(cover_dir) and os.path.exists(stego_dir):
        dataset = SteganalysisDataset(cover_dir, stego_dir)
        print(f"\nDataset size: {len(dataset)}")

        # Test loading a few samples
        print("\nTesting sample loading...")
        for i in range(min(3, len(dataset))):
            img, label = dataset[i]
            print(f"Sample {i}: Label={int(label)} ({'Cover' if label == 0 else 'Stego'})")

        print("\n" + "="*70)
        print("Dataset test successful!")
    else:
        print(f"Data directories not found")
        print("Please check the dataset paths.")
