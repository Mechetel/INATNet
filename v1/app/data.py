import os
from pathlib import Path
from typing import List, Tuple
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from app.config import Config

# ==================== Data Manager ====================

class SteganalysisDataset(Dataset):
    """Dataset for steganalysis"""

    def __init__(self, cover_dir: str, stego_dir: str, transform=None):
        self.cover_images = self._load_image_paths(cover_dir)
        self.stego_images = self._load_image_paths(stego_dir)

        # Combine and create labels
        self.images = self.cover_images + self.stego_images
        self.labels = [0] * len(self.cover_images) + [1] * len(self.stego_images)
        self.transform = transform

    def _load_image_paths(self, directory: str) -> List[str]:
        """Load all image paths from directory"""
        extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.pgm')
        paths = []
        for ext in extensions:
            paths.extend(Path(directory).glob(f"*{ext}"))
            paths.extend(Path(directory).glob(f"*{ext.upper()}"))
        return [str(p) for p in paths]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


class DataManager:
    """Manage data loading and preprocessing"""

    def __init__(self, config: Config):
        self.config = config
        self.transform_train = self._get_train_transforms()
        self.transform_val = self._get_val_transforms()

    def _get_train_transforms(self):
        """Get training transforms"""
        transforms = []
        # transforms.append(T.Resize((self.config.image_size, self.config.image_size)))

        if self.config.use_augmentation:
            if self.config.random_flip:
                transforms.append(T.RandomHorizontalFlip())
                transforms.append(T.RandomVerticalFlip())
            if self.config.random_crop:
                transforms.append(T.RandomResizedCrop(self.config.image_size, scale=(0.8, 1.0)))
            if self.config.rotation_degrees > 0:
                transforms.append(T.RandomRotation(self.config.rotation_degrees))

        transforms.append(T.ToTensor())
        return T.Compose(transforms)

    def _get_val_transforms(self):
        """Get validation transforms"""
        transforms = []
        # transforms.append(T.Resize((self.config.image_size, self.config.image_size)))
        transforms.append(T.ToTensor())

        return T.Compose(transforms)

    def create_dataloaders(self, train_cover_dir: str, train_stego_dir: str,
                          val_cover_dir: str, val_stego_dir: str) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation dataloaders"""
        train_dataset = SteganalysisDataset(train_cover_dir, train_stego_dir,
                                           transform=self.transform_train)
        val_dataset = SteganalysisDataset(val_cover_dir, val_stego_dir,
                                         transform=self.transform_val)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=False
        )

        return train_loader, val_loader
