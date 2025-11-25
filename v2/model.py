import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights


class SimpleCNN(nn.Module):
    """Simple CNN for Steganalysis (Cover vs Stego classification)"""
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 128x128

            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64x64

            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x32

            # Conv Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResNet50Classifier(nn.Module):
    """ResNet-50 based classifier for Steganalysis"""
    def __init__(self, pretrained=True, freeze_backbone=False):
        super(ResNet50Classifier, self).__init__()

        # Load pretrained ResNet-50
        if pretrained:
            resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            resnet = models.resnet50(weights=None)

        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        # Custom classifier for binary classification
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class EfficientNetClassifier(nn.Module):
    """EfficientNet-B0 based classifier for Steganalysis"""
    def __init__(self, pretrained=True, freeze_backbone=False):
        super(EfficientNetClassifier, self).__init__()

        # Load pretrained EfficientNet-B0
        if pretrained:
            efficientnet = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            efficientnet = models.efficientnet_b0(weights=None)

        # Extract features (remove classifier)
        self.features = efficientnet.features
        self.avgpool = efficientnet.avgpool

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        # Custom classifier for binary classification
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


def get_model(model_name='resnet50', pretrained=True, freeze_backbone=False):
    """
    Factory function to get model by name

    Args:
        model_name: One of ['simple', 'resnet50', 'efficientnet']
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze backbone weights

    Returns:
        model: PyTorch model
    """
    if model_name == 'simple':
        return SimpleCNN()
    elif model_name == 'resnet50':
        return ResNet50Classifier(pretrained=pretrained, freeze_backbone=freeze_backbone)
    elif model_name == 'efficientnet':
        return EfficientNetClassifier(pretrained=pretrained, freeze_backbone=freeze_backbone)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


if __name__ == '__main__':
    # Test models
    batch_size = 4
    x = torch.randn(batch_size, 3, 256, 256)

    print("Testing models...")
    print("="*70)

    for model_name in ['simple', 'resnet50', 'efficientnet']:
        print(f"\n{model_name.upper()}:")
        model = get_model(model_name, pretrained=False)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(x)
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

    print("\n" + "="*70)
    print("All models tested successfully!")
