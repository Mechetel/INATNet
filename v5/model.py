import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np
from torchvision import models


class PreLayer3x3(nn.Module):
    """3x3 SRM preprocessing layer."""

    def __init__(self, stride=1, padding=1):
        super(PreLayer3x3, self).__init__()
        self.in_channels = 1
        self.out_channels = 25
        self.kernel_size = (3, 3)
        self.stride = (stride, stride)
        self.padding = (padding, padding)

        self.weight = Parameter(torch.Tensor(25, 1, 3, 3), requires_grad=True)
        self.bias = Parameter(torch.Tensor(25), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        # Load SRM kernels from numpy file
        srm_npy = np.load('kernels/SRM3_3.npy')
        self.weight.data.numpy()[:] = srm_npy
        self.bias.data.zero_()

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding)


class PreLayer5x5(nn.Module):
    """5x5 SRM preprocessing layer."""

    def __init__(self, stride=1, padding=2):
        super(PreLayer5x5, self).__init__()
        self.in_channels = 1
        self.out_channels = 5
        self.kernel_size = (5, 5)
        self.stride = (stride, stride)
        self.padding = (padding, padding)

        self.weight = Parameter(torch.Tensor(5, 1, 5, 5), requires_grad=True)
        self.bias = Parameter(torch.Tensor(5), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        # Load SRM kernels from numpy file
        srm_npy = np.load('kernels/SRM5_5.npy')
        self.weight.data.numpy()[:] = srm_npy
        self.bias.data.zero_()

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding)


class PreprocessingLayer(nn.Module):
    """Preprocessing layer combining 3x3 and 5x5 SRM filters."""

    def __init__(self):
        super(PreprocessingLayer, self).__init__()
        self.conv1 = PreLayer3x3()
        self.conv2 = PreLayer5x5()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        return torch.cat([x1, x2], dim=1)  # 30 channels


class ChannelAttention(nn.Module):
    """Channel Attention Module from CBAM."""

    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module from CBAM."""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # Channel attention
        x = x * self.channel_attention(x)
        # Spatial attention
        x = x * self.spatial_attention(x)
        return x


class ConvLayer(nn.Module):
    """Main convolutional layers using pretrained DenseNet201 with CBAM."""

    def __init__(self):
        super(ConvLayer, self).__init__()

        # Adapter layer to convert 30 channels to 3 channels for DenseNet
        self.adapter = nn.Conv2d(30, 3, kernel_size=1, stride=1, padding=0)

        # Load pretrained DenseNet201
        densenet = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)

        # Extract features (everything except the classifier)
        self.densenet_features = densenet.features

        # DenseNet201 outputs 1920 channels
        self.densenet_out_channels = 1920

        # CBAM attention module
        self.cbam = CBAM(self.densenet_out_channels, reduction_ratio=16)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Residual projection for skip connection (from DenseNet input to classifier)
        self.residual_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(3, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Classifier with residual connection
        self.classifier = nn.Sequential(
            nn.Linear(self.densenet_out_channels + 512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        # Adapt from 30 channels to 3 channels for DenseNet
        x_adapted = self.adapter(x)

        # Store for residual connection
        residual = self.residual_proj(x_adapted)
        residual = residual.view(residual.size(0), -1)

        # Pass through DenseNet features
        x = self.densenet_features(x_adapted)

        # Apply CBAM attention
        x = self.cbam(x)

        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Concatenate with residual connection
        x = torch.cat([x, residual], dim=1)

        # Classification
        x = self.classifier(x)

        return x


class INATNet(nn.Module):
    """
    INATNet for image steganalysis with DenseNet201 backbone and CBAM.

    Architecture:
    1. SRM preprocessing layer (3x3 and 5x5 filters) -> 30 channels
    2. Adapter layer (30 channels -> 3 channels)
    3. Pretrained DenseNet201 features (retrainable)
    4. CBAM attention module
    5. Residual connection from DenseNet input to classifier
    6. Fully connected classifier
    """

    def __init__(self):
        super(INATNet, self).__init__()
        self.layer1 = PreprocessingLayer()
        self.layer2 = ConvLayer()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


# Example usage
if __name__ == "__main__":
    # Create model
    model = INATNet()

    # Example input (batch_size=2, channels=1, height=256, width=256)
    x = torch.randn(2, 1, 256, 256)

    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
