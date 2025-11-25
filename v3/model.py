"""Enhanced SRNet with Attention Mechanisms (SE and CBAM)"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Srnet(nn.Module):
    """Original SRNet architecture"""
    def __init__(self):
        super(Srnet, self).__init__()
        # Layer 1
        self.layer1 = nn.Conv2d(in_channels=1, out_channels=64,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Layer 2
        self.layer2 = nn.Conv2d(in_channels=64, out_channels=16,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)

        # Layer 3
        self.layer31 = nn.Conv2d(in_channels=16, out_channels=16,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn31 = nn.BatchNorm2d(16)
        self.layer32 = nn.Conv2d(in_channels=16, out_channels=16,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn32 = nn.BatchNorm2d(16)

        # Layer 4
        self.layer41 = nn.Conv2d(in_channels=16, out_channels=16,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn41 = nn.BatchNorm2d(16)
        self.layer42 = nn.Conv2d(in_channels=16, out_channels=16,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn42 = nn.BatchNorm2d(16)

        # Layer 5
        self.layer51 = nn.Conv2d(in_channels=16, out_channels=16,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn51 = nn.BatchNorm2d(16)
        self.layer52 = nn.Conv2d(in_channels=16, out_channels=16,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn52 = nn.BatchNorm2d(16)

        # Layer 6
        self.layer61 = nn.Conv2d(in_channels=16, out_channels=16,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn61 = nn.BatchNorm2d(16)
        self.layer62 = nn.Conv2d(in_channels=16, out_channels=16,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn62 = nn.BatchNorm2d(16)

        # Layer 7
        self.layer71 = nn.Conv2d(in_channels=16, out_channels=16,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn71 = nn.BatchNorm2d(16)
        self.layer72 = nn.Conv2d(in_channels=16, out_channels=16,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn72 = nn.BatchNorm2d(16)

        # Layer 8
        self.layer81 = nn.Conv2d(in_channels=16, out_channels=16,kernel_size=1, stride=2, padding=0, bias=False)
        self.bn81 = nn.BatchNorm2d(16)
        self.layer82 = nn.Conv2d(in_channels=16, out_channels=16,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn82 = nn.BatchNorm2d(16)
        self.layer83 = nn.Conv2d(in_channels=16, out_channels=16,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn83 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # Layer 9
        self.layer91 = nn.Conv2d(in_channels=16, out_channels=64,kernel_size=1, stride=2, padding=0, bias=False)
        self.bn91 = nn.BatchNorm2d(64)
        self.layer92 = nn.Conv2d(in_channels=16, out_channels=64,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn92 = nn.BatchNorm2d(64)
        self.layer93 = nn.Conv2d(in_channels=64, out_channels=64,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn93 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # Layer 10
        self.layer101 = nn.Conv2d(in_channels=64, out_channels=128,kernel_size=1, stride=2, padding=0, bias=False)
        self.bn101 = nn.BatchNorm2d(128)
        self.layer102 = nn.Conv2d(in_channels=64, out_channels=128,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn102 = nn.BatchNorm2d(128)
        self.layer103 = nn.Conv2d(in_channels=128, out_channels=128,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn103 = nn.BatchNorm2d(128)
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # Layer 11
        self.layer111 = nn.Conv2d(in_channels=128, out_channels=256,kernel_size=1, stride=2, padding=0, bias=False)
        self.bn111 = nn.BatchNorm2d(256)
        self.layer112 = nn.Conv2d(in_channels=128, out_channels=256,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn112 = nn.BatchNorm2d(256)
        self.layer113 = nn.Conv2d(in_channels=256, out_channels=256,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn113 = nn.BatchNorm2d(256)

        # Layer 12
        self.layer121 = nn.Conv2d(in_channels=256, out_channels=512,kernel_size=3, stride=2, padding=0, bias=False)
        self.bn121 = nn.BatchNorm2d(512)
        self.layer122 = nn.Conv2d(in_channels=512, out_channels=512,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn122 = nn.BatchNorm2d(512)

        # Fully Connected layer
        self.fc = nn.Linear(512, 2)

    def forward(self, inputs):
        # Layer 1
        conv = self.layer1(inputs)
        actv = F.relu(self.bn1(conv))

        # Layer 2
        conv = self.layer2(actv)
        actv = F.relu(self.bn2(conv))

        # Layer 3
        conv1 = self.layer31(actv)
        actv1 = F.relu(self.bn31(conv1))
        conv2 = self.layer32(actv1)
        bn = self.bn32(conv2)
        res = torch.add(actv, bn)

        # Layer 4
        conv1 = self.layer41(res)
        actv1 = F.relu(self.bn41(conv1))
        conv2 = self.layer42(actv1)
        bn = self.bn42(conv2)
        res = torch.add(res, bn)

        # Layer 5
        conv1 = self.layer51(res)
        actv1 = F.relu(self.bn51(conv1))
        conv2 = self.layer52(actv1)
        bn = self.bn52(conv2)
        res = torch.add(res, bn)

        # Layer 6
        conv1 = self.layer61(res)
        actv1 = F.relu(self.bn61(conv1))
        conv2 = self.layer62(actv1)
        bn = self.bn62(conv2)
        res = torch.add(res, bn)

        # Layer 7
        conv1 = self.layer71(res)
        actv1 = F.relu(self.bn71(conv1))
        conv2 = self.layer72(actv1)
        bn = self.bn72(conv2)
        res = torch.add(res, bn)

        # Layer 8
        convs = self.layer81(res)
        convs = self.bn81(convs)
        conv1 = self.layer82(res)
        actv1 = F.relu(self.bn82(conv1))
        conv2 = self.layer83(actv1)
        bn = self.bn83(conv2)
        pool = self.pool1(bn)
        res = torch.add(convs, pool)

        # Layer 9
        convs = self.layer91(res)
        convs = self.bn91(convs)
        conv1 = self.layer92(res)
        actv1 = F.relu(self.bn92(conv1))
        conv2 = self.layer93(actv1)
        bn = self.bn93(conv2)
        pool = self.pool2(bn)
        res = torch.add(convs, pool)

        # Layer 10
        convs = self.layer101(res)
        convs = self.bn101(convs)
        conv1 = self.layer102(res)
        actv1 = F.relu(self.bn102(conv1))
        conv2 = self.layer103(actv1)
        bn = self.bn103(conv2)
        pool = self.pool1(bn)
        res = torch.add(convs, pool)

        # Layer 11
        convs = self.layer111(res)
        convs = self.bn111(convs)
        conv1 = self.layer112(res)
        actv1 = F.relu(self.bn112(conv1))
        conv2 = self.layer113(actv1)
        bn = self.bn113(conv2)
        pool = self.pool1(bn)
        res = torch.add(convs, pool)

        # Layer 12
        conv1 = self.layer121(res)
        actv1 = F.relu(self.bn121(conv1))
        conv2 = self.layer122(actv1)
        bn = self.bn122(conv2)

        avgp = torch.mean(bn, dim=(2,3), keepdim=True)
        flatten = avgp.view(avgp.size(0),-1)
        fc = self.fc(flatten)
        out = F.log_softmax(fc, dim=1)
        return fc


# ==================== Attention Modules ====================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    """Channel Attention for CBAM"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial Attention for CBAM"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(y))


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


class GCBlock(nn.Module):
    """Global Context Block"""
    def __init__(self, channels, reduction=16):
        super(GCBlock, self).__init__()
        self.conv_mask = nn.Conv2d(channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.transform = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.LayerNorm([channels // reduction, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        mask = self.conv_mask(x).view(b, 1, h * w)
        mask = self.softmax(mask).view(b, 1, h, w)
        context = (x * mask).view(b, c, h * w).sum(dim=2, keepdim=True).view(b, c, 1, 1)
        context = self.transform(context)
        return x + context


class TripletAttention(nn.Module):
    """Triplet Attention Module"""
    def __init__(self, kernel_size=7):
        super(TripletAttention, self).__init__()
        self.cw = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(1)
        )
        self.hc = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(1)
        )
        self.hw = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # C-W attention
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_avg1 = torch.mean(x_perm1, dim=1, keepdim=True)
        x_max1, _ = torch.max(x_perm1, dim=1, keepdim=True)
        att1 = self.sigmoid(self.cw(torch.cat([x_avg1, x_max1], dim=1)))
        att1 = att1.permute(0, 2, 1, 3).contiguous()

        # H-C attention
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_avg2 = torch.mean(x_perm2, dim=1, keepdim=True)
        x_max2, _ = torch.max(x_perm2, dim=1, keepdim=True)
        att2 = self.sigmoid(self.hc(torch.cat([x_avg2, x_max2], dim=1)))
        att2 = att2.permute(0, 3, 2, 1).contiguous()

        # H-W attention
        x_avg3 = torch.mean(x, dim=1, keepdim=True)
        x_max3, _ = torch.max(x, dim=1, keepdim=True)
        att3 = self.sigmoid(self.hw(torch.cat([x_avg3, x_max3], dim=1)))

        return (x * att1 + x * att2 + x * att3) / 3.0


class InceptionAttentionBlock(nn.Module):
    """Inception-style block with all four attention mechanisms"""
    def __init__(self, in_channels, out_channels, reduction=16):
        super(InceptionAttentionBlock, self).__init__()

        branch_channels = out_channels // 4

        # Branch 1: SE Block with 1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            SEBlock(branch_channels, reduction)
        )

        # Branch 2: CBAM with 3x3 conv
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            CBAMBlock(branch_channels, reduction)
        )

        # Branch 3: GC Block with 5x5 conv
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 5, padding=2, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            GCBlock(branch_channels, reduction)
        )

        # Branch 4: Triplet Attention with 3x3 conv
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            TripletAttention()
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        return self.fusion(out)


# ==================== Enhanced SRNet with Attention ====================

class SRNetWithAttention(nn.Module):
    """
    SRNet with SE and CBAM attention modules inserted at key feature extraction stages.
    Uses pretrained SRNet weights and adds attention for better feature refinement.
    """
    def __init__(self, pretrained_path='SRNet_model_weights.pt', freeze_backbone=False):
        super(SRNetWithAttention, self).__init__()

        # Load pretrained SRNet
        srnet = Srnet()
        if pretrained_path:
            print(f"Loading pretrained SRNet from {pretrained_path}")
            srnet.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
            print("✓ Pretrained weights loaded successfully")

        # Copy layers from pretrained SRNet
        # Early layers (1-2)
        self.layer1 = srnet.layer1
        self.bn1 = srnet.bn1
        self.layer2 = srnet.layer2
        self.bn2 = srnet.bn2

        # Residual blocks (3-7) - 16 channels
        self.layer31 = srnet.layer31
        self.bn31 = srnet.bn31
        self.layer32 = srnet.layer32
        self.bn32 = srnet.bn32

        self.layer41 = srnet.layer41
        self.bn41 = srnet.bn41
        self.layer42 = srnet.layer42
        self.bn42 = srnet.bn42

        self.layer51 = srnet.layer51
        self.bn51 = srnet.bn51
        self.layer52 = srnet.layer52
        self.bn52 = srnet.bn52

        self.layer61 = srnet.layer61
        self.bn61 = srnet.bn61
        self.layer62 = srnet.layer62
        self.bn62 = srnet.bn62

        self.layer71 = srnet.layer71
        self.bn71 = srnet.bn71
        self.layer72 = srnet.layer72
        self.bn72 = srnet.bn72

        # Downsampling layers (8-11)
        self.layer81 = srnet.layer81
        self.bn81 = srnet.bn81
        self.layer82 = srnet.layer82
        self.bn82 = srnet.bn82
        self.layer83 = srnet.layer83
        self.bn83 = srnet.bn83
        self.pool1 = srnet.pool1

        self.layer91 = srnet.layer91
        self.bn91 = srnet.bn91
        self.layer92 = srnet.layer92
        self.bn92 = srnet.bn92
        self.layer93 = srnet.layer93
        self.bn93 = srnet.bn93
        self.pool2 = srnet.pool2

        self.layer101 = srnet.layer101
        self.bn101 = srnet.bn101
        self.layer102 = srnet.layer102
        self.bn102 = srnet.bn102
        self.layer103 = srnet.layer103
        self.bn103 = srnet.bn103

        self.layer111 = srnet.layer111
        self.bn111 = srnet.bn111
        self.layer112 = srnet.layer112
        self.bn112 = srnet.bn112
        self.layer113 = srnet.layer113
        self.bn113 = srnet.bn113

        # Final layers (12)
        self.layer121 = srnet.layer121
        self.bn121 = srnet.bn121
        self.layer122 = srnet.layer122
        self.bn122 = srnet.bn122

        # Add attention modules at strategic locations
        self.se_block_16 = SEBlock(16, reduction=4)      # After layer 7 (16 channels)
        self.cbam_block_64 = CBAMBlock(64, reduction=8)   # After layer 9 (64 channels)
        self.se_block_128 = SEBlock(128, reduction=8)     # After layer 10 (128 channels)
        self.cbam_block_256 = CBAMBlock(256, reduction=16) # After layer 11 (256 channels)
        self.se_block_512 = SEBlock(512, reduction=16)    # After layer 12 (512 channels)

        # Classifier
        self.fc = nn.Linear(512, 2)

        # Optionally freeze backbone
        if freeze_backbone:
            print("Freezing SRNet backbone weights...")
            for name, param in self.named_parameters():
                if 'se_block' not in name and 'cbam_block' not in name and 'fc' not in name:
                    param.requires_grad = False
            print("✓ Backbone frozen, only attention modules and classifier will be trained")

    def forward(self, inputs):
        # Layer 1-2
        conv = self.layer1(inputs)
        actv = F.relu(self.bn1(conv))
        conv = self.layer2(actv)
        actv = F.relu(self.bn2(conv))

        # Residual blocks 3-7
        conv1 = self.layer31(actv)
        actv1 = F.relu(self.bn31(conv1))
        conv2 = self.layer32(actv1)
        bn = self.bn32(conv2)
        res = torch.add(actv, bn)

        conv1 = self.layer41(res)
        actv1 = F.relu(self.bn41(conv1))
        conv2 = self.layer42(actv1)
        bn = self.bn42(conv2)
        res = torch.add(res, bn)

        conv1 = self.layer51(res)
        actv1 = F.relu(self.bn51(conv1))
        conv2 = self.layer52(actv1)
        bn = self.bn52(conv2)
        res = torch.add(res, bn)

        conv1 = self.layer61(res)
        actv1 = F.relu(self.bn61(conv1))
        conv2 = self.layer62(actv1)
        bn = self.bn62(conv2)
        res = torch.add(res, bn)

        conv1 = self.layer71(res)
        actv1 = F.relu(self.bn71(conv1))
        conv2 = self.layer72(actv1)
        bn = self.bn72(conv2)
        res = torch.add(res, bn)

        # SE attention after residual blocks (16 channels)
        res = self.se_block_16(res)

        # Layer 8
        convs = self.layer81(res)
        convs = self.bn81(convs)
        conv1 = self.layer82(res)
        actv1 = F.relu(self.bn82(conv1))
        conv2 = self.layer83(actv1)
        bn = self.bn83(conv2)
        pool = self.pool1(bn)
        res = torch.add(convs, pool)

        # Layer 9
        convs = self.layer91(res)
        convs = self.bn91(convs)
        conv1 = self.layer92(res)
        actv1 = F.relu(self.bn92(conv1))
        conv2 = self.layer93(actv1)
        bn = self.bn93(conv2)
        pool = self.pool2(bn)
        res = torch.add(convs, pool)

        # CBAM attention after layer 9 (64 channels)
        res = self.cbam_block_64(res)

        # Layer 10
        convs = self.layer101(res)
        convs = self.bn101(convs)
        conv1 = self.layer102(res)
        actv1 = F.relu(self.bn102(conv1))
        conv2 = self.layer103(actv1)
        bn = self.bn103(conv2)
        pool = self.pool1(bn)
        res = torch.add(convs, pool)

        # SE attention after layer 10 (128 channels)
        res = self.se_block_128(res)

        # Layer 11
        convs = self.layer111(res)
        convs = self.bn111(convs)
        conv1 = self.layer112(res)
        actv1 = F.relu(self.bn112(conv1))
        conv2 = self.layer113(actv1)
        bn = self.bn113(conv2)
        pool = self.pool1(bn)
        res = torch.add(convs, pool)

        # CBAM attention after layer 11 (256 channels)
        res = self.cbam_block_256(res)

        # Layer 12
        conv1 = self.layer121(res)
        actv1 = F.relu(self.bn121(conv1))
        conv2 = self.layer122(actv1)
        bn = self.bn122(conv2)

        # SE attention after layer 12 (512 channels)
        res = self.se_block_512(bn)

        # Global pooling and classification
        avgp = torch.mean(res, dim=(2,3), keepdim=True)
        flatten = avgp.view(avgp.size(0), -1)
        fc = self.fc(flatten)

        return fc


class INATNet_v3(nn.Module):
    """Placeholder for INATNet_v3 model"""
    def __init__(self, pretrained_path='SRNet_model_weights.pt', freeze_backbone=False):
        super(INATNet_v3, self).__init__()
        # Load pretrained SRNet
        srnet = Srnet()
        if pretrained_path:
            print(f"Loading pretrained SRNet from {pretrained_path}")
            srnet.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
            print("✓ Pretrained weights loaded successfully")

        # Copy layers from pretrained SRNet
        # Early layers (1-2)
        self.layer1 = srnet.layer1
        self.bn1 = srnet.bn1
        self.layer2 = srnet.layer2
        self.bn2 = srnet.bn2

        # Residual blocks (3-7) - 16 channels
        self.layer31 = srnet.layer31
        self.bn31 = srnet.bn31
        self.layer32 = srnet.layer32
        self.bn32 = srnet.bn32

        self.layer41 = srnet.layer41
        self.bn41 = srnet.bn41
        self.layer42 = srnet.layer42
        self.bn42 = srnet.bn42

        self.layer51 = srnet.layer51
        self.bn51 = srnet.bn51
        self.layer52 = srnet.layer52
        self.bn52 = srnet.bn52

        self.layer61 = srnet.layer61
        self.bn61 = srnet.bn61
        self.layer62 = srnet.layer62
        self.bn62 = srnet.bn62

        self.layer71 = srnet.layer71
        self.bn71 = srnet.bn71
        self.layer72 = srnet.layer72
        self.bn72 = srnet.bn72

        # Downsampling layers (8-11)
        self.layer81 = srnet.layer81
        self.bn81 = srnet.bn81
        self.layer82 = srnet.layer82
        self.bn82 = srnet.bn82
        self.layer83 = srnet.layer83
        self.bn83 = srnet.bn83
        self.pool1 = srnet.pool1

        self.layer91 = srnet.layer91
        self.bn91 = srnet.bn91
        self.layer92 = srnet.layer92
        self.bn92 = srnet.bn92
        self.layer93 = srnet.layer93
        self.bn93 = srnet.bn93
        self.pool2 = srnet.pool2

        self.layer101 = srnet.layer101
        self.bn101 = srnet.bn101
        self.layer102 = srnet.layer102
        self.bn102 = srnet.bn102
        self.layer103 = srnet.layer103
        self.bn103 = srnet.bn103

        self.layer111 = srnet.layer111
        self.bn111 = srnet.bn111
        self.layer112 = srnet.layer112
        self.bn112 = srnet.bn112
        self.layer113 = srnet.layer113
        self.bn113 = srnet.bn113

        # Final layers (12)
        self.layer121 = srnet.layer121
        self.bn121 = srnet.bn121
        self.layer122 = srnet.layer122
        self.bn122 = srnet.bn122

        self.inception_attention = InceptionAttentionBlock(512, 512, reduction=16)

        # Classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

        # Optionally freeze backbone
        if freeze_backbone:
            print("Freezing SRNet backbone weights...")
            for name, param in self.named_parameters():
                if 'se_block' not in name and 'cbam_block' not in name and 'fc' not in name and 'inception_attention' not in name and 'classifier' not in name:
                    param.requires_grad = False
            print("✓ Backbone frozen, only attention modules and classifier will be trained")

    def forward(self, inputs):
        # Layer 1-2
        conv = self.layer1(inputs)
        actv = F.relu(self.bn1(conv))
        conv = self.layer2(actv)
        actv = F.relu(self.bn2(conv))

        # Residual blocks 3-7
        conv1 = self.layer31(actv)
        actv1 = F.relu(self.bn31(conv1))
        conv2 = self.layer32(actv1)
        bn = self.bn32(conv2)
        res = torch.add(actv, bn)

        conv1 = self.layer41(res)
        actv1 = F.relu(self.bn41(conv1))
        conv2 = self.layer42(actv1)
        bn = self.bn42(conv2)
        res = torch.add(res, bn)

        conv1 = self.layer51(res)
        actv1 = F.relu(self.bn51(conv1))
        conv2 = self.layer52(actv1)
        bn = self.bn52(conv2)
        res = torch.add(res, bn)

        conv1 = self.layer61(res)
        actv1 = F.relu(self.bn61(conv1))
        conv2 = self.layer62(actv1)
        bn = self.bn62(conv2)
        res = torch.add(res, bn)

        conv1 = self.layer71(res)
        actv1 = F.relu(self.bn71(conv1))
        conv2 = self.layer72(actv1)
        bn = self.bn72(conv2)
        res = torch.add(res, bn)

        # Layer 8
        convs = self.layer81(res)
        convs = self.bn81(convs)
        conv1 = self.layer82(res)
        actv1 = F.relu(self.bn82(conv1))
        conv2 = self.layer83(actv1)
        bn = self.bn83(conv2)
        pool = self.pool1(bn)
        res = torch.add(convs, pool)

        # Layer 9
        convs = self.layer91(res)
        convs = self.bn91(convs)
        conv1 = self.layer92(res)
        actv1 = F.relu(self.bn92(conv1))
        conv2 = self.layer93(actv1)
        bn = self.bn93(conv2)
        pool = self.pool2(bn)
        res = torch.add(convs, pool)

        # Layer 10
        convs = self.layer101(res)
        convs = self.bn101(convs)
        conv1 = self.layer102(res)
        actv1 = F.relu(self.bn102(conv1))
        conv2 = self.layer103(actv1)
        bn = self.bn103(conv2)
        pool = self.pool1(bn)
        res = torch.add(convs, pool)

        # Layer 11
        convs = self.layer111(res)
        convs = self.bn111(convs)
        conv1 = self.layer112(res)
        actv1 = F.relu(self.bn112(conv1))
        conv2 = self.layer113(actv1)
        bn = self.bn113(conv2)
        pool = self.pool1(bn)
        res = torch.add(convs, pool)

        # Layer 12
        conv1 = self.layer121(res)
        actv1 = F.relu(self.bn121(conv1))
        conv2 = self.layer122(actv1)
        bn = self.bn122(conv2)

        # Inception-style attention block
        inception = self.inception_attention(bn)

        # Global pooling and classification
        avgp = self.global_pool(inception)
        flatten = avgp.view(avgp.size(0), -1)
        fc = self.classifier(flatten)

        return fc


# ==================== Factory Function ====================

def get_model(model_name='srnet', pretrained_path='SRNet_model_weights.pt', freeze_backbone=False):
    """
    Factory function to get model

    Args:
        model_name: 'srnet' or 'srnet_attention'
        pretrained_path: Path to pretrained SRNet weights
        freeze_backbone: Whether to freeze SRNet backbone

    Returns:
        model: PyTorch model
    """
    if model_name == 'srnet':
        model = Srnet()
        if pretrained_path:
            devise = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.load_state_dict(torch.load(pretrained_path, map_location=devise))
            print(f"✓ Loaded pretrained SRNet from {pretrained_path}")
        return model

    elif model_name == 'srnet_attention':
        return SRNetWithAttention(pretrained_path=pretrained_path, freeze_backbone=freeze_backbone)
    elif model_name == 'inatnet_v3':
        return INATNet_v3(pretrained_path=pretrained_path, freeze_backbone=freeze_backbone)

    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == '__main__':
    # Test the model
    print("="*70)
    print("Testing Srnet")
    # print("Testing SRNetWithAttention")
    # print("Testing INATNet_v3")
    print("="*70)

    # Create model (without pretrained weights for testing)
    model = Srnet()
    # model = SRNetWithAttention(pretrained_path=None, freeze_backbone=False)
    # model = INATNet_v3(pretrained_path=None, freeze_backbone=False)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # attention_params = sum(p.numel() for n, p in model.named_parameters()
    #                       if 'se_block' in n or 'cbam_block' in n)
    # attention_params = sum(p.numel() for n, p in model.named_parameters()
    #                       if 'inception_attention' in n)


    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    # print(f"Attention module parameters: {attention_params:,}")
    # print(f"Attention overhead: {100*attention_params/total_params:.2f}%")

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 1, 512, 512)  # Grayscale 512x512

    print(f"\nInput shape: {x.shape}")

    model.eval()
    with torch.no_grad():
        output = model(x)

    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    print("\n" + "="*70)
    print("✓ Model test successful!")
    print("="*70)
