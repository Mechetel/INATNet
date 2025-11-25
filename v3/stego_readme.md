# Steganalysis Classification System

This project provides a complete pipeline for training and testing steganalysis models to detect steganographic content in images (distinguishing between cover and stego images).

## Dataset Structure

The code is configured to work with the GBRASNET dataset, specifically BOWS2 with WOW algorithm at 0.2bpp payload:

```
GBRASNET/
└── BOWS2/
    ├── cover/
    │   ├── train/  [8000 images]
    │   ├── val/    [1000 images]
    │   └── test/   [1000 images]
    └── stego/
        └── WOW/
            └── 0.2bpp/
                ├── train/  [8000 images]
                ├── val/    [1000 images]
                └── test/   [1000 images]
```

## Files Description

1. **model_stego.py** - Contains three model architectures:
   - `SimpleCNN` - Custom lightweight CNN
   - `ResNet50Classifier` - Transfer learning with ResNet-50
   - `EfficientNetClassifier` - Transfer learning with EfficientNet-B0

2. **utils_stego.py** - Utility functions:
   - `SteganalysisDataset` - Custom dataset loader for cover/stego pairs
   - Data transformation and augmentation
   - Metrics calculation
   - Checkpoint saving/loading

3. **train_stego.py** - Training script with:
   - Training loop with validation
   - Early stopping
   - Learning rate scheduling
   - Model checkpointing

4. **test_stego.py** - Testing script that generates:
   - ROC curve and AUC score
   - Confusion matrix
   - Classification metrics
   - Probability distributions

## Quick Start

### 1. Training

Basic training with default parameters (ResNet-50):

```bash
python train_stego.py
```

Training with custom parameters:

```bash
python train_stego.py \
    --model resnet50 \
    --epochs 30 \
    --batch_size 32 \
    --lr 0.001 \
    --img_size 256 \
    --output_dir ./checkpoints_steganalysis
```

Training different models:

```bash
# Simple CNN
python train_stego.py --model simple --epochs 50

# EfficientNet
python train_stego.py --model efficientnet --epochs 30

# ResNet-50 with frozen backbone (fine-tuning)
python train_stego.py --model resnet50 --freeze_backbone
```

### 2. Testing

Test the trained model:

```bash
python test_stego.py
```

Test with custom parameters:

```bash
python test_stego.py \
    --model resnet50 \
    --model_path ./checkpoints_steganalysis/best_model_resnet50.pth \
    --batch_size 32 \
    --show_plot
```

## Command Line Arguments

### Training Arguments

**Data paths:**
- `--cover_train_path` - Path to training cover images
- `--stego_train_path` - Path to training stego images
- `--cover_val_path` - Path to validation cover images
- `--stego_val_path` - Path to validation stego images

**Model parameters:**
- `--model` - Model architecture: `simple`, `resnet50`, `efficientnet` (default: `resnet50`)
- `--pretrained` - Use pretrained weights (default: `True`)
- `--freeze_backbone` - Freeze backbone weights (default: `False`)

**Training hyperparameters:**
- `--epochs` - Number of training epochs (default: `30`)
- `--batch_size` - Batch size (default: `32`)
- `--lr` - Learning rate (default: `0.001`)
- `--weight_decay` - Weight decay for regularization (default: `1e-4`)
- `--img_size` - Image size for resizing (default: `256`)
- `--augment` - Use data augmentation (default: `True`)
- `--early_stopping_patience` - Early stopping patience (default: `10`)

**System parameters:**
- `--num_workers` - Number of data loading workers (default: `4`)
- `--output_dir` - Output directory for checkpoints (default: `./checkpoints_steganalysis`)

### Testing Arguments

**Data paths:**
- `--cover_test_path` - Path to test cover images
- `--stego_test_path` - Path to test stego images

**Model parameters:**
- `--model` - Model architecture to test
- `--model_path` - Path to trained model checkpoint

**Testing parameters:**
- `--batch_size` - Batch size (default: `32`)
- `--img_size` - Image size (default: `256`)
- `--num_workers` - Number of workers (default: `4`)
- `--output_dir` - Output directory for results (default: `./test_results_steganalysis`)
- `--show_plot` - Display plots after testing

## Model Architectures

### SimpleCNN
- Lightweight custom architecture
- ~500K parameters
- 4 convolutional blocks
- Good for quick experiments

### ResNet50 (Recommended)
- Transfer learning from ImageNet
- ~24M parameters
- Best accuracy for steganalysis
- Supports backbone freezing

### EfficientNet-B0
- Efficient architecture
- ~4M parameters
- Good balance of speed and accuracy

## Output Files

### Training Outputs
- `best_model_{model_name}.pth` - Best model based on validation accuracy
- `final_model_{model_name}.pth` - Model after all epochs

### Testing Outputs
- `test_results_steganalysis.png` - Visualization with ROC curve, confusion matrix, and probability distribution
- `test_results_steganalysis.txt` - Detailed metrics and results

## Performance Metrics

The system reports:
- **Accuracy** - Overall classification accuracy
- **Precision** - Correct stego predictions / Total stego predictions
- **Recall** - Correct stego predictions / Actual stego images
- **F1 Score** - Harmonic mean of precision and recall
- **AUC** - Area Under ROC Curve (0.5 = random, 1.0 = perfect)

## Tips for Best Results

1. **Start with ResNet-50** - Best overall performance
2. **Use pretrained weights** - Faster convergence
3. **Monitor validation metrics** - Watch for overfitting
4. **Adjust learning rate** - Try 0.0001 for fine-tuning
5. **Use data augmentation** - Improves generalization
6. **Check AUC score** - Better metric than accuracy for imbalanced data

## Example Training Session

```bash
# Train ResNet-50 with default settings
python train_stego.py

# Expected output:
# Using device: cuda
# Train samples: 16000
# Validation samples: 2000
# Trainable parameters: 24,064,577
#
# Epoch 1/30: Loss: 0.3245, Acc: 0.8523
# [Val] Acc: 0.8712
# ✓ Best model saved!
```

## Example Testing Session

```bash
# Test the trained model
python test_stego.py --show_plot

# Expected output:
# Test samples: 2000
# Accuracy: 0.8856 (88.56%)
# AUC: 0.9423 (94.23%)
# ✓ Results saved!
```

## Troubleshooting

**CUDA out of memory:**
- Reduce `--batch_size` (try 16 or 8)
- Reduce `--img_size` (try 128)

**Low accuracy:**
- Check data paths are correct
- Ensure cover and stego images match
- Try different learning rates
- Increase epochs

**Training too slow:**
- Reduce `--num_workers` on systems with limited CPU
- Use `--freeze_backbone` for faster fine-tuning
- Try SimpleCNN for quick experiments

## Requirements

```
torch>=1.9.0
torchvision>=0.10.0
numpy
matplotlib
seaborn
scikit-learn
Pillow
tqdm
```

Install with:
```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn Pillow tqdm
```
