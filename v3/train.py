import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import argparse
import os
from model import get_model
from utils import get_data_loaders, calculate_metrics, save_checkpoint


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass - outputs are logits [batch_size, 2]
        outputs = model(images)

        # Convert labels to long for CrossEntropyLoss
        loss = criterion(outputs, labels.long())

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Get predictions
        preds = torch.argmax(outputs, dim=1).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{(preds == labels).float().mean().item():.4f}'
        })

    epoch_loss = running_loss / len(train_loader.dataset)
    metrics = calculate_metrics(all_labels, all_preds)

    return epoch_loss, metrics


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels.long())

            running_loss += loss.item() * images.size(0)

            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(val_loader.dataset)
    metrics = calculate_metrics(all_labels, all_preds)

    return epoch_loss, metrics, all_labels, all_probs


def train(args):
    """Main training function"""

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Model: {args.model}")
    print(f"Image size: {args.img_size}")

    # Create data loaders
    print("\n" + "="*70)
    print("Loading datasets...")
    print("="*70)
    print(f"Cover train path: {args.cover_train_path}")
    print(f"Stego train path: {args.stego_train_path}")
    print(f"Cover val path:   {args.cover_val_path}")
    print(f"Stego val path:   {args.stego_val_path}")

    train_loader, val_loader = get_data_loaders(
        args.cover_train_path,
        args.stego_train_path,
        args.cover_val_path,
        args.stego_val_path,
        args.batch_size,
        args.img_size,
        augment=args.augment,
        num_workers=args.num_workers
    )

    print(f"\nTrain samples: {len(train_loader.dataset)}")
    if val_loader:
        print(f"Validation samples: {len(val_loader.dataset)}")

    # Initialize model
    print("\n" + "="*70)
    print("Initializing model...")
    print("="*70)

    model = get_model(
        model_name=args.model,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Model already has sigmoid
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0

    print("\n" + "="*70)
    print("Starting training...")
    print("="*70 + "\n")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 70)

        # Train
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1
        )

        # Print training metrics
        print(f"\n[Train] Loss: {train_loss:.4f} | "
              f"Acc: {train_metrics['accuracy']:.4f} | "
              f"Precision: {train_metrics['precision']:.4f} | "
              f"Recall: {train_metrics['recall']:.4f} | "
              f"F1: {train_metrics['f1_score']:.4f}")

        # Validate if validation loader exists
        if val_loader:
            val_loss, val_metrics, _, _ = validate(
                model, val_loader, criterion, device
            )

            # Learning rate scheduler step
            scheduler.step(val_loss)

            print(f"[Val]   Loss: {val_loss:.4f} | "
                  f"Acc: {val_metrics['accuracy']:.4f} | "
                  f"Precision: {val_metrics['precision']:.4f} | "
                  f"Recall: {val_metrics['recall']:.4f} | "
                  f"F1: {val_metrics['f1_score']:.4f}")

            # Save best model based on validation accuracy
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_val_loss = val_loss
                model_path = os.path.join(args.output_dir, f'best_model_{args.model}.pth')
                save_checkpoint(model, optimizer, epoch, val_loss, model_path)
                print(f"✓ Best model saved! (Val Acc: {best_val_acc:.4f}, Val Loss: {best_val_loss:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= args.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        else:
            # If no validation set, save based on training loss
            if train_loss < best_val_loss:
                best_val_loss = train_loss
                model_path = os.path.join(args.output_dir, f'best_model_{args.model}.pth')
                save_checkpoint(model, optimizer, epoch, train_loss, model_path)
                print(f"✓ Best model saved! (Train Loss: {best_val_loss:.4f})")

    print("\n" + "="*70)
    print("Training completed!")
    if val_loader:
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        print(f"Best validation loss: {best_val_loss:.4f}")
    else:
        print(f"Best training loss: {best_val_loss:.4f}")
    print("="*70)

    # Save final model
    final_model_path = os.path.join(args.output_dir, f'final_model_{args.model}.pth')
    save_checkpoint(model, optimizer, args.epochs, train_loss, final_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Steganalysis Classifier')

    # Data paths - BOWS2 WOW 0.2bpp
    parser.add_argument(
        "--cover_train_path",
        type=str,
        default="/Users/dmitryhoma/Projects/datasets/ready_to_use/GBRASNET/BOWS2/cover/train",
        help="Path to training cover images"
    )
    parser.add_argument(
        "--stego_train_path",
        type=str,
        default="/Users/dmitryhoma/Projects/datasets/ready_to_use/GBRASNET/BOWS2/stego/WOW/0.2bpp/train",
        help="Path to training stego images"
    )
    parser.add_argument(
        "--cover_val_path",
        type=str,
        default="/Users/dmitryhoma/Projects/datasets/ready_to_use/GBRASNET/BOWS2/cover/val",
        help="Path to validation cover images"
    )
    parser.add_argument(
        "--stego_val_path",
        type=str,
        default="/Users/dmitryhoma/Projects/datasets/ready_to_use/GBRASNET/BOWS2/stego/WOW/0.2bpp/val",
        help="Path to validation stego images"
    )

    # Model parameters
    parser.add_argument(
        "--model",
        type=str,
        default="srnet",
        choices=["srnet", "srnet_attention", "inatnet_v3"],
        help="Model architecture to use"
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Use pretrained weights"
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        default=False,
        help="Freeze backbone weights (only train classifier)"
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--img_size", type=int, default=256, help="Image size")
    parser.add_argument(
        "--augment",
        action="store_true",
        default=True,
        help="Use data augmentation"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=50,
        help="Early stopping patience"
    )

    # System parameters
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints_steganalysis",
        help="Output directory for checkpoints"
    )

    args = parser.parse_args()

    train(args)
