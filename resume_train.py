from app.config import Config
from app.inat_net import INATNet
from app.trainer import Trainer
from app.data import DataManager

"""
Resume training from a saved checkpoint
"""

# Load config from saved file
config = Config.load('configs/experiment_config.json')

# Update epochs to continue training
config.num_epochs = 1  # Train for 1 more epoch

# Initialize model
model = INATNet(num_classes=2)

# Create trainer
trainer = Trainer(model, config, experiment_name="resumed_training")

# Load checkpoint
checkpoint_path = "checkpoints/advanced_exp/checkpoint_epoch_50.pth"
trainer.load_checkpoint(checkpoint_path)

# Setup data
data_manager = DataManager(config)
train_loader, val_loader = data_manager.create_dataloaders(
    train_cover_dir="datasets/train/cover",
    train_stego_dir="datasets/train/stego",
    val_cover_dir="datasets/val/cover",
    val_stego_dir="datasets/val/stego"
)

# Continue training
trainer.fit(train_loader, val_loader)

print("✓ Resumed training completed!")