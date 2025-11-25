import json
import torch
from typing import Optional, Dict


# ==================== Configuration Management ====================

class Config:
    """Configuration manager for training and inference"""

    def __init__(self, config_dict: Optional[Dict] = None):
        # Model config
        self.num_classes = 2
        self.model_name = "INATNet"

        # Training config
        self.batch_size = 32
        self.num_epochs = 100
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        self.optimizer = "adam"

        # Data config
        self.image_size = 256
        self.num_workers = 4
        self.pin_memory = True

        # Augmentation config
        self.use_augmentation = True
        self.random_flip = True
        self.random_crop = True
        self.rotation_degrees = 15

        # Scheduler config
        self.use_scheduler = True
        self.scheduler_type = "reduce_on_plateau"
        self.scheduler_patience = 5
        self.scheduler_factor = 0.5
        self.scheduler_min_lr = 1e-7

        # Early stopping config
        self.early_stopping = True
        self.early_stopping_patience = 10
        self.early_stopping_min_delta = 0.001

        # Checkpoint config
        self.save_best_only = True
        self.save_frequency = 5
        self.checkpoint_dir = "checkpoints"

        self.custom_progress_callback = False

        # Logging config
        self.log_dir = "logs"
        self.tensorboard = False
        self.log_frequency = 10

        # Device config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mixed_precision = True

        # Override with custom config
        if config_dict:
            for key, value in config_dict.items():
                setattr(self, key, value)

    def save(self, filepath: str):
        """Save configuration to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
        print(f"Config saved to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Load configuration from JSON"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)

    def __repr__(self):
        return json.dumps(self.__dict__, indent=2)
