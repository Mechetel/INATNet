import os
import json
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional
import matplotlib.pyplot as plt


# ==================== Logger System ====================

class Logger:
    """Custom logger for training process"""

    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(experiment_name)

        # Metrics storage
        self.metrics_history = defaultdict(list)

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log metrics for an epoch"""
        self.metrics_history['epoch'].append(epoch)
        for key, value in metrics.items():
            self.metrics_history[key].append(value)

        # Format message
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.info(f"Epoch {epoch} - {metrics_str}")

    def save_metrics(self):
        """Save metrics history to file"""
        metrics_file = self.log_dir / f"{self.experiment_name}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(dict(self.metrics_history), f, indent=4)

    def plot_metrics(self, save_path: Optional[str] = None):
        """Plot training metrics"""
        if not self.metrics_history:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Metrics - {self.experiment_name}', fontsize=16)

        epochs = self.metrics_history['epoch']

        # Loss plot
        if 'train_loss' in self.metrics_history:
            axes[0, 0].plot(epochs, self.metrics_history['train_loss'], label='Train Loss')
        if 'val_loss' in self.metrics_history:
            axes[0, 0].plot(epochs, self.metrics_history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Accuracy plot
        if 'train_acc' in self.metrics_history:
            axes[0, 1].plot(epochs, self.metrics_history['train_acc'], label='Train Acc')
        if 'val_acc' in self.metrics_history:
            axes[0, 1].plot(epochs, self.metrics_history['val_acc'], label='Val Acc')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Accuracy over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Learning rate plot
        if 'learning_rate' in self.metrics_history:
            axes[1, 0].plot(epochs, self.metrics_history['learning_rate'])
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)

        # F1 Score plot
        if 'val_f1' in self.metrics_history:
            axes[1, 1].plot(epochs, self.metrics_history['val_f1'], label='Val F1')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('F1 Score')
            axes[1, 1].set_title('F1 Score over Time')
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = self.log_dir / f"{self.experiment_name}_metrics.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close()
        self.info(f"Metrics plot saved to {save_path}")

