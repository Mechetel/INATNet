import os
import time
from typing import Any, Dict, Optional, List, Tuple
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from app.config import Config
from app.logger import Logger
from app.callbacks import CustomProgressCallback, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from app.metrics_calculator import MetricsCalculator


# ==================== Trainer ====================

class Trainer:
    """Main trainer class"""
    
    def __init__(self, model: nn.Module, config: Config, experiment_name: str = "steganalysis"):
        self.model = model
        self.config = config
        self.experiment_name = experiment_name
        
        # Setup device
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Setup logger
        self.logger = Logger(config.log_dir, experiment_name)
        self.logger.info(f"Initialized trainer for {experiment_name}")
        self.logger.info(f"Using device: {self.device}")
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler() if config.use_scheduler else None
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup mixed precision
        self.scaler = torch.amp.GradScaler() if config.mixed_precision and config.device == "cuda" else None
        
        # Setup callbacks
        self.callbacks = []
        self._setup_callbacks()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metrics = {}
    
    def _create_optimizer(self):
        """Create optimizer"""
        if self.config.optimizer.lower() == "adam":
            return optim.Adam(self.model.parameters(), 
                            lr=self.config.learning_rate,
                            weight_decay=self.config.weight_decay)
        elif self.config.optimizer.lower() == "sgd":
            return optim.SGD(self.model.parameters(),
                           lr=self.config.learning_rate,
                           momentum=0.9,
                           weight_decay=self.config.weight_decay)
        elif self.config.optimizer.lower() == "adamw":
            return optim.AdamW(self.model.parameters(),
                             lr=self.config.learning_rate,
                             weight_decay=self.config.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.config.scheduler_type == "reduce_on_plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience,
                min_lr=self.config.scheduler_min_lr,
                verbose=True
            )
        elif self.config.scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.scheduler_min_lr
            )
        elif self.config.scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        return None
    
    def _setup_callbacks(self):
        """Setup callbacks"""
        # Early stopping
        if self.config.early_stopping:
            self.callbacks.append(
                EarlyStopping(
                    patience=self.config.early_stopping_patience,
                    min_delta=self.config.early_stopping_min_delta,
                    monitor='val_loss'
                )
            )
        
        # Model checkpoint
        self.callbacks.append(
            ModelCheckpoint(
                checkpoint_dir=self.config.checkpoint_dir,
                save_best_only=self.config.save_best_only,
                monitor='val_acc',
                save_frequency=self.config.save_frequency
            )
        )
        if self.config.custom_progress_callback:
            self.callbacks.append(CustomProgressCallback())

        # Learning rate scheduler
        if self.scheduler:
            self.callbacks.append(LearningRateScheduler(self.scheduler))

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Callbacks
            for callback in self.callbacks:
                callback.on_batch_begin(batch_idx, self)
            
            # Forward pass with mixed precision
            if self.scaler:
                with torch.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Callbacks
            for callback in self.callbacks:
                callback.on_batch_end(batch_idx, loss.item(), self)
            
            self.global_step += 1
        
        # Calculate metrics
        epoch_loss = running_loss / len(train_loader)
        all_predictions = torch.tensor(all_predictions)
        all_targets = torch.tensor(all_targets)
        metrics = MetricsCalculator.compute_all_metrics(all_predictions, all_targets)
        metrics['train_loss'] = epoch_loss
        metrics['train_acc'] = metrics.pop('accuracy')
        
        return metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        
        # Calculate metrics
        val_loss = running_loss / len(val_loader)
        all_predictions = torch.tensor(all_predictions)
        all_targets = torch.tensor(all_targets)
        metrics = MetricsCalculator.compute_all_metrics(all_predictions, all_targets)
        metrics['val_loss'] = val_loss
        metrics['val_acc'] = metrics.pop('accuracy')
        metrics['val_precision'] = metrics.pop('precision')
        metrics['val_recall'] = metrics.pop('recall')
        metrics['val_f1'] = metrics.pop('f1')
        
        return metrics
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop"""
        self.logger.info("="*60)
        self.logger.info("Starting training")
        self.logger.info("="*60)
        self.logger.info(f"Total epochs: {self.config.num_epochs}")
        self.logger.info(f"Batch size: {self.config.batch_size}")
        self.logger.info(f"Learning rate: {self.config.learning_rate}")
        
        # Callbacks
        for callback in self.callbacks:
            callback.on_train_begin(self)
        
        try:
            for epoch in range(1, self.config.num_epochs + 1):
                self.current_epoch = epoch
                
                # Callbacks
                for callback in self.callbacks:
                    callback.on_epoch_begin(epoch, self)
                
                # Train
                train_metrics = self.train_epoch(train_loader)
                
                # Validate
                val_metrics = self.validate(val_loader)
                
                # Combine metrics
                all_metrics = {**train_metrics, **val_metrics}
                all_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
                
                # Log metrics
                self.logger.log_metrics(epoch, all_metrics)
                
                # Callbacks
                for callback in self.callbacks:
                    callback.on_epoch_end(epoch, all_metrics, self)
                
                # Check early stopping
                early_stopping_callback = next((cb for cb in self.callbacks if isinstance(cb, EarlyStopping)), None)
                if early_stopping_callback and early_stopping_callback.should_stop:
                    self.logger.info("Early stopping triggered. Training stopped.")
                    break
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        
        finally:
            # Callbacks
            for callback in self.callbacks:
                callback.on_train_end(self)
            
            # Save final results
            self.logger.save_metrics()
            self.logger.plot_metrics()
            
            self.logger.info("="*60)
            self.logger.info("Training complete!")
            self.logger.info("="*60)
    
    def save_checkpoint(self, filepath: str, epoch: int, metrics: Dict[str, float]):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config.__dict__,
            'global_step': self.global_step
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint.get('global_step', 0)
        
        self.logger.info(f"Checkpoint loaded from {filepath}")
        self.logger.info(f"Resumed from epoch {self.current_epoch}")
        
        return checkpoint.get('metrics', {})