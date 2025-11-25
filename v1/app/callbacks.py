import torch
import torch.optim as optim
from pathlib import Path
import time

# ==================== Callbacks System ====================

class Callback:
    """Base callback class"""
    
    def on_train_begin(self, trainer):
        pass
    
    def on_train_end(self, trainer):
        pass
    
    def on_epoch_begin(self, epoch, trainer):
        pass
    
    def on_epoch_end(self, epoch, metrics, trainer):
        pass
    
    def on_batch_begin(self, batch_idx, trainer):
        pass
    
    def on_batch_end(self, batch_idx, loss, trainer):
        pass


class CustomProgressCallback(Callback):
    """Custom callback to log additional information"""
    
    def __init__(self):
        self.start_time = None
    
    def on_train_begin(self, trainer):
        self.start_time = time.time()
        trainer.logger.info("🚀 Training started!")
    
    def on_epoch_end(self, epoch, metrics, trainer):
        elapsed = time.time() - self.start_time
        trainer.logger.info(f"⏱️  Elapsed time: {elapsed/60:.1f} minutes")
    
    def on_train_end(self, trainer):
        total_time = time.time() - self.start_time
        trainer.logger.info(f"✅ Total training time: {total_time/3600:.2f} hours")


class EarlyStopping(Callback):
    """Early stopping callback"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, monitor: str = 'val_loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_value = None
        self.counter = 0
        self.should_stop = False
    
    def on_epoch_end(self, epoch, metrics, trainer):
        current_value = metrics.get(self.monitor)
        if current_value is None:
            return
        
        # For loss, lower is better
        if self.monitor.endswith('loss'):
            improved = (self.best_value is None or 
                       current_value < self.best_value - self.min_delta)
        # For accuracy/f1, higher is better
        else:
            improved = (self.best_value is None or 
                       current_value > self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.counter = 0
            trainer.logger.info(f"✓ {self.monitor} improved to {current_value:.4f}")
        else:
            self.counter += 1
            trainer.logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.should_stop = True
                trainer.logger.info(f"Early stopping triggered!")


class ModelCheckpoint(Callback):
    """Model checkpoint callback"""
    
    def __init__(self, checkpoint_dir: str, save_best_only: bool = True, 
                 monitor: str = 'val_acc', save_frequency: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.save_frequency = save_frequency
        self.best_value = None
    
    def on_epoch_end(self, epoch, metrics, trainer):
        current_value = metrics.get(self.monitor)
        
        # Save best model
        if self.save_best_only and current_value is not None:
            # For loss, lower is better
            if self.monitor.endswith('loss'):
                is_best = (self.best_value is None or current_value < self.best_value)
            # For accuracy/f1, higher is better
            else:
                is_best = (self.best_value is None or current_value > self.best_value)
            
            if is_best:
                self.best_value = current_value
                filepath = self.checkpoint_dir / "best_model.pth"
                trainer.save_checkpoint(filepath, epoch, metrics)
                trainer.logger.info(f"✓ Best model saved with {self.monitor}={current_value:.4f}")
        
        # Save periodic checkpoint
        if epoch % self.save_frequency == 0:
            filepath = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            trainer.save_checkpoint(filepath, epoch, metrics)


class LearningRateScheduler(Callback):
    """Learning rate scheduler callback"""
    
    def __init__(self, scheduler):
        self.scheduler = scheduler
    
    def on_epoch_end(self, epoch, metrics, trainer):
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            val_loss = metrics.get('val_loss')
            if val_loss is not None:
                self.scheduler.step(val_loss)
        else:
            self.scheduler.step()
        
        current_lr = trainer.optimizer.param_groups[0]['lr']
        trainer.logger.info(f"Learning rate: {current_lr:.2e}")

