"""
Complete Usage Guide for Steganalysis Network Ecosystem
This file contains practical, ready-to-use examples
"""

import torch
import numpy as np
from pathlib import Path

# Assuming all ecosystem components are imported
# from steganalysis_ecosystem import *
# from steganalysis_net import INATNet


# ==================== SCENARIO 1: Quick Start Training ====================

def quick_start_training():
    """
    Simplest way to start training
    """
    print("\n" + "="*70)
    print("SCENARIO 1: Quick Start Training")
    print("="*70)
    
    # Step 1: Create config with default settings
    config = Config({
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'image_size': 256,
    })
    
    # Step 2: Initialize model
    model = INATNet(num_classes=2)
    
    # Step 3: Create trainer
    trainer = Trainer(model, config, experiment_name="quick_start_exp")
    
    # Step 4: Setup data
    data_manager = DataManager(config)
    train_loader, val_loader = data_manager.create_dataloaders(
        train_cover_dir="datasets/train/cover",
        train_stego_dir="datasets/train/stego",
        val_cover_dir="datasets/val/cover",
        val_stego_dir="datasets/val/stego"
    )
    
    # Step 5: Train!
    trainer.fit(train_loader, val_loader)
    
    print("✓ Training completed! Check 'logs/' and 'checkpoints/' directories")


# ==================== SCENARIO 2: Advanced Training with Custom Settings ====================

def advanced_training():
    """
    Advanced training with custom configurations
    """
    print("\n" + "="*70)
    print("SCENARIO 2: Advanced Training with Custom Settings")
    print("="*70)
    
    # Create advanced configuration
    config = Config({
        # Model settings
        'num_classes': 2,
        
        # Training settings
        'batch_size': 16,
        'num_epochs': 100,
        'learning_rate': 0.0005,
        'weight_decay': 1e-4,
        'optimizer': 'adamw',
        
        # Data settings
        'image_size': 512,  # Higher resolution
        'num_workers': 8,
        
        # Augmentation settings
        'use_augmentation': True,
        'random_flip': True,
        'random_crop': True,
        'rotation_degrees': 15,
        
        # Scheduler settings
        'use_scheduler': True,
        'scheduler_type': 'cosine',
        'scheduler_min_lr': 1e-7,
        
        # Early stopping
        'early_stopping': True,
        'early_stopping_patience': 15,
        'early_stopping_min_delta': 0.0005,
        
        # Checkpointing
        'save_best_only': False,  # Save all checkpoints
        'save_frequency': 5,
        'checkpoint_dir': 'checkpoints/advanced_exp',
        
        # Logging
        'log_dir': 'logs/advanced_exp',
        
        # Performance
        'device': 'cuda',
        'mixed_precision': True,
    })
    
    # Save config for reproducibility
    config.save('configs/advanced_training_config.json')
    
    # Initialize model
    model = INATNet(num_classes=2)
    
    # Create trainer
    trainer = Trainer(model, config, experiment_name="advanced_training")
    
    # Setup data
    data_manager = DataManager(config)
    train_loader, val_loader = data_manager.create_dataloaders(
        train_cover_dir="datasets/train/cover",
        train_stego_dir="datasets/train/stego",
        val_cover_dir="datasets/val/cover",
        val_stego_dir="datasets/val/stego"
    )
    
    # Train
    trainer.fit(train_loader, val_loader)
    
    print("✓ Advanced training completed!")


# ==================== SCENARIO 3: Resume Training from Checkpoint ====================

def resume_training():
    """
    Resume training from a saved checkpoint
    """
    print("\n" + "="*70)
    print("SCENARIO 3: Resume Training from Checkpoint")
    print("="*70)
    
    # Load config from saved file
    config = Config.load('configs/advanced_training_config.json')
    
    # Update epochs to continue training
    config.num_epochs = 150  # Train for 50 more epochs
    
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


# ==================== SCENARIO 4: Inference on Single Image ====================

def single_image_inference():
    """
    Run inference on a single image
    """
    print("\n" + "="*70)
    print("SCENARIO 4: Single Image Inference")
    print("="*70)
    
    # Create config
    config = Config({'image_size': 256, 'device': 'cuda'})
    
    # Load model
    model = INATNet(num_classes=2)
    model_manager = ModelManager()
    model_manager.load_model(model, "checkpoints/best_model.pth")
    
    # Create inference engine
    inference_engine = InferenceEngine(model, config)
    
    # Predict on single image
    image_path = "test_images/suspicious_image.jpg"
    prediction, probabilities = inference_engine.predict_single(image_path)
    
    print(f"\nImage: {image_path}")
    print(f"Prediction: {'STEGO (Hidden message detected!)' if prediction == 1 else 'COVER (Clean image)'}")
    print(f"Confidence: {probabilities[prediction]:.2%}")
    print(f"Probabilities: Cover={probabilities[0]:.2%}, Stego={probabilities[1]:.2%}")
    
    return prediction, probabilities


# ==================== SCENARIO 5: Batch Inference on Multiple Images ====================

def batch_inference():
    """
    Run inference on multiple images efficiently
    """
    print("\n" + "="*70)
    print("SCENARIO 5: Batch Inference on Multiple Images")
    print("="*70)
    
    # Setup
    config = Config({'image_size': 256, 'device': 'cuda'})
    model = INATNet(num_classes=2)
    model_manager = ModelManager()
    model_manager.load_model(model, "checkpoints/best_model.pth")
    
    inference_engine = InferenceEngine(model, config)
    
    # Get all images from directory
    test_dir = Path("test_images")
    image_paths = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    
    print(f"\nProcessing {len(image_paths)} images...")
    
    # Batch prediction
    predictions, probabilities = inference_engine.predict_batch([str(p) for p in image_paths])
    
    # Display results
    print("\nResults:")
    print("-" * 70)
    for i, (path, pred, prob) in enumerate(zip(image_paths, predictions, probabilities)):
        status = "STEGO" if pred == 1 else "COVER"
        confidence = prob[pred]
        print(f"{i+1}. {path.name:30s} -> {status:6s} (confidence: {confidence:.2%})")
    
    # Summary statistics
    num_stego = np.sum(predictions == 1)
    num_cover = np.sum(predictions == 0)
    print("-" * 70)
    print(f"Summary: {num_cover} cover images, {num_stego} stego images detected")
    
    return predictions, probabilities


# ==================== SCENARIO 6: Evaluate Model on Test Set ====================

def evaluate_test_set():
    """
    Comprehensive evaluation on test set with metrics
    """
    print("\n" + "="*70)
    print("SCENARIO 6: Evaluate Model on Test Set")
    print("="*70)
    
    # Setup
    config = Config({'batch_size': 32, 'image_size': 256, 'device': 'cuda'})
    model = INATNet(num_classes=2)
    
    # Load best model
    trainer = Trainer(model, config, experiment_name="evaluation")
    trainer.load_checkpoint("checkpoints/best_model.pth")
    
    # Create test dataloader
    data_manager = DataManager(config)
    from torch.utils.data import DataLoader
    test_dataset = SteganalysisDataset(
        cover_dir="datasets/test/cover",
        stego_dir="datasets/test/stego",
        transform=data_manager.transform_val
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Evaluate
    test_metrics = trainer.validate(test_loader)
    
    # Display results
    print("\nTest Set Results:")
    print("-" * 70)
    print(f"Accuracy:  {test_metrics['val_acc']:.2f}%")
    print(f"Precision: {test_metrics['val_precision']:.2f}%")
    print(f"Recall:    {test_metrics['val_recall']:.2f}%")
    print(f"F1 Score:  {test_metrics['val_f1']:.2f}%")
    print(f"Loss:      {test_metrics['val_loss']:.4f}")
    print("-" * 70)
    
    return test_metrics


# ==================== SCENARIO 7: Multi-Size Training (Progressive) ====================

def progressive_size_training():
    """
    Train progressively on increasing image sizes
    """
    print("\n" + "="*70)
    print("SCENARIO 7: Progressive Multi-Size Training")
    print("="*70)
    
    # Start with smaller images
    sizes = [128, 256, 512]
    
    model = INATNet(num_classes=2)
    
    for size in sizes:
        print(f"\n{'='*70}")
        print(f"Training phase: {size}x{size}")
        print(f"{'='*70}")
        
        config = Config({
            'batch_size': 64 // (size // 128),  # Adjust batch size
            'num_epochs': 30,
            'learning_rate': 0.001 / (size / 128),  # Adjust LR
            'image_size': size,
            'checkpoint_dir': f'checkpoints/size_{size}',
            'log_dir': f'logs/size_{size}',
        })
        
        trainer = Trainer(model, config, experiment_name=f"progressive_size_{size}")
        
        data_manager = DataManager(config)
        train_loader, val_loader = data_manager.create_dataloaders(
            train_cover_dir="datasets/train/cover",
            train_stego_dir="datasets/train/stego",
            val_cover_dir="datasets/val/cover",
            val_stego_dir="datasets/val/stego"
        )
        
        trainer.fit(train_loader, val_loader)
    
    print("\n✓ Progressive training completed!")


# ==================== SCENARIO 8: Custom Callbacks ====================

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


def training_with_custom_callbacks():
    """
    Training with custom callbacks
    """
    print("\n" + "="*70)
    print("SCENARIO 8: Training with Custom Callbacks")
    print("="*70)
    
    config = Config({'batch_size': 32, 'num_epochs': 50})
    model = INATNet(num_classes=2)
    
    trainer = Trainer(model, config, experiment_name="custom_callbacks")
    
    # Add custom callback
    trainer.callbacks.append(CustomProgressCallback())
    
    # Setup data
    data_manager = DataManager(config)
    train_loader, val_loader = data_manager.create_dataloaders(
        train_cover_dir="datasets/train/cover",
        train_stego_dir="datasets/train/stego",
        val_cover_dir="datasets/val/cover",
        val_stego_dir="datasets/val/stego"
    )
    
    trainer.fit(train_loader, val_loader)


# ==================== SCENARIO 9: Experiment Comparison ====================

def compare_experiments():
    """
    Compare multiple experiments
    """
    print("\n" + "="*70)
    print("SCENARIO 9: Experiment Comparison")
    print("="*70)
    
    experiments = [
        ("experiment_001", "logs/experiment_001/experiment_001_metrics.json"),
        ("experiment_002", "logs/experiment_002/experiment_002_metrics.json"),
        ("experiment_003", "logs/experiment_003/experiment_003_metrics.json"),
    ]
    
    print("\nComparison of Experiments:")
    print("-" * 70)
    print(f"{'Experiment':<20} {'Best Val Acc':<15} {'Best F1':<15} {'Final Loss':<15}")
    print("-" * 70)
    
    for exp_name, metrics_file in experiments:
        if Path(metrics_file).exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            best_acc = max(metrics.get('val_acc', [0]))
            best_f1 = max(metrics.get('val_f1', [0]))
            final_loss = metrics.get('val_loss', [0])[-1] if metrics.get('val_loss') else 0
            
            print(f"{exp_name:<20} {best_acc:<15.2f} {best_f1:<15.2f} {final_loss:<15.4f}")
    
    print("-" * 70)


# ==================== SCENARIO 10: Production Deployment ====================

class ProductionInference:
    """
    Production-ready inference with error handling and logging
    """
    
    def __init__(self, model_path: str, config_path: str):
        self.config = Config.load(config_path)
        self.model = INATNet(num_classes=2)
        
        # Load model
        model_manager = ModelManager()
        model_manager.load_model(self.model, model_path)
        
        # Create inference engine
        self.inference_engine = InferenceEngine(self.model, self.config)
        
        # Setup logging
        self.logger = Logger("production_logs", "production_inference")
    
    def predict_with_logging(self, image_path: str) -> Dict:
        """
        Predict with comprehensive logging and error handling
        """
        try:
            start_time = time.time()
            
            # Validate file exists
            if not Path(image_path).exists():
                self.logger.error(f"File not found: {image_path}")
                return {'error': 'File not found'}
            
            # Run prediction
            prediction, probabilities = self.inference_engine.predict_single(image_path)
            
            inference_time = time.time() - start_time
            
            result = {
                'image_path': image_path,
                'prediction': 'stego' if prediction == 1 else 'cover',
                'confidence': float(probabilities[prediction]),
                'probabilities': {
                    'cover': float(probabilities[0]),
                    'stego': float(probabilities[1])
                },
                'inference_time_ms': inference_time * 1000,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.2%})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {str(e)}")
            return {'error': str(e)}


def production_deployment():
    """
    Production deployment example
    """
    print("\n" + "="*70)
    print("SCENARIO 10: Production Deployment")
    print("="*70)
    
    # Initialize production inference
    prod_inference = ProductionInference(
        model_path="checkpoints/best_model.pth",
        config_path="configs/production_config.json"
    )
    
    # Process images
    test_images = [
        "production_test/image1.jpg",
        "production_test/image2.jpg",
        "production_test/image3.jpg",
    ]
    
    results = []
    for image_path in test_images:
        result = prod_inference.predict_with_logging(image_path)
        results.append(result)
        
        if 'error' not in result:
            print(f"\n{image_path}:")
            print(f"  Prediction: {result['prediction']}")
            print(f"  Confidence: {result['confidence']:.2%}")
            print(f"  Time: {result['inference_time_ms']:.2f}ms")
    
    # Save results to file
    results_file = "production_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n✓ Results saved to {results_file}")


# ==================== Main Usage Guide ====================

def print_usage_guide():
    """
    Print comprehensive usage guide
    """
    print("\n" + "="*70)
    print("STEGANALYSIS NETWORK ECOSYSTEM - USAGE GUIDE")
    print("="*70)
    
    guide = """
    
📚 AVAILABLE SCENARIOS:

1. quick_start_training()
   → Simplest way to start training with default settings

2. advanced_training()
   → Training with custom configurations and augmentation

3. resume_training()
   → Continue training from a saved checkpoint

4. single_image_inference()
   → Run inference on a single image

5. batch_inference()
   → Process multiple images efficiently

6. evaluate_test_set()
   → Comprehensive evaluation with metrics

7. progressive_size_training()
   → Train on progressively larger image sizes

8. training_with_custom_callbacks()
   → Add custom behavior during training

9. compare_experiments()
   → Compare results from multiple experiments

10. production_deployment()
    → Production-ready inference with logging


📁 PROJECT STRUCTURE:

project/
├── datasets/
│   ├── train/
│   │   ├── cover/
│   │   └── stego/
│   ├── val/
│   │   ├── cover/
│   │   └── stego/
│   └── test/
│       ├── cover/
│       └── stego/
├── checkpoints/         # Saved models
├── logs/                # Training logs and metrics
├── configs/             # Configuration files
├── models/              # Model architecture
└── results/             # Inference results


🚀 QUICK START EXAMPLE:

    from steganalysis_ecosystem import *
    from steganalysis_net import INATNet
    
    # 1. Create config
    config = Config({'batch_size': 32, 'num_epochs': 50})
    
    # 2. Initialize model and trainer
    model = INATNet(num_classes=2)
    trainer = Trainer(model, config, experiment_name="my_experiment")
    
    # 3. Setup data
    data_manager = DataManager(config)
    train_loader, val_loader = data_manager.create_dataloaders(
        train_cover_dir="datasets/train/cover",
        train_stego_dir="datasets/train/stego",
        val_cover_dir="datasets/val/cover",
        val_stego_dir="datasets/val/stego"
    )
    
    # 4. Train!
    trainer.fit(train_loader, val_loader)
    
    # 5. Inference
    inference = InferenceEngine(model, config)
    pred, probs = inference.predict_single("test_image.jpg")


💡 TIPS:

• Start with image_size=256 for faster experimentation
• Use mixed_precision=True for faster training on modern GPUs
• Enable early_stopping to prevent overfitting
• Save configs for reproducibility
• Monitor logs/ directory for training progress
• Use batch_inference() for processing many images
• Adjust batch_size based on your GPU memory


📊 MONITORING TRAINING:

• Check logs/{experiment_name}/ for training logs
• Metrics are saved as JSON and plotted automatically
• Best model is saved in checkpoints/best_model.pth
• Use tensorboard for real-time monitoring (if enabled)


🔧 TROUBLESHOOTING:

• Out of memory? Reduce batch_size or image_size
• Training too slow? Enable mixed_precision
• Overfitting? Increase weight_decay or add augmentation
• Underfitting? Train longer or increase model capacity

"""
    
    print(guide)


# ==================== Main Entry Point ====================

if __name__ == "__main__":
    print_usage_guide()
    
    print("\n" + "="*70)
    print("Choose a scenario to run (or run all with 'all'):")
    print("="*70)
    print("1. Quick Start Training")
    print("2. Advanced Training")
    print("3. Single Image Inference")
    print("4. Batch Inference")
    print("5. Evaluate Test Set")
    print("6. Compare Experiments")
    print("7. Production Deployment")
    print("="*70)
    
    # Example: Uncomment to run a specific scenario
    # quick_start_training()
    # advanced_training()
    # single_image_inference()
    # batch_inference()
    # evaluate_test_set()
    # compare_experiments()
    # production_deployment()