"""
Steganalysis Network Ecosystem
A complete, modular framework for training, validating, and deploying steganalysis models
"""
from app.inat_net import INATNet
from app.trainer import Trainer
from app.data import DataManager
from app.inference_engine import InferenceEngine
from app.config import Config

if __name__ == '__main__':
    # ==================== Usage Example ====================
    """Example of how to use the ecosystem"""
    print("="*60)
    print("Steganalysis Network Ecosystem - Usage Example")
    print("="*60)

    # 1. Create configuration
    config = Config({
        'batch_size': 4,
        'num_epochs': 50,
        'image_size': 360,
        'early_stopping': False,
        'save_best_only': False,
        'use_scheduler': False,
        'custom_progress_callback': True
    })

    # Save configuration
    config.save('experiment_config.json')

    # 2. Initialize model (assuming INATNet is imported)
    model = INATNet(num_classes=2)

    # 3. Create trainer
    trainer = Trainer(model, config, experiment_name="experiment_001")

    # 4. Setup data
    data_manager = DataManager(config)
    train_loader, val_loader = data_manager.create_dataloaders(
        train_cover_dir="data/custom_big/train/cover",
        train_stego_dir="data/custom_big/train/stego",
        val_cover_dir="data/custom_big/val/cover",
        val_stego_dir="data/custom_big/val/stego"
    )

    # 5. Train
    trainer.fit(train_loader, val_loader)

    # 6. Inference
    inference_engine = InferenceEngine(model, config)
    prediction, probabilities = inference_engine.predict_single("images/stego_image.png")
    print(f"Prediction: {'Stego' if prediction == 1 else 'Cover'}")
    print(f"Probabilities: {probabilities}")

    prediction, probabilities = inference_engine.predict_single("images/cover_image.png")
    print(f"Prediction: {'Stego' if prediction == 1 else 'Cover'}")
    print(f"Probabilities: {probabilities}")
