from app.inat_net import INATNet
from helpers.utils import get_model_info
from app.trainer import Trainer
from app.data import DataManager
from app.config import Config
from app.inference_engine import InferenceEngine
from app.model_manager import ModelManager
from torch.utils.data import DataLoader
from app.data import SteganalysisDataset


model = INATNet(num_classes=2)
get_model_info(model)

config = Config.load('configs/experiment_config.json')
model_manager = ModelManager()
loaded_model = model_manager.load_model(model, "/Users/dmitryhoma/Projects/phd_dissertation/state_3/INATNet/models/inatnet_v1_20251031_134512.pth")

inference_engine = InferenceEngine(model, config)
prediction, probabilities = inference_engine.predict_single("images/stego_image.png")
print(f"Prediction: {'Stego' if prediction == 1 else 'Cover'}")
print(f"Probabilities: {probabilities}")

prediction, probabilities = inference_engine.predict_single("images/cover_image.png")
print(f"Prediction: {'Stego' if prediction == 1 else 'Cover'}")
print(f"Probabilities: {probabilities}")


# Setup
config = Config.load('configs/experiment_config.json')
model = INATNet(num_classes=2)

# Load best model
trainer = Trainer(model, config, experiment_name="evaluation")
trainer.load_checkpoint("models/inatnet_v1_20251031_134512.pth")

# Create test dataloader
data_manager = DataManager(config)

test_dataset = SteganalysisDataset(
  cover_dir="data/custom_big/test/cover",
  stego_dir="data/custom_big/test/stego",
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