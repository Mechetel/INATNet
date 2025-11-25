import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import List, Tuple
from app.config import Config
import torchvision.transforms as T


# ==================== Inference Engine ====================

class InferenceEngine:
    """Engine for model inference"""

    def __init__(self, model: nn.Module, config: Config):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.model.eval()

        # Setup transforms
        self.transform = T.Compose([
            T.Resize((config.image_size, config.image_size)),
            T.ToTensor()
        ])

    def predict_single(self, image_path: str) -> Tuple[int, np.ndarray]:
        """Predict on a single image"""
        image = Image.open(image_path).convert('RGB')
        input_tensor = T.ToTensor()(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1)

        return prediction.item(), probabilities[0].cpu().numpy()

    def predict_batch(self, image_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict on a batch of images"""
        images = []
        for path in image_paths:
            image = Image.open(path).convert('RGB')
            images.append(self.transform(image))

        batch = torch.stack(images).to(self.device)

        with torch.no_grad():
            outputs = self.model(batch)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

        return predictions.cpu().numpy(), probabilities.cpu().numpy()

    def predict_from_array(self, image_array: np.ndarray) -> Tuple[int, np.ndarray]:
        """Predict from numpy array"""
        image = Image.fromarray(image_array.astype(np.uint8))
        input_tensor = T.ToTensor()(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1)

        return prediction.item(), probabilities[0].cpu().numpy()
