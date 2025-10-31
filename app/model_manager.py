import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
import torch
import torch.nn as nn

# ==================== Model Manager ====================

class ModelManager:
    """Manage model loading, saving, and version control"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def save_model(self, model: nn.Module, name: str, metadata: Optional[Dict] = None):
        """Save model with metadata"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = self.model_dir / f"{name}_{timestamp}.pth"
        
        save_dict = {
            'model_state_dict': model.state_dict(),
            'metadata': metadata or {},
            'timestamp': timestamp
        }
        
        torch.save(save_dict, model_path)
        print(f"Model saved to {model_path}")
        
        # Save metadata separately
        if metadata:
            metadata_path = model_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
        
        return model_path
    
    def load_model(self, model: nn.Module, model_path: str) -> Dict:
        """Load model and return metadata"""
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {model_path}")
        return checkpoint.get('metadata', {})
    
    def list_models(self) -> List[str]:
        """List all saved models"""
        return [str(p) for p in self.model_dir.glob("*.pth")]