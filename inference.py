# inference.py
"""
Simple inference script for testing individual images
"""

import os
import sys
import torch
from PIL import Image
import argparse

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.dirname(__file__))

from models.resnet_model import SkinConcernDetector
from src.preprocessing import create_transforms
from src.config import Config


class FaceConcernInference:
    """Simple inference class for individual image testing"""
    
    def __init__(self, model_path=None):
        self.config = Config()
        self.device = self.config.DEVICE
        
        # Load model
        self.model = SkinConcernDetector(
            num_classes=self.config.NUM_CLASSES,
            pretrained=False
        ).to(self.device)
        
        # Load weights
        if model_path is None:
            model_path = self.config.BEST_MODEL_PATH
            
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Preprocessing
        self.transform = create_transforms(train=False, image_size=self.config.IMAGE_SIZE)
        
        print(f"✅ Model loaded from: {model_path}")
        print(f"✅ Device: {self.device}")
    
    def predict(self, image_path):
        """Predict skin concerns for an image"""
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
        
        # Format results
        results = {}
        for i, concern in enumerate(self.config.CONCERN_LABELS):
            prob = float(probabilities[i])
            predicted = prob > self.config.THRESHOLD
            results[concern] = {
                'probability': prob,
                'predicted': predicted
            }
        
        return results
    
    def print_results(self, results, image_path):
        """Print formatted results"""
        print(f"\n🔍 Analysis for: {os.path.basename(image_path)}")
        print("="*50)
        
        for concern, data in results.items():
            status = "✅ DETECTED" if data['predicted'] else "❌ Not detected"
            concern_name = concern.replace('_', ' ').title()
            print(f"{concern_name:15}: {data['probability']:.3f} ({status})")
        
        detected = [c for c, d in results.items() if d['predicted']]
        if detected:
            print(f"\n🎯 Detected concerns: {', '.join([c.replace('_', ' ').title() for c in detected])}")
        else:
            print("\n✨ No skin concerns detected above threshold!")


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='Test Face Concern Detector on an image')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--model', help='Path to model file (optional)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"❌ Image file not found: {args.image_path}")
        return
    
    try:
        # Initialize inference
        inferencer = FaceConcernInference(args.model)
        
        # Make prediction
        results = inferencer.predict(args.image_path)
        
        # Print results
        inferencer.print_results(results, args.image_path)
        
    except Exception as e:
        print(f"❌ Error during inference: {str(e)}")


if __name__ == '__main__':
    main()
