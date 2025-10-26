# test_model.py
"""
Test script to verify if the trained Face Concern Detector model is working
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.dirname(__file__))

from models.resnet_model import SkinConcernDetector
from src.preprocessing import create_transforms
from src.config import Config


class ModelTester:
    """Test the trained model"""
    
    def __init__(self, model_path=None):
        self.config = Config()
        self.device = self.config.DEVICE
        self.concern_labels = self.config.CONCERN_LABELS
        
        # Load model
        self.model = SkinConcernDetector(
            num_classes=self.config.NUM_CLASSES,
            pretrained=False  # We'll load trained weights
        ).to(self.device)
        
        # Load trained weights
        if model_path is None:
            model_path = self.config.BEST_MODEL_PATH
        
        if os.path.exists(model_path):
            print(f"✅ Loading model from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Model loaded! Trained for {checkpoint['epoch']+1} epochs")
            print(f"✅ Best validation loss: {checkpoint['val_loss']:.4f}")
        else:
            print(f"❌ Model file not found: {model_path}")
            print("❌ Please train the model first by running: python3 src/train.py")
            return
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Create transform for inference
        self.transform = create_transforms(train=False, image_size=self.config.IMAGE_SIZE)
    
    def predict_single_image(self, image_path):
        """Predict skin concerns for a single image"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
                predictions = (probabilities > self.config.THRESHOLD).astype(int)
            
            # Format results
            results = {}
            for i, concern in enumerate(self.concern_labels):
                results[concern] = {
                    'probability': float(probabilities[i]),
                    'predicted': bool(predictions[i])
                }
            
            return results, image
            
        except Exception as e:
            print(f"❌ Error processing image {image_path}: {str(e)}")
            return None, None
    
    def test_with_sample_images(self):
        """Test model with sample images from the dataset"""
        print("\n🔍 Testing model with sample images...")
        
        # Look for sample images in processed directory
        processed_dir = self.config.PROCESSED_DIR
        if not os.path.exists(processed_dir):
            print(f"❌ Processed directory not found: {processed_dir}")
            return
        
        # Get some sample images
        image_files = [f for f in os.listdir(processed_dir) 
                      if f.endswith(('.jpg', '.png', '.jpeg'))][:5]
        
        if not image_files:
            print("❌ No sample images found in processed directory")
            return
        
        print(f"✅ Found {len(image_files)} sample images")
        
        # Create output directory for results
        results_dir = os.path.join(self.config.OUTPUT_DIR, 'test_results')
        os.makedirs(results_dir, exist_ok=True)
        
        for i, img_file in enumerate(image_files):
            print(f"\n--- Testing Image {i+1}: {img_file} ---")
            
            img_path = os.path.join(processed_dir, img_file)
            results, image = self.predict_single_image(img_path)
            
            if results:
                # Print results
                print("Predictions:")
                for concern, data in results.items():
                    status = "✅ DETECTED" if data['predicted'] else "❌ Not detected"
                    print(f"  {concern.replace('_', ' ').title()}: {data['probability']:.3f} ({status})")
                
                # Save visualization
                self.visualize_prediction(image, results, img_file, results_dir)
    
    def visualize_prediction(self, image, results, filename, save_dir):
        """Create visualization of prediction results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Show original image
        ax1.imshow(image)
        ax1.set_title(f"Input Image: {filename}")
        ax1.axis('off')
        
        # Show prediction results as bar chart
        concerns = list(results.keys())
        probabilities = [results[c]['probability'] for c in concerns]
        colors = ['green' if results[c]['predicted'] else 'red' for c in concerns]
        
        bars = ax2.bar(range(len(concerns)), probabilities, color=colors, alpha=0.7)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('Probability')
        ax2.set_title('Skin Concern Detection Results')
        ax2.set_xticks(range(len(concerns)))
        ax2.set_xticklabels([c.replace('_', '\n').title() for c in concerns], rotation=45)
        
        # Add threshold line
        ax2.axhline(y=self.config.THRESHOLD, color='black', linestyle='--', alpha=0.5, label=f'Threshold ({self.config.THRESHOLD})')
        ax2.legend()
        
        # Add probability values on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(save_dir, f"prediction_{filename}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Visualization saved: {save_path}")
    
    def test_model_architecture(self):
        """Test if model architecture is working correctly"""
        print("\n🔧 Testing model architecture...")
        
        # Test with dummy input
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        try:
            with torch.no_grad():
                output = self.model(dummy_input)
            
            print(f"✅ Model forward pass successful!")
            print(f"✅ Input shape: {dummy_input.shape}")
            print(f"✅ Output shape: {output.shape}")
            print(f"✅ Expected output shape: (1, {self.config.NUM_CLASSES})")
            
            if output.shape == (1, self.config.NUM_CLASSES):
                print("✅ Output shape is correct!")
                
                # Check output values
                probabilities = torch.sigmoid(output).cpu().numpy()[0]
                print(f"✅ Output probabilities: {probabilities}")
                print("✅ All probabilities are between 0 and 1!" if all(0 <= p <= 1 for p in probabilities) else "❌ Invalid probabilities!")
                
                return True
            else:
                print(f"❌ Output shape mismatch! Expected (1, {self.config.NUM_CLASSES}), got {output.shape}")
                return False
                
        except Exception as e:
            print(f"❌ Model forward pass failed: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Run comprehensive model testing"""
        print("🚀 Starting comprehensive model testing...")
        print("="*60)
        
        # Test 1: Model architecture
        arch_ok = self.test_model_architecture()
        
        if arch_ok:
            # Test 2: Sample image predictions
            self.test_with_sample_images()
            
            print("\n" + "="*60)
            print("✅ Model testing completed!")
            print(f"📁 Check results in: {self.config.OUTPUT_DIR}/test_results/")
            print("\nTo test with your own images:")
            print("  from test_model import ModelTester")
            print("  tester = ModelTester()")
            print("  results, image = tester.predict_single_image('path/to/your/image.jpg')")
        else:
            print("\n❌ Model architecture test failed. Please check your model training.")


def main():
    """Main testing function"""
    tester = ModelTester()
    if hasattr(tester, 'model'):  # Check if model loaded successfully
        tester.run_all_tests()


if __name__ == '__main__':
    main()
