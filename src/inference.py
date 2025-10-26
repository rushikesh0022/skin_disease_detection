# src/inference.py
"""
Inference script for Face Concern Detector
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet_model import SkinConcernDetector
from src.preprocessing import create_transforms, FacePreprocessor
from src.config import Config
from src.gradcam import apply_gradcam_to_image


class FaceConcernPredictor:
    """Predictor for skin concern detection"""
    
    def __init__(self, model_path, config):
        """
        Args:
            model_path: Path to trained model weights
            config: Configuration object
        """
        self.config = config
        self.device = config.DEVICE
        
        # Load model
        self.model = SkinConcernDetector(
            num_classes=config.NUM_CLASSES,
            pretrained=False
        ).to(self.device)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Initialize preprocessor
        self.preprocessor = FacePreprocessor(
            image_size=config.IMAGE_SIZE,
            margin=config.FACE_MARGIN
        )
        
        # Initialize transform
        self.transform = create_transforms(train=False, 
                                          image_size=config.IMAGE_SIZE)
        
        print(f"Model loaded from {model_path}")
        print(f"Running on device: {self.device}")
    
    def predict(self, image_path, return_gradcam=False):
        """
        Predict skin concerns for image
        
        Args:
            image_path: Path to input image
            return_gradcam: Whether to generate GradCAM
            
        Returns:
            Dictionary with predictions
        """
        # Preprocess image
        face_image = self.preprocessor.preprocess_image(image_path)
        
        if face_image is None:
            return {'error': 'No face detected'}
        
        # Transform
        input_tensor = self.transform(face_image).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_tensor)
            scores = outputs.cpu().numpy()[0]
        
        # Create results dictionary
        results = {
            'image_path': image_path,
            'scores': {},
            'detected_concerns': []
        }
        
        for idx, concern in enumerate(self.config.CONCERN_LABELS):
            score = float(scores[idx])
            results['scores'][concern] = score
            
            if score > self.config.THRESHOLD:
                results['detected_concerns'].append(concern)
        
        # Generate GradCAM if requested
        if return_gradcam:
            from src.gradcam import MultiLabelGradCAM
            
            gradcam = MultiLabelGradCAM(self.model, 'layer4')
            cams = gradcam.generate_cams(input_tensor, 
                                        self.config.CONCERN_LABELS)
            results['gradcam'] = cams
            results['face_image'] = face_image
        
        return results
    
    def predict_batch(self, image_paths):
        """
        Predict for multiple images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for img_path in image_paths:
            result = self.predict(img_path)
            results.append(result)
        return results
    
    def visualize_predictions(self, image_path, save_path=None):
        """
        Visualize predictions with GradCAM
        
        Args:
            image_path: Path to input image
            save_path: Optional path to save visualization
        """
        # Get predictions
        results = self.predict(image_path, return_gradcam=True)
        
        if 'error' in results:
            print(f"Error: {results['error']}")
            return
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Original image
        original = Image.open(image_path)
        axes[0].imshow(original)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Face crop
        axes[1].imshow(results['face_image'])
        axes[1].set_title('Detected Face', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # GradCAM for each concern
        import cv2
        for idx, concern in enumerate(self.config.CONCERN_LABELS):
            if idx + 2 >= len(axes):
                break
            
            cam_data = results['gradcam'][concern]
            score = cam_data['score']
            cam = cam_data['cam']
            
            # Resize and apply colormap
            face_img = results['face_image']
            cam_resized = cv2.resize(cam, (face_img.width, face_img.height))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), 
                                       cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Overlay
            img_array = np.array(face_img)
            overlayed = heatmap * 0.4 + img_array * 0.6
            overlayed = np.uint8(overlayed)
            
            axes[idx + 2].imshow(overlayed)
            color = 'green' if score > self.config.THRESHOLD else 'red'
            axes[idx + 2].set_title(
                f'{concern.replace("_", " ").title()}: {score:.1%}',
                fontsize=11, fontweight='bold', color=color
            )
            axes[idx + 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Saved visualization to {save_path}")
        
        plt.show()
        
        return results


def main():
    """Demo inference"""
    config = Config()
    
    # Initialize predictor
    predictor = FaceConcernPredictor(
        model_path=config.BEST_MODEL_PATH,
        config=config
    )
    
    # Example: predict single image
    test_image = os.path.join(config.SAMPLE_DIR, 'test_face.jpg')
    
    if os.path.exists(test_image):
        # Simple prediction
        results = predictor.predict(test_image)
        
        print("\n" + "="*50)
        print("PREDICTION RESULTS")
        print("="*50)
        print(f"Image: {results['image_path']}")
        print("\nScores:")
        for concern, score in results['scores'].items():
            status = "✓ DETECTED" if score > config.THRESHOLD else "✗ Not detected"
            print(f"  {concern.replace('_', ' ').title():<20} {score:.2%} {status}")
        
        # Visualize with GradCAM
        save_path = os.path.join(config.VIZ_DIR, 'prediction_result.png')
        predictor.visualize_predictions(test_image, save_path)
    else:
        print(f"Test image not found: {test_image}")
        print("Please place test images in:", config.SAMPLE_DIR)


if __name__ == '__main__':
    main()
