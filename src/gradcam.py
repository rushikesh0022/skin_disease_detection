# src/gradcam.py
"""
GradCAM implementation for visualizing model predictions
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: Trained model
            target_layer: Layer to compute GradCAM (e.g., model.backbone.layer4)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook to save forward activation"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward gradient"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        """
        Generate Class Activation Map
        
        Args:
            input_image: Input tensor [1, 3, H, W]
            target_class: Target class index (None for max prediction)
            
        Returns:
            CAM heatmap [H, W]
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling on gradients
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()
    
    def visualize_cam(self, original_image, cam, alpha=0.5):
        """
        Overlay CAM on original image
        
        Args:
            original_image: Original PIL image
            cam: CAM heatmap [H, W]
            alpha: Overlay transparency
            
        Returns:
            Visualization as PIL Image
        """
        # Resize CAM to match original image
        cam_resized = cv2.resize(cam, (original_image.width, original_image.height))
        
        # Convert to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Convert original to array
        img_array = np.array(original_image)
        
        # Overlay
        overlayed = heatmap * alpha + img_array * (1 - alpha)
        overlayed = np.uint8(overlayed)
        
        return Image.fromarray(overlayed)


class MultiLabelGradCAM:
    """
    GradCAM for multi-label classification
    Generates separate CAM for each concern
    """
    
    def __init__(self, model, target_layer_name='layer4'):
        """
        Args:
            model: Trained SkinConcernDetector
            target_layer_name: Name of target layer
        """
        self.model = model
        self.model.eval()
        
        # Get target layer
        self.target_layer = getattr(model.backbone, target_layer_name)
        self.gradients = {}
        self.activations = {}
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations['value'] = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients['value'] = grad_output[0].detach()
    
    def generate_cams(self, input_tensor, concern_labels):
        """
        Generate CAM for each skin concern
        
        Args:
            input_tensor: Input tensor [1, 3, 224, 224]
            concern_labels: List of concern names
            
        Returns:
            Dictionary mapping concern -> (score, cam)
        """
        # Forward pass
        self.model.zero_grad()
        outputs = self.model(input_tensor)
        
        cams = {}
        
        for idx, concern in enumerate(concern_labels):
            # Backward for this concern
            self.model.zero_grad()
            outputs[0, idx].backward(retain_graph=True)
            
            # Get gradients and activations
            gradients = self.gradients['value'][0]  # [C, H, W]
            activations = self.activations['value'][0]  # [C, H, W]
            
            # Compute weights
            weights = torch.mean(gradients, dim=(1, 2))
            
            # Weighted sum
            cam = torch.zeros(activations.shape[1:])
            for i in range(len(weights)):
                cam += weights[i] * activations[i]
            
            # ReLU and normalize
            cam = F.relu(cam)
            if cam.max() > 0:
                cam = (cam - cam.min()) / (cam.max() - cam.min())
            
            cams[concern] = {
                'score': outputs[0, idx].item(),
                'cam': cam.cpu().numpy()
            }
        
        return cams
    
    def visualize_all_concerns(self, original_image, cams, threshold=0.5):
        """
        Create visualization grid for all concerns
        
        Args:
            original_image: PIL Image
            cams: Dictionary from generate_cams
            threshold: Only visualize if score > threshold
            
        Returns:
            Grid visualization
        """
        n_concerns = len(cams)
        fig, axes = plt.subplots(2, (n_concerns + 1) // 2, 
                                figsize=(12, 6))
        axes = axes.flatten()
        
        for idx, (concern, data) in enumerate(cams.items()):
            score = data['score']
            cam = data['cam']
            
            # Resize CAM
            cam_resized = cv2.resize(cam, 
                                    (original_image.width, 
                                     original_image.height))
            
            # Create heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), 
                                       cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Overlay
            img_array = np.array(original_image)
            overlayed = heatmap * 0.4 + img_array * 0.6
            overlayed = np.uint8(overlayed)
            
            # Plot
            axes[idx].imshow(overlayed)
            color = 'green' if score > threshold else 'red'
            axes[idx].set_title(f'{concern}: {score:.2%}', 
                               color=color, fontweight='bold')
            axes[idx].axis('off')
        
        # Hide extra subplots
        for idx in range(len(cams), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        return fig


def apply_gradcam_to_image(model, image_path, config, save_path=None):
    """
    Complete GradCAM pipeline for single image
    
    Args:
        model: Trained model
        image_path: Path to input image
        config: Configuration object
        save_path: Optional path to save visualization
        
    Returns:
        Predictions and visualizations
    """
    from src.preprocessing import create_transforms
    
    # Load and preprocess image
    original_image = Image.open(image_path).convert('RGB')
    transform = create_transforms(train=False, 
                                  image_size=config.IMAGE_SIZE)
    input_tensor = transform(original_image).unsqueeze(0)
    input_tensor = input_tensor.to(config.DEVICE)
    
    # Initialize GradCAM
    gradcam = MultiLabelGradCAM(model, 'layer4')
    
    # Generate CAMs
    cams = gradcam.generate_cams(input_tensor, config.CONCERN_LABELS)
    
    # Visualize
    fig = gradcam.visualize_all_concerns(original_image, cams, 
                                         threshold=config.THRESHOLD)
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to {save_path}")
    
    return cams, fig
