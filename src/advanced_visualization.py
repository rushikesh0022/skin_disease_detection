# src/advanced_visualization.py
"""
Advanced visualization with precise skin concern localization
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib.patches as patches


class PreciseLocalizer:
    """
    Precise localization of skin concerns with visual overlays
    """
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.DEVICE
        
        # Face landmark regions (approximate percentages)
        self.face_regions = {
            'forehead': {'y': (0.05, 0.35), 'x': (0.2, 0.8)},
            'left_eye': {'y': (0.25, 0.45), 'x': (0.15, 0.4)},
            'right_eye': {'y': (0.25, 0.45), 'x': (0.6, 0.85)},
            'left_cheek': {'y': (0.4, 0.75), 'x': (0.1, 0.45)},
            'right_cheek': {'y': (0.4, 0.75), 'x': (0.55, 0.9)},
            'nose': {'y': (0.35, 0.65), 'x': (0.35, 0.65)},
            'chin': {'y': (0.7, 0.95), 'x': (0.25, 0.75)}
        }
    
    def generate_gradcam_with_localization(self, input_tensor, original_image):
        """
        Generate GradCAM with precise localization for each concern
        
        Args:
            input_tensor: Preprocessed image tensor [1, 3, 224, 224]
            original_image: Original PIL Image
            
        Returns:
            Dictionary with localization data for each concern
        """
        self.model.eval()
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
        
        results = {}
        
        for idx, concern in enumerate(self.config.CONCERN_LABELS):
            score = float(probabilities[idx])
            
            if score > 0.5:  # Lower threshold to show overlays
                # Generate GradCAM
                gradcam = self._generate_concern_gradcam(input_tensor, idx)
                
                # Get precise localizations
                if concern == 'acne':
                    locations = self._localize_acne_spots(gradcam, original_image.size)
                elif concern == 'dark_circles':
                    locations = self._localize_dark_circles(gradcam, original_image.size)
                elif concern == 'redness':
                    locations = self._localize_redness_areas(gradcam, original_image.size)
                elif concern == 'wrinkles':
                    locations = self._localize_wrinkles(gradcam, original_image.size)
                else:
                    locations = []
                
                results[concern] = {
                    'score': score,
                    'severity': self._calculate_severity(score),
                    'gradcam': gradcam,
                    'locations': locations
                }
        
        return results
    
    def _generate_concern_gradcam(self, input_tensor, concern_idx):
        """Generate GradCAM for specific concern"""
        # Register hooks for ResNet layer4
        activations = {}
        gradients = {}
        
        def save_activation(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
        
        def save_gradient(name):
            def hook(module, grad_input, grad_output):
                gradients[name] = grad_output[0].detach()
            return hook
        
        # Register hooks
        target_layer = self.model.backbone.layer4
        target_layer.register_forward_hook(save_activation('layer4'))
        target_layer.register_full_backward_hook(save_gradient('layer4'))
        
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Backward pass for specific concern
        output[0, concern_idx].backward()
        
        # Get gradients and activations
        grads = gradients['layer4'][0]  # [C, H, W]
        acts = activations['layer4'][0]  # [C, H, W]
        
        # Compute weights and weighted activations  
        weights = torch.mean(grads, dim=(1, 2))
        cam = torch.zeros(acts.shape[1:], device=acts.device)  # Same device as activations
        
        for i in range(len(weights)):
            cam += weights[i] * acts[i]
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam.cpu().numpy()
    
    def _localize_acne_spots(self, gradcam, image_size):
        """
        Detect individual acne spots from GradCAM
        
        Returns:
            List of (x, y, radius, intensity) tuples
        """
        # Resize GradCAM to image size
        cam_resized = cv2.resize(gradcam, image_size)
        
        # Threshold to find high activation areas
        threshold = np.percentile(cam_resized, 70)  # Top 30% activations
        binary_mask = cam_resized > threshold
        
        # Find connected components (potential acne spots) using OpenCV
        binary_mask_uint8 = binary_mask.astype(np.uint8) * 255
        num_labels, labeled_mask = cv2.connectedComponents(binary_mask_uint8)
        
        spots = []
        for label in range(1, num_labels):  # Skip background (label 0)
            # Get component mask
            component_mask = (labeled_mask == label).astype(np.uint8)
            
            # Calculate area
            area = np.sum(component_mask)
            
            # Handle both individual spots AND diffuse acne areas
            if 20 < area < 10000:  # Increased max to catch larger affected areas
                # Find centroid
                moments = cv2.moments(component_mask)
                if moments['m00'] > 0:
                    x = int(moments['m10'] / moments['m00'])
                    y = int(moments['m01'] / moments['m00'])
                    
                    # Estimate radius from area
                    radius = np.sqrt(area / np.pi)
                    # Get intensity from original GradCAM
                    intensity = cam_resized[y, x] if 0 <= y < cam_resized.shape[0] and 0 <= x < cam_resized.shape[1] else 0
                    
                    spots.append({
                        'x': int(x),
                        'y': int(y), 
                        'radius': max(5, min(50, int(radius))),  # Cap radius between 5-50
                    'intensity': float(intensity),
                    'severity': self._intensity_to_severity(intensity)
                })
            elif area >= 10000:
                # For very large areas, divide into multiple indicator zones
                # Sample multiple points from the large region
                y_coords, x_coords = np.where(component_mask > 0)
                if len(x_coords) > 0:
                    # Create a reasonable grid of indicators across the affected area
                    # Limit to ~50-100 indicators for large areas to avoid clutter
                    num_indicators = min(50, max(10, int(np.sqrt(area) / 50)))
                    step = len(x_coords) // num_indicators
                    
                    for i in range(0, len(x_coords), max(step, 1)):
                        if i < len(x_coords) and len(spots) < 100:  # Cap total at 100
                            x, y = int(x_coords[i]), int(y_coords[i])
                            intensity = cam_resized[y, x] if 0 <= y < cam_resized.shape[0] and 0 <= x < cam_resized.shape[1] else 0
                            
                            if intensity > threshold * 1.2:  # Only mark high-intensity areas (20% above threshold)
                                spots.append({
                                    'x': x,
                                    'y': y, 
                                    'radius': 20,  # Slightly larger for visibility
                                'intensity': float(intensity),
                                'severity': self._intensity_to_severity(intensity)
                            })
        
        return spots
    
    def _localize_dark_circles(self, gradcam, image_size):
        """
        Create heatmap regions under eyes for dark circles
        
        Returns:
            List of heatmap regions under eyes
        """
        cam_resized = cv2.resize(gradcam, image_size)
        
        # Define eye regions
        h, w = image_size[1], image_size[0]
        
        eye_regions = [
            # Left eye area (under eye)
            {
                'name': 'left_under_eye',
                'bbox': (int(w * 0.15), int(h * 0.35), int(w * 0.25), int(h * 0.2)),
                'region': cam_resized[int(h * 0.35):int(h * 0.55), int(w * 0.15):int(w * 0.4)]
            },
            # Right eye area (under eye)
            {
                'name': 'right_under_eye', 
                'bbox': (int(w * 0.6), int(h * 0.35), int(w * 0.25), int(h * 0.2)),
                'region': cam_resized[int(h * 0.35):int(h * 0.55), int(w * 0.6):int(w * 0.85)]
            }
        ]
        
        heatmap_regions = []
        for eye_region in eye_regions:
            if eye_region['region'].size > 0:
                avg_intensity = np.mean(eye_region['region'])
                max_intensity = np.max(eye_region['region'])
                
                if avg_intensity > 0.3:  # Threshold for significant dark circles
                    heatmap_regions.append({
                        'name': eye_region['name'],
                        'bbox': eye_region['bbox'],
                        'intensity': float(avg_intensity),
                        'max_intensity': float(max_intensity),
                        'severity': self._intensity_to_severity(avg_intensity),
                        'heatmap_data': eye_region['region']
                    })
        
        return heatmap_regions
    
    def _localize_redness_areas(self, gradcam, image_size):
        """
        Identify redness areas on cheeks and forehead
        
        Returns:
            List of redness regions
        """
        cam_resized = cv2.resize(gradcam, image_size)
        h, w = image_size[1], image_size[0]
        
        # Define typical redness regions
        redness_regions = [
            {
                'name': 'forehead',
                'bbox': (int(w * 0.2), int(h * 0.05), int(w * 0.6), int(h * 0.3)),
                'region': cam_resized[int(h * 0.05):int(h * 0.35), int(w * 0.2):int(w * 0.8)]
            },
            {
                'name': 'left_cheek',
                'bbox': (int(w * 0.1), int(h * 0.4), int(w * 0.35), int(h * 0.35)),
                'region': cam_resized[int(h * 0.4):int(h * 0.75), int(w * 0.1):int(w * 0.45)]
            },
            {
                'name': 'right_cheek',
                'bbox': (int(w * 0.55), int(h * 0.4), int(w * 0.35), int(h * 0.35)),
                'region': cam_resized[int(h * 0.4):int(h * 0.75), int(w * 0.55):int(w * 0.9)]
            }
        ]
        
        detected_regions = []
        for region_info in redness_regions:
            region_cam = region_info['region']
            if region_cam.size > 0:
                # Check for significant redness activation
                threshold = np.percentile(region_cam, 70)  # Top 30%
                high_activation_mask = region_cam > threshold
                
                if np.sum(high_activation_mask) > region_cam.size * 0.1:  # At least 10% activation
                    avg_intensity = np.mean(region_cam)
                    detected_regions.append({
                        'name': region_info['name'],
                        'bbox': region_info['bbox'],
                        'intensity': float(avg_intensity),
                        'severity': self._intensity_to_severity(avg_intensity),
                        'heatmap_data': region_cam,
                        'activation_mask': high_activation_mask
                    })
        
        return detected_regions
    
    def _localize_wrinkles(self, gradcam, image_size):
        """
        Detect wrinkle lines on forehead and around eyes
        
        Returns:
            List of wrinkle line segments
        """
        cam_resized = cv2.resize(gradcam, image_size)
        
        # Focus on wrinkle-prone areas
        h, w = image_size[1], image_size[0]
        
        wrinkle_areas = {
            'forehead': {
                'region': cam_resized[int(h * 0.1):int(h * 0.4), int(w * 0.2):int(w * 0.8)],
                'offset': (int(w * 0.2), int(h * 0.1)),
                'line_direction': 'horizontal'  # Forehead wrinkles are usually horizontal
            },
            'eye_area': {
                'region': cam_resized[int(h * 0.2):int(h * 0.6), int(w * 0.1):int(w * 0.9)],
                'offset': (int(w * 0.1), int(h * 0.2)),
                'line_direction': 'radial'  # Eye wrinkles radiate from corners
            }
        }
        
        detected_lines = []
        
        for area_name, area_info in wrinkle_areas.items():
            region = area_info['region']
            if region.size == 0:
                continue
                
            # Enhance linear features using morphological operations
            threshold = np.percentile(region, 80)  # Top 20%
            binary_mask = region > threshold
            
            if np.sum(binary_mask) > 10:  # Minimum activation required
                # Use Hough line transform to detect lines
                lines = cv2.HoughLinesP(
                    binary_mask.astype(np.uint8) * 255,
                    rho=1,
                    theta=np.pi/180,
                    threshold=15,  # Increased from 10 for more selective detection
                    minLineLength=20,  # Increased from 15 for longer lines only
                    maxLineGap=8  # Increased to merge nearby line segments
                )
                
                if lines is not None and len(detected_lines) < 50:  # Limit total lines
                    # Sort lines by length to get most prominent wrinkles
                    line_data = []
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        line_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                        
                        # Only keep lines above minimum length threshold
                        if line_length > 25:
                            line_data.append((line, line_length))
                    
                    # Sort by length and take top lines
                    line_data.sort(key=lambda x: x[1], reverse=True)
                    max_lines_per_area = 15  # Limit per area
                    
                    for line, line_length in line_data[:max_lines_per_area]:
                        if len(detected_lines) >= 50:  # Global limit
                            break
                            
                        x1, y1, x2, y2 = line[0]
                        # Adjust coordinates to full image
                        x1 += area_info['offset'][0]
                        y1 += area_info['offset'][1] 
                        x2 += area_info['offset'][0]
                        y2 += area_info['offset'][1]
                        
                        # Calculate line intensity
                        try:
                            intensity = np.mean([region[y1-area_info['offset'][1], x1-area_info['offset'][0]],
                                               region[y2-area_info['offset'][1], x2-area_info['offset'][0]]])
                        except:
                            intensity = 0.5
                        
                        detected_lines.append({
                            'area': area_name,
                            'start': (x1, y1),
                            'end': (x2, y2),
                            'length': float(line_length),
                            'intensity': float(intensity),
                            'severity': self._intensity_to_severity(intensity)
                        })
        
        return detected_lines
    
    def _calculate_severity(self, score):
        """Convert probability score to severity level"""
        if score < 0.3:
            return "None"
        elif score < 0.5:
            return "Mild"
        elif score < 0.7:
            return "Moderate"
        else:
            return "Severe"
    
    def _intensity_to_severity(self, intensity):
        """Convert GradCAM intensity to severity"""
        if intensity < 0.3:
            return "Mild"
        elif intensity < 0.6:
            return "Moderate"
        else:
            return "Severe"


class AdvancedVisualizer:
    """
    Create advanced visualizations with precise overlays
    """
    
    def __init__(self, config):
        self.config = config
    
    def create_detailed_visualization(self, original_image, localization_results):
        """
        Create comprehensive visualization with all concern overlays
        
        Args:
            original_image: Original PIL Image
            localization_results: Results from PreciseLocalizer
            
        Returns:
            PIL Image with all overlays
        """
        # Convert to RGB if needed
        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')
        
        # Create drawing canvas
        img_draw = original_image.copy()
        draw = ImageDraw.Draw(img_draw)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
            small_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
        except:
            font = ImageFont.load_default()
            small_font = font
        
        # Draw overlays for each detected concern
        for concern, data in localization_results.items():
            if concern == 'acne':
                img_draw = self._draw_acne_spots(img_draw, data['locations'], draw, font)
            elif concern == 'dark_circles':
                img_draw = self._draw_dark_circle_heatmap(img_draw, data['locations'])
            elif concern == 'redness':
                img_draw = self._draw_redness_heatmap(img_draw, data['locations'])
            elif concern == 'wrinkles':
                img_draw = self._draw_wrinkle_lines(img_draw, data['locations'], draw)
        
        return img_draw
    
    def _draw_acne_spots(self, image, spots, draw, font):
        """Draw red circles over detected acne spots"""
        for spot in spots:
            x, y, radius = spot['x'], spot['y'], spot['radius']
            severity = spot['severity']
            
            # Color intensity based on severity
            if severity == "Mild":
                outline_color = (255, 150, 150)  # Light red
                fill_color = (255, 200, 200)     # Light fill
            elif severity == "Moderate":
                outline_color = (255, 100, 100)   # Medium red
                fill_color = (255, 150, 150)     # Medium fill
            else:  # Severe
                outline_color = (255, 50, 50)     # Dark red
                fill_color = (255, 100, 100)     # Strong fill
            
            # Draw filled circle with semi-transparency effect
            bbox = [x-radius, y-radius, x+radius, y+radius]
            # Draw a semi-transparent circle by drawing multiple circles with decreasing opacity
            draw.ellipse(bbox, fill=fill_color, outline=None)
            draw.ellipse(bbox, outline=outline_color, width=3)
            
            # Draw severity label (small circle indicator)
            label_radius = 8 if severity == "Mild" else 10 if severity == "Moderate" else 12
            severity_colors = {
                "Mild": (255, 200, 100),    # Yellow-orange
                "Moderate": (255, 150, 0),   # Orange
                "Severe": (255, 0, 0)        # Red
            }
            draw.ellipse([x-5, y-5, x+5, y+5], fill=severity_colors.get(severity, (255, 0, 0)))
        
        return image
    
    def _draw_dark_circle_heatmap(self, image, regions):
        """Draw heatmap overlay for dark circles under eyes"""
        img_array = np.array(image).astype(np.float32)
        
        for region in regions:
            bbox = region['bbox']
            x, y, w, h = bbox
            
            # Ensure bbox is within image bounds
            x = max(0, min(x, image.width - 1))
            y = max(0, min(y, image.height - 1))
            w = min(w, image.width - x)
            h = min(h, image.height - y)
            
            if w <= 0 or h <= 0:
                continue
            
            # Create heatmap overlay
            heatmap_data = region.get('heatmap_data')
            if heatmap_data is not None and heatmap_data.size > 0:
                try:
                    # Resize heatmap to bbox size
                    heatmap_resized = cv2.resize(heatmap_data, (w, h))
                    
                    # Apply Gaussian blur for smoother heatmap
                    heatmap_smooth = cv2.GaussianBlur(heatmap_resized, (5, 5), 0)
                    
                    # Create colored heatmap (purple-blue gradient for dark circles)
                    heatmap_colored = np.zeros((h, w, 3), dtype=np.float32)
                    
                    # Create gradient effect
                    heatmap_colored[:, :, 0] = heatmap_smooth * 180  # Blue
                    heatmap_colored[:, :, 2] = heatmap_smooth * 200  # Red (for purple)
                    
                    # Apply color mapping
                    heatmap_colored = cv2.applyColorMap(
                        (heatmap_smooth * 255).astype(np.uint8), 
                        cv2.COLORMAP_MAGMA
                    ).astype(np.float32)
                    
                    # Enhanced alpha blending
                    alpha = np.clip(heatmap_smooth * 0.7, 0, 0.7)  # Variable transparency
                    alpha = alpha[:, :, np.newaxis]
                    
                    # Blend with original image
                    img_array[y:y+h, x:x+w] = (
                        alpha * heatmap_colored + (1-alpha) * img_array[y:y+h, x:x+w]
                    )
                except Exception as e:
                    print(f"Error drawing dark circle heatmap: {e}")
                    continue
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    def _draw_redness_heatmap(self, image, regions):
        """Draw heatmap overlay for redness areas"""
        img_array = np.array(image).astype(np.float32)
        
        for region in regions:
            bbox = region['bbox']
            x, y, w, h = bbox
            
            # Ensure bbox is within image bounds
            x = max(0, min(x, image.width - 1))
            y = max(0, min(y, image.height - 1))
            w = min(w, image.width - x)
            h = min(h, image.height - y)
            
            if w <= 0 or h <= 0:
                continue
            
            heatmap_data = region.get('heatmap_data')
            if heatmap_data is not None and heatmap_data.size > 0:
                try:
                    # Resize heatmap to bbox size
                    heatmap_resized = cv2.resize(heatmap_data, (w, h))
                    
                    # Apply Gaussian blur for smoother heatmap
                    heatmap_smooth = cv2.GaussianBlur(heatmap_resized, (7, 7), 0)
                    
                    # Create a more sophisticated red heatmap
                    # Convert to uint8 for color mapping
                    heatmap_uint8 = (heatmap_smooth * 255).astype(np.uint8)
                    
                    # Apply custom colormap (red-focused)
                    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_HOT).astype(np.float32)
                    
                    # Enhance red channel and reduce others for more prominent redness
                    heatmap_colored[:, :, 2] *= 1.5  # Boost red
                    heatmap_colored[:, :, 1] *= 0.3  # Reduce green
                    heatmap_colored[:, :, 0] *= 0.3  # Reduce blue
                    
                    # Normalize values
                    heatmap_colored = np.clip(heatmap_colored, 0, 255)
                    
                    # Create dynamic alpha based on intensity
                    alpha = np.clip(heatmap_smooth * 0.8, 0.2, 0.8)  # Variable transparency
                    alpha = alpha[:, :, np.newaxis]
                    
                    # Blend with original image
                    img_array[y:y+h, x:x+w] = (
                        alpha * heatmap_colored + (1-alpha) * img_array[y:y+h, x:x+w]
                    )
                    
                    # Add subtle glow effect
                    glow = cv2.GaussianBlur(heatmap_colored, (15, 15), 0)
                    glow_alpha = np.clip(heatmap_smooth * 0.3, 0, 0.3)[:, :, np.newaxis]
                    img_array[y:y+h, x:x+w] += glow_alpha * glow
                    
                except Exception as e:
                    print(f"Error drawing redness heatmap: {e}")
                    continue
        
        # Final normalization
        img_array = np.clip(img_array, 0, 255)
        return Image.fromarray(img_array.astype(np.uint8))
    
    def _draw_wrinkle_lines(self, image, lines, draw):
        """Draw line indicators over detected wrinkles"""
        for line in lines:
            start = line['start']
            end = line['end']
            severity = line['severity']
            
            # Ensure coordinates are valid
            if (not (0 <= start[0] < image.width and 0 <= start[1] < image.height) or
                not (0 <= end[0] < image.width and 0 <= end[1] < image.height)):
                continue
            
            # Color and thickness based on severity
            if severity == "Mild":
                color = (200, 200, 0)    # Yellow
                width = 2
                outline_color = (255, 255, 100)  # Light yellow outline
            elif severity == "Moderate":
                color = (255, 165, 0)    # Orange
                width = 3
                outline_color = (255, 200, 100)  # Light orange outline
            else:  # Severe
                color = (255, 69, 0)     # Red-orange
                width = 4
                outline_color = (255, 120, 50)   # Light red-orange outline
            
            # Draw line with slight offset to create outline effect
            # Draw outline
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                try:
                    outline_start = (start[0] + dx, start[1] + dy)
                    outline_end = (end[0] + dx, end[1] + dy)
                    draw.line([outline_start, outline_end], fill=outline_color, width=width+2)
                except:
                    pass
            
            # Draw main line on top
            draw.line([start, end], fill=color, width=width)
        
        return image
    
    def create_analysis_panel(self, localization_results, image_size):
        """
        Create detailed analysis panel with severity breakdown
        
        Returns:
            PIL Image with analysis information
        """
        panel_width = 300
        panel_height = max(400, len(localization_results) * 100)
        
        # Create panel image
        panel = Image.new('RGB', (panel_width, panel_height), 'white')
        draw = ImageDraw.Draw(panel)
        
        # Try to load font
        try:
            title_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 18)
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
        except:
            title_font = ImageFont.load_default()
            font = title_font
        
        # Draw title
        draw.text((10, 10), "Skin Analysis Report", fill='black', font=title_font)
        
        y_pos = 50
        for concern, data in localization_results.items():
            score = data['score']
            severity = data['severity']
            locations = data['locations']
            
            # Concern title
            concern_name = concern.replace('_', ' ').title()
            draw.text((10, y_pos), f"{concern_name}:", fill='black', font=font)
            y_pos += 20
            
            # Score and severity
            draw.text((20, y_pos), f"Confidence: {score:.1%}", fill='blue', font=font)
            y_pos += 15
            draw.text((20, y_pos), f"Severity: {severity}", fill='red', font=font)
            y_pos += 15
            
            # Location details
            if concern == 'acne' and locations:
                draw.text((20, y_pos), f"Spots detected: {len(locations)}", fill='black', font=font)
                y_pos += 15
            elif concern in ['dark_circles', 'redness'] and locations:
                regions = [loc['name'] for loc in locations]
                draw.text((20, y_pos), f"Affected areas: {', '.join(regions)}", fill='black', font=font)
                y_pos += 15
            elif concern == 'wrinkles' and locations:
                draw.text((20, y_pos), f"Lines detected: {len(locations)}", fill='black', font=font)
                y_pos += 15
            
            y_pos += 20
        
        return panel


def create_complete_analysis(model, image_path, config):
    """
    Complete analysis pipeline with precise localization and visualization
    
    Args:
        model: Trained model
        image_path: Path to image
        config: Configuration object
        
    Returns:
        Dictionary with analysis results and visualizations
    """
    from src.preprocessing import create_transforms
    
    # Load and preprocess image
    original_image = Image.open(image_path).convert('RGB')
    transform = create_transforms(train=False, image_size=config.IMAGE_SIZE)
    input_tensor = transform(original_image).unsqueeze(0).to(config.DEVICE)
    
    # Initialize localizer and visualizer
    localizer = PreciseLocalizer(model, config)
    visualizer = AdvancedVisualizer(config)
    
    # Get precise localizations
    localization_results = localizer.generate_gradcam_with_localization(
        input_tensor, original_image
    )
    
    # Create visualizations
    detailed_image = visualizer.create_detailed_visualization(
        original_image, localization_results
    )
    
    analysis_panel = visualizer.create_analysis_panel(
        localization_results, original_image.size
    )
    
    return {
        'original_image': original_image,
        'detailed_visualization': detailed_image,
        'analysis_panel': analysis_panel,
        'localization_data': localization_results
    }


if __name__ == '__main__':
    # Demo usage
    from src.config import Config
    from models.resnet_model import SkinConcernDetector
    
    config = Config()
    
    # Load model
    model = SkinConcernDetector(num_classes=config.NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.DEVICE)
    
    # Test image
    test_image = "data/processed/sample_image.jpg"  # Replace with actual image
    
    if os.path.exists(test_image):
        results = create_complete_analysis(model, test_image, config)
        
        # Save results
        results['detailed_visualization'].save('outputs/detailed_analysis.png')
        results['analysis_panel'].save('outputs/analysis_report.png')
        
        print("✅ Advanced analysis complete!")
        print("📊 Check outputs/detailed_analysis.png for visual results")
        print("📋 Check outputs/analysis_report.png for detailed report")
