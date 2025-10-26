# src/preprocessing.py
"""
Face detection and preprocessing pipeline
"""

import cv2
import numpy as np
from mtcnn import MTCNN
from PIL import Image
import torch


class FacePreprocessor:
    """Face detection and preprocessing"""
    
    def __init__(self, image_size=224, margin=20):
        self.detector = MTCNN(device='CPU:0')
        self.image_size = image_size
        self.margin = margin
    
    def detect_face(self, image):
        """
        Detect face in image
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Face bounding box or None
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Detect faces
        results = self.detector.detect_faces(image)
        
        if len(results) == 0:
            return None
        
        # Get the face with highest confidence
        best_face = max(results, key=lambda x: x['confidence'])
        return best_face['box']
    
    def crop_face(self, image, bbox):
        """
        Crop face from image with margin
        
        Args:
            image: PIL Image or numpy array
            bbox: [x, y, w, h]
            
        Returns:
            Cropped face as PIL Image
        """
        # Convert to numpy if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        x, y, w, h = bbox
        
        # Add margin
        x1 = max(0, x - self.margin)
        y1 = max(0, y - self.margin)
        x2 = min(img_array.shape[1], x + w + self.margin)
        y2 = min(img_array.shape[0], y + h + self.margin)
        
        # Crop
        face = img_array[y1:y2, x1:x2]
        
        # Convert back to PIL
        return Image.fromarray(face)
    
    def preprocess_image(self, image_path):
        """
        Complete preprocessing pipeline
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed face image or None if no face detected
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Detect face
        bbox = self.detect_face(image)
        
        if bbox is None:
            print(f"No face detected in {image_path}")
            return None
        
        # Crop face
        face = self.crop_face(image, bbox)
        
        # Resize
        face = face.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        return face
    
    def align_face(self, image, landmarks):
        """
        Align face using eye landmarks (optional enhancement)
        
        Args:
            image: PIL Image
            landmarks: Dictionary with facial landmarks
            
        Returns:
            Aligned face image
        """
        # Get eye coordinates
        left_eye = landmarks.get('left_eye')
        right_eye = landmarks.get('right_eye')
        
        if left_eye is None or right_eye is None:
            return image
        
        # Calculate angle
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        # Rotate image
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(img_array, M, (w, h))
        
        return Image.fromarray(aligned)


def create_transforms(train=True, image_size=224):
    """
    Create data augmentation transforms
    
    Args:
        train: Whether this is for training (applies augmentation)
        image_size: Target image size
        
    Returns:
        torchvision transforms
    """
    from torchvision import transforms
    
    if train:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    return transform
