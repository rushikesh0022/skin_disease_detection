# src/config.py
"""
Configuration file for Face Concern Detector
"""

import torch

class Config:
    # Dataset paths
    DATA_DIR = './data/raw'
    PROCESSED_DIR = './data/processed'
    SAMPLE_DIR = './data/sample_images'
    
    # Model settings
    MODEL_NAME = 'resnet18'
    NUM_CLASSES = 4  # Acne, Dark Circles, Redness, Wrinkles
    PRETRAINED = True
    
    # Training hyperparameters
    BATCH_SIZE = 16  # Small batch for Mac
    NUM_EPOCHS = 7
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # Image settings
    IMAGE_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Data split
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    
    # Device configuration (Mac optimized)
    DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Model save paths
    WEIGHTS_DIR = './models/saved_weights'
    BEST_MODEL_PATH = './models/saved_weights/best_model.pth'
    
    # Output paths
    OUTPUT_DIR = './outputs'
    VIZ_DIR = './outputs/visualizations'
    LOG_DIR = './outputs/logs'
    
    # Class labels
    CONCERN_LABELS = ['acne', 'dark_circles', 'redness', 'wrinkles']
    
    # Threshold for binary classification
    THRESHOLD = 0.7
    
    # GradCAM settings
    TARGET_LAYER = 'layer4'  # For ResNet
    
    # Face detection settings
    FACE_MARGIN = 20  # Margin around detected face
    MIN_FACE_SIZE = 40
    
    # Training settings
    NUM_WORKERS = 2  # Reduced for Mac
    EARLY_STOPPING_PATIENCE = 7
    SAVE_FREQUENCY = 5
    
    # Augmentation settings
    RANDOM_ROTATION = 10
    COLOR_JITTER = 0.2
    HORIZONTAL_FLIP = 0.5
