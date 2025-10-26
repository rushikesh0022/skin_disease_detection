# models/resnet_model.py
"""
ResNet-based multi-label classifier for skin concern detection
"""

import torch
import torch.nn as nn
from torchvision import models


class SkinConcernDetector(nn.Module):
    """
    Multi-label classifier for detecting skin concerns
    Based on ResNet18 architecture
    """
    
    def __init__(self, num_classes=4, pretrained=True, dropout=0.5):
        """
        Args:
            num_classes: Number of skin concerns to detect
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout rate
        """
        super(SkinConcernDetector, self).__init__()
        
        # Load pretrained ResNet18
        if pretrained:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet18(weights=None)
        
        # Get number of features from last layer
        num_features = self.backbone.fc.in_features
        
        # Replace classifier with multi-label head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        # Sigmoid for multi-label classification
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, 3, 224, 224]
            
        Returns:
            Output probabilities [batch_size, num_classes]
        """
        logits = self.backbone(x)
        probs = self.sigmoid(logits)
        return probs
    
    def get_features(self, x):
        """
        Extract features before final classifier
        Useful for GradCAM
        """
        # Forward through all layers except fc
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        return x


class EfficientNetDetector(nn.Module):
    """
    Alternative: EfficientNet-B0 based detector
    """
    
    def __init__(self, num_classes=4, pretrained=True, dropout=0.5):
        super(EfficientNetDetector, self).__init__()
        
        # Load EfficientNet
        if pretrained:
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.efficientnet_b0(weights=None)
        
        # Get number of features
        num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.backbone(x)


def create_model(model_name='resnet18', num_classes=4, pretrained=True):
    """
    Factory function to create model
    
    Args:
        model_name: 'resnet18' or 'efficientnet_b0'
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        
    Returns:
        Model instance
    """
    if model_name == 'resnet18':
        return SkinConcernDetector(num_classes, pretrained)
    elif model_name == 'efficientnet_b0':
        return EfficientNetDetector(num_classes, pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")


class MultiLabelLoss(nn.Module):
    """
    Binary Cross Entropy Loss for multi-label classification
    """
    
    def __init__(self, pos_weight=None):
        """
        Args:
            pos_weight: Weight for positive samples (to handle imbalance)
        """
        super(MultiLabelLoss, self).__init__()
        self.criterion = nn.BCELoss(weight=pos_weight)
    
    def forward(self, predictions, targets):
        """
        Calculate loss
        
        Args:
            predictions: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size, num_classes]
            
        Returns:
            Loss value
        """
        return self.criterion(predictions, targets)


def freeze_backbone(model, freeze=True):
    """
    Freeze/unfreeze backbone layers for transfer learning
    
    Args:
        model: SkinConcernDetector instance
        freeze: If True, freeze backbone parameters
    """
    for param in model.backbone.parameters():
        param.requires_grad = not freeze
    
    # Always keep final classifier trainable
    for param in model.backbone.fc.parameters():
        param.requires_grad = True
    
    if freeze:
        print("Backbone frozen. Only training final classifier.")
    else:
        print("Full model training enabled.")
