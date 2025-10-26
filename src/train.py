# src/train.py
"""
Training script for Face Concern Detector
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet_model import SkinConcernDetector, MultiLabelLoss
from src.dataset import SkinConcernDataset
from src.preprocessing import create_transforms
from src.config import Config


class Trainer:
    """Training manager"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        
        # Initialize model
        self.model = SkinConcernDetector(
            num_classes=config.NUM_CLASSES,
            pretrained=config.PRETRAINED
        ).to(self.device)
        
        # Loss and optimizer
        self.criterion = MultiLabelLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc='Training')
        
        for images, labels in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(train_loader)
        return epoch_loss
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validation'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        val_loss = running_loss / len(val_loader)
        
        # Calculate metrics
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        metrics = self.calculate_metrics(all_preds, all_labels)
        
        return val_loss, metrics
    
    def calculate_metrics(self, predictions, labels, threshold=0.5):
        """Calculate evaluation metrics"""
        # Binary predictions
        pred_binary = (predictions > threshold).astype(int)
        
        # Per-class accuracy
        accuracy = (pred_binary == labels).mean(axis=0)
        
        # Overall metrics
        overall_acc = (pred_binary == labels).mean()
        
        metrics = {
            'overall_accuracy': overall_acc,
            'per_class_accuracy': accuracy,
            'predictions': predictions,
            'labels': labels
        }
        
        return metrics
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        os.makedirs(self.config.WEIGHTS_DIR, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        # Save latest
        torch.save(checkpoint, 
                  os.path.join(self.config.WEIGHTS_DIR, 'latest_model.pth'))
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.config.BEST_MODEL_PATH)
            print(f"Best model saved with val_loss: {val_loss:.4f}")
    
    def plot_losses(self):
        """Plot training curves"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        plt.savefig(os.path.join(self.config.OUTPUT_DIR, 'training_curves.png'))
        plt.close()
    
    def train(self, train_loader, val_loader, num_epochs):
        """Full training loop"""
        print(f"Training on device: {self.device}")
        print(f"Number of epochs: {num_epochs}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, metrics = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Print stats
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
        
        # Plot final curves
        self.plot_losses()
        print("\nTraining completed!")


def main():
    """Main training function"""
    config = Config()
    
    # Create data loaders
    train_transform = create_transforms(train=True, image_size=config.IMAGE_SIZE)
    val_transform = create_transforms(train=False, image_size=config.IMAGE_SIZE)
    
    # Load datasets
    train_dataset = SkinConcernDataset(
        data_dir=config.PROCESSED_DIR,
        annotations_file=os.path.join(config.PROCESSED_DIR, 'combined_annotations_train.csv'),
        transform=train_transform
    )
    
    val_dataset = SkinConcernDataset(
        data_dir=config.PROCESSED_DIR,
        annotations_file=os.path.join(config.PROCESSED_DIR, 'combined_annotations_val.csv'),
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Train model
    trainer.train(train_loader, val_loader, config.NUM_EPOCHS)


if __name__ == '__main__':
    main()
