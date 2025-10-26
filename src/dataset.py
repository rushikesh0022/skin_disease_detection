# src/dataset.py
"""
Custom Dataset class for skin concern detection
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SkinConcernDataset(Dataset):
    """
    Dataset for multi-label skin concern classification
    """
    
    def __init__(self, data_dir, annotations_file=None, transform=None, 
                 preprocess=True):
        """
        Args:
            data_dir: Directory with all images
            annotations_file: CSV with image names and labels
            transform: Optional transform to be applied
            preprocess: Whether to apply preprocessing
        """
        self.data_dir = data_dir
        self.transform = transform
        self.preprocess = preprocess
        
        # Load annotations if provided
        if annotations_file and os.path.exists(annotations_file):
            self.annotations = pd.read_csv(annotations_file)
            self.image_names = self.annotations['image_name'].tolist()
            # Assuming columns: image_name, acne, dark_circles, redness, wrinkles
            self.labels = self.annotations[['acne', 'dark_circles', 
                                           'redness', 'wrinkles']].values
        else:
            # Load all images from directory
            self.image_names = [f for f in os.listdir(data_dir) 
                               if f.endswith(('.jpg', '.png', '.jpeg'))]
            self.labels = None
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        """
        Get item by index
        
        Returns:
            image: Preprocessed image tensor
            label: Multi-label tensor (if available)
        """
        # Load image
        img_name = self.image_names[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label if available
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label
        else:
            return image, img_name


def create_annotation_csv(data_dir, output_file):
    """
    Create annotation CSV from organized folder structure
    
    Expected structure:
        data_dir/
            acne/
            dark_circles/
            redness/
            wrinkles/
    
    Args:
        data_dir: Root directory with concern folders
        output_file: Output CSV file path
    """
    concerns = ['acne', 'dark_circles', 'redness', 'wrinkles']
    
    data = []
    for concern in concerns:
        concern_dir = os.path.join(data_dir, concern)
        if os.path.exists(concern_dir):
            for img_name in os.listdir(concern_dir):
                if img_name.endswith(('.jpg', '.png', '.jpeg')):
                    # Create one-hot encoding
                    labels = {c: 1 if c == concern else 0 for c in concerns}
                    data.append({
                        'image_name': img_name,
                        'image_path': os.path.join(concern_dir, img_name),
                        **labels
                    })
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Created annotation file: {output_file}")
    print(f"Total images: {len(df)}")


def split_dataset(annotations_file, train_ratio=0.8, val_ratio=0.1):
    """
    Split dataset into train/val/test sets
    
    Args:
        annotations_file: Path to annotations CSV
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        
    Returns:
        train_df, val_df, test_df
    """
    df = pd.read_csv(annotations_file)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    
    # Save splits
    base_name = annotations_file.replace('.csv', '')
    train_df.to_csv(f'{base_name}_train.csv', index=False)
    val_df.to_csv(f'{base_name}_val.csv', index=False)
    test_df.to_csv(f'{base_name}_test.csv', index=False)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df


class KaggleDatasetAdapter:
    """
    Adapter for multiple Kaggle datasets
    """
    
    @staticmethod
    def download_datasets():
        """
        Download both Kaggle datasets using kagglehub
        
        Returns:
            paths: Dictionary with dataset paths
        """
        try:
            import kagglehub
        except ImportError:
            print("❌ kagglehub not found. Installing...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'kagglehub'])
            import kagglehub
        
        print("🚀 Starting dataset download...")
        print("This may take several minutes depending on internet speed...")
        
        # Download Acne-Wrinkles-Spots dataset
        print("\n📥 Downloading Acne-Wrinkles-Spots Classification dataset...")
        try:
            acne_spots_path = kagglehub.dataset_download(
                "ranvijaybalbir/acne-wrinkles-spots-classification"
            )
            print(f"✅ Acne-Wrinkles-Spots downloaded to: {acne_spots_path}")
        except Exception as e:
            print(f"⚠️  Warning: Could not download Acne-Wrinkles-Spots dataset: {e}")
            acne_spots_path = None
        
        # Download Skin Defects dataset  
        print("\n📥 Downloading Skin Defects (Acne, Redness, Bags) dataset...")
        try:
            skin_defects_path = kagglehub.dataset_download(
                "trainingdatapro/skin-defects-acne-redness-and-bags-under-the-eyes"
            )
            print(f"✅ Skin Defects downloaded to: {skin_defects_path}")
        except Exception as e:
            print(f"⚠️  Warning: Could not download Skin Defects dataset: {e}")
            skin_defects_path = None
        
        if not acne_spots_path and not skin_defects_path:
            raise Exception("❌ Failed to download any datasets. Please check your Kaggle API credentials.")
        
        return {
            'acne_spots': acne_spots_path,
            'skin_defects': skin_defects_path
        }
    
    @staticmethod
    def prepare_combined_dataset(output_dir):
        """
        Prepare combined dataset from both Kaggle sources
        
        Args:
            output_dir: Output directory for processed data
            
        Returns:
            Path to combined annotations CSV
        """
        import shutil
        
        # Download datasets
        dataset_paths = KaggleDatasetAdapter.download_datasets()
        
        os.makedirs(output_dir, exist_ok=True)
        annotations = []
        
        # Process Acne-Wrinkles-Spots dataset
        print("\n🔄 Processing Acne-Wrinkles-Spots dataset...")
        acne_spots_dir = dataset_paths['acne_spots']
        
        # Map categories for first dataset
        acne_spots_mapping = {
            'acne': 'acne',
            'wrinkles': 'wrinkles', 
            'spots': 'redness'  # Map spots to redness
        }
        
        for kaggle_cat, our_cat in acne_spots_mapping.items():
            # Check nested structure: category/category/files
            src_dir = os.path.join(acne_spots_dir, kaggle_cat, kaggle_cat)
            if not os.path.exists(src_dir):
                src_dir = os.path.join(acne_spots_dir, kaggle_cat)
            
            if os.path.exists(src_dir):
                for img_file in os.listdir(src_dir):
                    if img_file.endswith(('.jpg', '.png', '.jpeg')):
                        # Create unique filename
                        unique_name = f"dataset1_{kaggle_cat}_{img_file}"
                        
                        # Copy to processed directory
                        src_path = os.path.join(src_dir, img_file)
                        dst_path = os.path.join(output_dir, unique_name)
                        shutil.copy2(src_path, dst_path)
                        
                        # Create annotation
                        labels = {
                            'acne': 1 if our_cat == 'acne' else 0,
                            'dark_circles': 0,  # Not in this dataset
                            'redness': 1 if our_cat == 'redness' else 0,
                            'wrinkles': 1 if our_cat == 'wrinkles' else 0
                        }
                        
                        annotations.append({
                            'image_name': unique_name,
                            'source': 'acne_spots',
                            **labels
                        })
        
        # Process Skin Defects dataset
        print("🔄 Processing Skin Defects dataset...")
        skin_defects_dir = dataset_paths['skin_defects']
        
        # Map categories for second dataset
        skin_defects_mapping = {
            'acne': 'acne',
            'redness': 'redness',
            'bags_under_eyes': 'dark_circles'  # Map bags to dark circles
        }
        
        for kaggle_cat, our_cat in skin_defects_mapping.items():
            # Check the files/ directory structure
            base_files_dir = os.path.join(skin_defects_dir, 'files', kaggle_cat.replace('_under_eyes', '').replace('bags_', 'bags'))
            
            if os.path.exists(base_files_dir):
                # Walk through all subdirectories to find images
                for root, dirs, files in os.walk(base_files_dir):
                    for img_file in files:
                        if img_file.endswith(('.jpg', '.png', '.jpeg')):
                            # Create unique filename with subdirectory info
                            rel_path = os.path.relpath(root, base_files_dir)
                            unique_name = f"dataset2_{kaggle_cat}_{rel_path.replace('/', '_')}_{img_file}"
                            
                            # Copy to processed directory
                            src_path = os.path.join(root, img_file)
                            dst_path = os.path.join(output_dir, unique_name)
                            shutil.copy2(src_path, dst_path)
                        
                        # Create annotation
                        labels = {
                            'acne': 1 if our_cat == 'acne' else 0,
                            'dark_circles': 1 if our_cat == 'dark_circles' else 0,
                            'redness': 1 if our_cat == 'redness' else 0,
                            'wrinkles': 0  # Not in this dataset
                        }
                        
                        annotations.append({
                            'image_name': unique_name,
                            'source': 'skin_defects',
                            **labels
                        })
        
        # Save combined annotations
        df = pd.DataFrame(annotations)
        output_csv = os.path.join(output_dir, 'combined_annotations.csv')
        df.to_csv(output_csv, index=False)
        
        # Print statistics
        print(f"\n📊 Dataset Statistics:")
        print(f"Total images: {len(df)}")
        print(f"From Acne-Wrinkles-Spots: {len(df[df['source'] == 'acne_spots'])}")
        print(f"From Skin Defects: {len(df[df['source'] == 'skin_defects'])}")
        print(f"\nConcern distribution:")
        for concern in ['acne', 'dark_circles', 'redness', 'wrinkles']:
            count = df[concern].sum()
            print(f"  {concern}: {count} images")
        print(f"✓ Combined dataset saved to: {output_dir}")
        
        return output_csv
    
    @staticmethod
    def prepare_dataset(kaggle_dir, output_dir):
        """
        Legacy method for backward compatibility
        Redirects to combined dataset preparation
        """
        return KaggleDatasetAdapter.prepare_combined_dataset(output_dir)
